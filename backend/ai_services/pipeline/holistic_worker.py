"""Chạy MediaPipe Holistic trong một tiến trình RIÊNG.

Vì sao phải tách: MediaPipe Tasks (pybind11) giữ GIL suốt lần suy luận. Đo trên máy
RTX 4050 / WSL2 (28/10/2026): một `process_packet()` cho 2 track chặn GIL **100–113 ms**
(lần đầu, chưa warm: 886 ms). Callback WebRTC và script Streamlit đều là thread Python,
nên trong lúc đó chúng không được chạy. `streamlit_webrtc/process.py` khi không kịp xử lý
sẽ trả lại `_last_out_frame` (frame CŨ) cho trình duyệt → hình vừa giật vừa trễ dần.

Tách sang tiến trình con thì nó chậm bao nhiêu cũng không giành GIL với luồng video.
Hai bên nói chuyện bằng pickle qua socket AF_UNIX:

    cha  -> con : {"cmd": "init", "kwargs": {...}}      (một lần, lúc khởi động)
    con  -> cha : {"ok": True, "pid": ...}
    cha  -> con : {"cmd": "process", "frame": ndarray, "packet": TrackPacket}
    con  -> cha : {"ok": True, "results": tuple[TrackHolisticResult, ...]}
    cha  -> con : {"cmd": "quit"}

`TrackHolisticResult`/`LandmarkPoint`/`BoundingBox` đều là dataclass thuần nên pickle được;
không có object protobuf nào đi qua dây.

Nếu tiến trình con chết hoặc không trả lời trong `request_timeout`, `RemoteHolistic` tự
khởi động lại (tối đa `max_restarts` lần); quá số đó thì chạy Holistic ngay trong tiến trình
chính như cũ — luồng hành vi không bao giờ chết, chỉ chậm lại.

Chạy thử riêng (kiểm tra tiến trình con có sống không):

    .venv/bin/python -m backend.ai_services.pipeline.holistic_worker --selftest
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Iterable

__all__ = ["RemoteHolistic", "worker_main"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_AUTHKEY = b"proctor-holistic-worker"
_ADDRESS_ENV = "PROCTOR_HOLISTIC_ADDRESS"
#: Số core để dành cho app (callback + encode + Streamlit). MediaPipe spawn nhiều thread nên
#: nếu để nó dùng hết 16 core thì encode VP8 bị chậm 3x và hình giật.
_RESERVED_CORES_ENV = "PROCTOR_HOLISTIC_RESERVED_CORES"
_DEFAULT_RESERVED_CORES = 4

#: Ưu tiên CPU thấp hơn tiến trình chính: luồng video/encode phải được phục vụ trước.
DEFAULT_NICE = 10


def _address_family() -> str | None:
    return None if os.name == "nt" else "AF_UNIX"


def _make_address(tmpdir: str) -> str:
    if os.name == "nt":
        return r"\\.\pipe\proctor-holistic-" + uuid.uuid4().hex
    return os.path.join(tmpdir, "holistic.sock")


class RemoteHolistic:
    """Proxy cùng giao diện với `HolisticLandmarkExtractor` nhưng chạy ở tiến trình khác."""

    def __init__(
        self,
        *,
        extractor_kwargs: dict[str, Any] | None = None,
        request_timeout: float = 4.0,
        startup_timeout: float = 180.0,
        max_restarts: int = 3,
        nice: int = DEFAULT_NICE,
    ) -> None:
        self._kwargs = dict(extractor_kwargs or {})
        self._request_timeout = float(request_timeout)
        self._startup_timeout = float(startup_timeout)
        self._max_restarts = int(max_restarts)
        self._nice = int(nice)

        # Bản trong tiến trình chính: chỉ dùng để VẼ khung xương (không nạp model, không chặn GIL)
        # và làm đường lui khi tiến trình con chết.
        self._local = _new_extractor(self._kwargs)
        self._local_started = False

        self._proc: subprocess.Popen | None = None
        self._conn: Any = None
        self._listener: Listener | None = None
        self._tmpdir: str | None = None
        self._log_path: str | None = None
        self._restarts = 0
        self._last_error: str | None = None
        # Chỉ MỘT lần start() được chạy tại một thời điểm. Bản cũ submit self.start từ mỗi
        # tick lỗi -> hai tiến trình con dựng song song, cái sau đè `_conn` của cái trước
        # ("AttributeError: 'NoneType' object has no attribute 'recv'") rồi rơi về chạy
        # tại chỗ (giữ GIL -> hình đứng vài giây).
        self._start_lock = threading.Lock()
        self._restart_pending = False
        self._closed = False

        self.stats = {"remote": 0, "fallback": 0, "restarts": 0}

        atexit.register(self.close)

    # ------------------------------------------------------------------ vòng đời
    @property
    def alive(self) -> bool:
        return self._conn is not None and self._proc is not None and self._proc.poll() is None

    @property
    def mode(self) -> str:
        """`process` = đang tách tiến trình, `in-process` = đã rơi về chạy tại chỗ."""
        return "process" if self.alive else "in-process"

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> bool:
        """Khởi động tiến trình con. Trả False nếu thất bại (khi đó dùng đường lui)."""
        with self._start_lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        if self._closed:
            return False
        if self.alive:
            return True
        self._cleanup()

        if self._restarts > self._max_restarts:
            return False
        if self._restarts:
            self.stats["restarts"] += 1
        self._restarts += 1

        try:
            self._tmpdir = tempfile.mkdtemp(prefix="proctor_holistic_")
            address = _make_address(self._tmpdir)
            self._listener = Listener(address, family=_address_family(), authkey=_AUTHKEY)

            self._log_path = os.path.join(self._tmpdir, "worker.log")
            log_handle = open(self._log_path, "wb")
            env = dict(os.environ)
            env[_ADDRESS_ENV] = address
            env.setdefault(
                _RESERVED_CORES_ENV,
                os.environ.get(_RESERVED_CORES_ENV, str(_DEFAULT_RESERVED_CORES)),
            )
            # Tiến trình con phải mở được file model và import package từ gốc dự án.
            env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "backend.ai_services.pipeline.holistic_worker",
                 address, str(self._nice)],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
                env=env,
                start_new_session=True,
            )
            log_handle.close()

            # accept() chặn tới khi con kết nối -> đặt trong thread để còn timeout.
            with ThreadPoolExecutor(max_workers=1) as pool:
                self._conn = pool.submit(self._listener.accept).result(timeout=self._startup_timeout)

            self._conn.send({"cmd": "init", "kwargs": self._kwargs})
            reply = self._recv(self._startup_timeout)
            if not reply.get("ok"):
                raise RuntimeError(str(reply.get("error") or "init thất bại"))
            print(f"[HOLISTIC-WORKER] tiến trình riêng pid={self._proc.pid} đã sẵn sàng")
            return True
        except Exception as exc:  # thiếu mediapipe trong env con, hết RAM, timeout...
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[HOLISTIC-WORKER][WARNING] không tách được tiến trình ({self._last_error}) "
                  f"-> chạy Holistic ngay trong app")
            self._cleanup()
            return False

    def close(self) -> None:
        # Giữ khoá: nếu một lần restart đang chạy nền thì phải đợi nó xong, nếu không
        # `_cleanup()` sẽ đóng listener/conn giữa lúc `start()` đang bắt tay
        # ("AttributeError: 'NoneType' object has no attribute 'recv'").
        with self._start_lock:
            self._closed = True
            try:
                if self._conn is not None:
                    self._conn.send({"cmd": "quit"})
            except Exception:
                pass
            self._cleanup()
        try:
            if self._local_started:
                self._local.__exit__(None, None, None)
        except Exception:
            pass

    def __enter__(self) -> "RemoteHolistic":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ giao diện model
    def process_packet(self, frame: Any, packet: Any) -> tuple[Any, ...]:
        """Landmark cho mọi track đang hiện. Không bao giờ raise: lỗi thì trả () và đếm."""
        if self.alive:
            try:
                self._conn.send({"cmd": "process", "frame": frame, "packet": packet})
                reply = self._recv(self._request_timeout)
                if reply.get("ok"):
                    self.stats["remote"] += 1
                    return tuple(reply.get("results") or ())
                raise RuntimeError(str(reply.get("error") or "process thất bại"))
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                with self._start_lock:
                    self._cleanup()
                self._schedule_restart()
                # Tick này không có landmark là chấp nhận được; KHÔNG nạp model ngay trong
                # tiến trình chính (giữ GIL vài giây -> video đứng). Chỉ dùng đường lui tại
                # chỗ khi đã hết số lần restart.
                if self._restarts > self._max_restarts:
                    return self._fallback_process_packet(frame, packet)
                return ()
        # Chưa có tiến trình con (đang dựng lại / chưa từng dựng được). Trong lúc chờ thì trả
        # rỗng, KHÔNG nạp model vào tiến trình chính; chỉ chạy tại chỗ khi đã hết lượt restart.
        if self._restarts > self._max_restarts:
            return self._fallback_process_packet(frame, packet)
        self._schedule_restart()
        return ()

    def _schedule_restart(self) -> None:
        """Dựng lại tiến trình con ở thread nền, tối đa một lần đang chờ.

        Không chặn: `start()` giữ lock trong lúc chờ accept (tới `startup_timeout`), nên
        luồng gaze không được phép đứng chờ ở đây.
        """
        if not self._start_lock.acquire(blocking=False):
            return  # một lần start() khác đang chạy -> nó sẽ dựng xong
        try:
            if self._closed or self._restart_pending or self._restarts > self._max_restarts:
                return
            self._restart_pending = True
        finally:
            self._start_lock.release()

        def _run() -> None:
            try:
                self.start()
            finally:
                with self._start_lock:
                    self._restart_pending = False

        threading.Thread(target=_run, name="holistic-restart", daemon=True).start()

    def draw_results(self, frame: Any, results: Iterable[Any]) -> Any:
        """Vẽ khung xương: hàm thuần OpenCV, không đụng model."""
        return self._local.draw_results(frame, results)

    @property
    def task_input_size(self) -> int:
        return getattr(self._local, "task_input_size", 512)

    # ------------------------------------------------------------------ nội bộ
    def _fallback_process_packet(self, frame: Any, packet: Any) -> tuple[Any, ...]:
        self.stats["fallback"] += 1
        if not self._local_started:
            try:
                self._local.__enter__()
                self._local_started = True
                print("[HOLISTIC-WORKER] dùng Holistic tại chỗ (chia sẻ GIL với video)")
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return ()
        try:
            return tuple(self._local.process_packet(frame, packet))
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return ()

    def _recv(self, timeout: float) -> dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("chưa kết nối tiến trình con")
        if not self._conn.poll(timeout):
            raise TimeoutError(f"không có phản hồi sau {timeout:.1f}s")
        return self._conn.recv()

    def _cleanup(self) -> None:
        conn, proc, listener = self._conn, self._proc, self._listener
        self._conn = self._proc = self._listener = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


def _new_extractor(kwargs: dict[str, Any]) -> Any:
    from backend.ai_services.pose_gaze.pose_gaze.holistic.landmark import (
        HolisticLandmarkExtractor,
    )

    return HolisticLandmarkExtractor(**kwargs)


# ====================================================================== tiến trình con
def _pin_to_free_cores() -> None:
    """Giới hạn số core của tiến trình con để luôn còn core cho callback + encode VP8.

    MediaPipe Tasks spawn thread theo số core nhìn thấy. Đo trên RTX 4050/WSL2: khi nó dùng
    hết 16 core, encode 720p từ 7 ms tăng lên 26 ms và callback vượt hạn 50 ms của 20 fps.
    Đặt biến môi trường `PROCTOR_HOLISTIC_RESERVED_CORES` để đổi số core để dành (mặc định 4).
    """
    try:
        total = os.cpu_count() or 1
        reserved = int(os.environ.get(_RESERVED_CORES_ENV, _DEFAULT_RESERVED_CORES))
        usable = max(1, total - max(0, reserved))
        if usable >= total:
            return
        os.sched_setaffinity(0, set(range(usable)))
    except (AttributeError, OSError, ValueError):
        pass  # Windows / kernel không hỗ trợ: không sao, chỉ mất phần tối ưu


def worker_main(argv: list[str]) -> int:
    """Vòng lặp của tiến trình con. `argv = [address, nice]`."""
    if not argv:
        print("dùng: python -m backend.ai_services.pipeline.holistic_worker <address> [nice]")
        return 2

    address = argv[0]
    nice = int(argv[1]) if len(argv) > 1 else DEFAULT_NICE
    try:
        os.nice(nice)
    except (AttributeError, OSError):
        pass

    _pin_to_free_cores()

    conn = Client(address, family=_address_family(), authkey=_AUTHKEY)
    extractor: Any = None
    try:
        while True:
            message = conn.recv()
            cmd = message.get("cmd")
            if cmd == "quit":
                break
            if cmd == "init":
                try:
                    extractor = _new_extractor(dict(message.get("kwargs") or {}))
                    extractor.__enter__()
                    conn.send({"ok": True, "pid": os.getpid()})
                except Exception as exc:
                    conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    break
                continue
            if cmd == "process":
                if extractor is None:
                    conn.send({"ok": False, "error": "chưa init"})
                    continue
                try:
                    results = extractor.process_packet(message["frame"], message["packet"])
                    conn.send({"ok": True, "results": results})
                except Exception as exc:
                    conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                continue
            conn.send({"ok": False, "error": f"lệnh lạ: {cmd!r}"})
    except (EOFError, OSError, KeyboardInterrupt):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if extractor is not None:
            try:
                extractor.__exit__(None, None, None)
            except Exception:
                pass
    return 0


def _selftest() -> int:
    """Chạy thử 1 round-trip với một frame giả để chắc socket + pickle hoạt động."""
    import numpy as np

    from backend.ai_services.pose_gaze.tracking.schemas import (
        BoundingBox,
        TrackPacket,
        TrackedPerson,
    )

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    track = TrackedPerson(track_id=1, bbox=BoundingBox(80, 40, 600, 700), confidence=0.9,
                          age_frames=1, missed_frames=0, is_present=True, student_id="student_01")
    packet = TrackPacket("selftest", 1, int(time.time() * 1000), (track,))

    worker = RemoteHolistic(extractor_kwargs={"smooth_landmarks": True})
    t0 = time.perf_counter()
    ok = worker.start()
    print(f"[SELFTEST] start -> {ok} ({time.perf_counter() - t0:.1f}s), mode={worker.mode}")
    for i in range(3):
        t0 = time.perf_counter()
        results = worker.process_packet(frame, packet)
        print(f"[SELFTEST] tick {i}: {len(results)} kết quả trong {1000 * (time.perf_counter() - t0):.1f} ms")
    print(f"[SELFTEST] stats={worker.stats} error={worker.last_error}")
    worker.close()
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        raise SystemExit(_selftest())
    raise SystemExit(worker_main(sys.argv[1:]))
