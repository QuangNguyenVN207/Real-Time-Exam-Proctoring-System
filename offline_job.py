"""Chạy `run_offline.py` ở **tiến trình con** để giao diện Streamlit theo dõi được tiến trình.

Vì sao không chạy trong luồng nội bộ: app realtime đã nạp sẵn 4 bộ model (~3GB). Chạy offline trong
cùng tiến trình là nạp thêm bộ thứ hai vào **cùng một process** → máy 16GB chạm swap, kernel OOM-kill
giết luôn Streamlit (đo 2026-10-28: app chết ở frame 270/8s, `oom_kill_process` đúng lúc đó).
Tiến trình con cô lập RAM/GIL: UI vẫn phản hồi, job chết thì chỉ job mất, thoát là OS thu hồi sạch.

Tiến trình con in tiến độ dạng 1 dòng JSON có tiền tố `@@PROGRESS@@` (xem `run_offline._jsonl_progress`)
và tự ghi `preview.jpg` mỗi 3s — cha không cần truyền frame qua pipe.

Module này **không** import streamlit.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import run_offline

__all__ = ["OfflineJob"]

ROOT = Path(__file__).resolve().parent
PROGRESS_PREFIX = run_offline.PROGRESS_PREFIX
_TAIL_LINES = 25          # số dòng stdout cuối giữ lại để báo lỗi
_CANCEL_GRACE_S = 5.0

_LIVE: set["OfflineJob"] = set()


@atexit.register
def _stop_all() -> None:
    """Streamlit tắt thì không để lại tiến trình offline mồ côi."""
    for job in list(_LIVE):
        job.cancel()


class OfflineJob:
    """Một lần chạy offline: argv cho `run_offline.py`, đọc tiến độ qua stdout của tiến trình con."""

    def __init__(self, argv: list[str], out_dir: Path, source_video: Path) -> None:
        self.argv = [str(a) for a in argv]
        self.out_dir = Path(out_dir)
        self.source_video = Path(source_video)
        self.preview_path = self.out_dir / "preview.jpg"

        self.status = "queued"          # queued | running | done | error | cancelled
        self.stage = ""
        self.detail = ""
        self.frame_id = 0
        self.total_frames = 0
        self.alerts = 0
        self.logs: list[str] = []
        self.error: Optional[str] = None
        self.elapsed = 0.0
        self.playback_path: Optional[Path] = None
        self.report: dict[str, Any] = {}

        self._lock = threading.Lock()
        self._tail: list[str] = []
        self._cancel = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "OfflineJob":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        argv = list(self.argv)
        if "--progress-jsonl" not in argv:
            argv.append("--progress-jsonl")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.status = "running"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "run_offline.py"), *argv],
                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error = f"Không khởi chạy được tiến trình offline: {e}"
            return self
        _LIVE.add(self)
        self._thread = threading.Thread(target=self._watch, name="offline-job", daemon=True)
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        """Chờ chạy xong (dùng trong test); UI bình thường không cần gọi."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def cancel(self) -> None:
        """Dừng job: SIGTERM, quá 5s thì SIGKILL."""
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=_CANCEL_GRACE_S)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ------------------------------------------------------------------ giám sát tiến trình con
    def _watch(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        self._apply_event(json.loads(line[len(PROGRESS_PREFIX):]))
                    except Exception as e:
                        print(f"[OFFLINE-UI] bỏ qua dòng tiến độ lỗi: {e}")
                    continue
                with self._lock:
                    self._tail = (self._tail + [line])[-_TAIL_LINES:]
                if line:
                    print(line, flush=True)   # vẫn đẩy log của job ra console Streamlit
            code = proc.wait()
        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error = f"Lỗi đọc tiến trình offline: {e}"
            _LIVE.discard(self)
            return
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        if self._cancel.is_set():
            with self._lock:
                self.status = "cancelled"
                self.error = "Đã huỷ theo yêu cầu."
            _LIVE.discard(self)
            return
        if code != 0:
            with self._lock:
                self.status = "error"
                tail = "\n".join(self._tail[-6:])
                self.error = f"run_offline thoát với mã {code}." + (f"\n{tail}" if tail else "")
            _LIVE.discard(self)
            return
        try:
            self._finalize()
        finally:
            _LIVE.discard(self)

    def _finalize(self) -> None:
        """Remux sang H.264 cho browser + nạp `report.json` để UI tóm tắt."""
        annotated = self.out_dir / "annotated.mp4"
        playback = None
        if annotated.exists():
            playback = run_offline.to_h264(annotated, self.out_dir / "playback_h264.mp4")
        report_path = self.out_dir / "report.json"
        report = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[OFFLINE-UI] Không đọc được report.json: {e}")
        with self._lock:
            self.playback_path = playback
            self.report = report
            self.status = "done"
            self.stage = "done"
            self.detail = ""

    # ------------------------------------------------------------------ tiến độ
    def _apply_event(self, event: dict[str, Any]) -> None:
        stage = event.get("stage", "")
        with self._lock:
            self.stage = stage
            if event.get("detail"):
                self.detail = str(event["detail"])
            if stage == "video":
                self.total_frames = int(event.get("total_frames") or 0)
            elif stage == "frame":
                self.detail = ""            # dòng "đang nạp audio" không còn đúng khi đã chạy frame
                self.frame_id = int(event.get("frame_id") or 0)
                self.alerts = int(event.get("alerts") or 0)
                self.logs = list(event.get("logs") or [])
            elif stage == "error":
                self.error = str(event.get("error") or "lỗi không rõ")
            elif stage == "done":
                self.elapsed = float(event.get("elapsed") or 0.0)
                self.total_frames = int(event.get("frames") or self.total_frames)
                self.frame_id = self.total_frames

    # ------------------------------------------------------------------ đọc cho UI
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            done = self.total_frames or 1
            return {
                "status": self.status,
                "stage": self.stage,
                "detail": self.detail,
                "frame_id": self.frame_id,
                "total_frames": self.total_frames,
                "fraction": min(1.0, self.frame_id / done) if self.status == "running" else (
                    1.0 if self.status == "done" else 0.0),
                "alerts": self.alerts,
                "logs": list(self.logs),
                "error": self.error,
                "elapsed": self.elapsed,
                "preview": str(self.preview_path) if self.preview_path.exists() else None,
                "playback": str(self.playback_path) if self.playback_path and Path(self.playback_path).exists() else None,
                "out_dir": str(self.out_dir),
                "report": dict(self.report),
                "tail": list(self._tail[-8:]),
            }
