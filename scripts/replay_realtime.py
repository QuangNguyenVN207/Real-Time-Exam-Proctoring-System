"""Đo hiệu năng luồng realtime bằng 1 file video — không cần camera, không cần Streamlit.

Dựng lại đúng cấu trúc của `app_webrtc.py`: 1 luồng phát frame (đóng vai callback WebRTC) đẩy ảnh vào
3 queue `maxsize=2` (rớt frame cũ nhất) và các luồng AI tiêu thụ, rồi in số đo của `pipeline.metrics`:

    .venv/bin/python scripts/replay_realtime.py data/IMG_7817.MOV
    .venv/bin/python scripts/replay_realtime.py data/IMG_7817.MOV --seconds 30 --resize 960x540
    .venv/bin/python scripts/replay_realtime.py data/IMG_7817.MOV --fast --modules object,face,action

Mặc định đẩy frame giống app hiện tại: **cổng thời gian** 8 fps cho hành vi, 10 fps cho YOLO+FaceNet,
mỗi 5 frame cho giấy, và MediaPipe Holistic chạy ở **tiến trình riêng** (như `app_webrtc.py`).

`--fast` đọc video nhanh nhất có thể (đo trần năng lực từng module), mặc định phát đúng fps của video
để thấy fps thật và số frame bị rớt như khi camera chạy. `--encode` thêm stage encode VP8 y như aiortc
(đo độ mượt hình mà người dùng thật sự cảm nhận), `--no-isolate` để Holistic chạy tại chỗ (cột "trước"
khi A/B tối ưu 2026-10-28).

Số đo ở đây dùng để A/B: chạy trước và sau một thay đổi, so cột p50/p95 và cột rớt. Không dùng để
kết luận độ chính xác — muốn kiểm cảnh báo thì chạy `run_offline.py` và so `events.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TORCH_SHOW_CPP_WARNINGS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import numpy as np
import av

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.ai_services.object_detect.object_detect import (
    ObjectDetector,
    ObjectDetectModule,
    load_phone_confuser,
)
from backend.ai_services.object_detect.paper_count_pipeline import PaperCountPipeline
from backend.ai_services.pipeline import events, face_scan, gaze_scan, object_scan, paper_scan
from backend.ai_services.pipeline import overlays as overlay_draw
from backend.ai_services.pipeline.metrics import METRICS
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.core.device import configure_cpu_threads, resolve_model_path, select_device

# Cùng ngân sách thread với app: 4 model chạy song song, mỗi cái không được chiếm hết core.
configure_cpu_threads()

ALL_MODULES = ("object", "face", "paper", "action")
GAZE_TARGET_FPS = 8.0  # app: cổng thời gian GAZE_MIN_INTERVAL, khớp temporal_policy target_fps=8
VISION_TARGET_FPS = 10.0  # app: cổng thời gian VISION_MIN_INTERVAL (YOLO + FaceNet)
PAPER_EVERY = 5       # app: frame_count % 5
QUEUE_SIZE = 2
OVERLAY_TTL = 1.5
LABELS = {
    "callback": "callback (bơm frame + vẽ)",
    "encode": "VP8 encode (aiortc)",
    "object": "YOLO vật cấm",
    "face": "FaceNet người lạ",
    "paper": "Đếm giấy",
    "action": "Tracking + action",
    "draw": "Vẽ overlay",
}


def _make_encoder(width: int, height: int, fps: float, time_base=None):
    """Bộ encode y như aiortc dùng cho WebRTC (VP8, deadline=realtime)."""
    from fractions import Fraction

    ctx = av.CodecContext.create("libvpx", "w")
    ctx.width, ctx.height = int(width), int(height)
    ctx.pix_fmt = "yuv420p"
    ctx.time_base = time_base or Fraction(1, max(1, int(round(fps)) or 30))
    ctx.framerate = max(1, int(round(fps)) or 30)
    ctx.options = {"deadline": "realtime", "cpu-used": "8"}
    ctx.open()
    return ctx


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--seconds", type=float, default=20.0, help="số giây video đem chạy (0 = hết video)")
    p.add_argument("--fast", action="store_true", help="không giả lập nhịp camera, đọc nhanh nhất có thể")
    p.add_argument("--modules", default=",".join(ALL_MODULES),
                   help=f"module bật, mặc định tất cả: {','.join(ALL_MODULES)}")
    p.add_argument("--resize", default="", help="thu frame về WxH trước khi đẩy vào queue (vd 1280x720)")
    p.add_argument("--no-isolate", action="store_true",
                   help="chạy MediaPipe Holistic trong tiến trình chính (mặc định: tách tiến trình như app)")
    p.add_argument("--encode", action="store_true",
                   help="encode VP8 mỗi frame như aiortc (đo độ mượt hình thật, không chỉ độ trễ module)")
    p.add_argument("--json", type=Path, default=None, help="ghi snapshot số đo ra file JSON để so sánh")
    return p.parse_args(argv)


def _vision_profile():
    """Trùng với bảng `_DEVICE_PROFILE` trong app_webrtc.py."""
    device = select_device()
    vision_path = resolve_model_path(ROOT / "weights" / "best.pt", device)
    vision_device = {"cuda": None, "intel": "GPU"}.get(device.kind, "cpu")
    person_device = "auto" if device.kind == "cuda" else "cpu"
    return device, str(vision_path), vision_device, person_device


class Modules:
    """Nạp model một lần, giữ đúng tham số app realtime dùng."""

    def __init__(self, enabled: set[str], args: argparse.Namespace) -> None:
        self.enabled = enabled
        self.args = args
        device, vision_path, vision_device, person_device = _vision_profile()
        print(f"[DEVICE] {device.kind}: {device.name}")
        self.tracking_dir = Path(tempfile.mkdtemp(prefix="replay_tracking_"))

        if "object" in enabled:
            self.yolo = ObjectDetector(model_path=vision_path, device=vision_device, confidence_threshold=0.7)
            self.phone_confuser = load_phone_confuser(vision_device)
        if "face" in enabled:
            self.face = FaceVerifier(db_path=str(ROOT / "data" / "student_faces"))
        if "paper" in enabled:
            self.paper_pipeline = self._build_paper(person_device)
        self.gaze_model = None
        if "action" in enabled:
            self.stack = gaze_scan.build_stack(
                device=person_device,
                live_pair=["student_01:student_02"],
                isolate_holistic=not args.no_isolate,
            )
            self.gaze_model = self.stack   # khung xương vẽ lại từ latest_packet/latest_results

    def _build_paper(self, person_device: str) -> PaperCountPipeline:
        person = UltralyticsPersonDetector(
            model_path=str(ROOT / "weights" / "yolov8n.pt"), confidence_threshold=0.55, device=person_device)
        detector = ObjectDetectModule(device=None)  # auto -> OV/iGPU (A/B 2026-10-29)
        pipeline = PaperCountPipeline(
            person_detector=person, object_detector=detector, storage_root=self.tracking_dir, max_people=2)
        # Giấy chỉ đếm giấy: tắt fallback smartphone y như app realtime
        if hasattr(detector, "_smartphone_fallback_enabled"):
            detector._smartphone_fallback_enabled = False
        if hasattr(detector, "_auxiliary_class_ids"):
            detector._auxiliary_class_ids = []
        if hasattr(pipeline, "enable_smartphone_detection"):
            pipeline.enable_smartphone_detection = False
        self.paper_session = f"replay_{int(time.time())}"
        pipeline.create_session(self.paper_session, restore_existing=False)
        return pipeline


class Sink:
    """Thay cho RESULT_QUEUE + OVERLAY_LOCK của app: đếm cảnh báo theo module, giữ overlay để vẽ."""

    def __init__(self) -> None:
        self.alerts: dict[str, int] = {}
        self.overlays: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.frame_count = 0   # app: SHARED_STATE["frame_count"], luồng giấy đọc lúc xử lý

    def add(self, alert: Optional[dict[str, Any]]) -> None:
        if alert is None:
            return
        module = alert.get("module", "?")
        self.alerts[module] = self.alerts.get(module, 0) + 1
        with self.lock:
            events.prune_overlays(self.overlays, alert)


def _consumer(name: str, src: queue.Queue, stop: threading.Event, fn) -> None:
    while not stop.is_set():
        try:
            data = src.get(timeout=0.2)
        except queue.Empty:
            continue
        if data is None:
            return
        frame, ts = data
        try:
            fn(frame, ts)
        except Exception as exc:  # một frame lỗi không được làm chết luồng như app
            import traceback
            print(f"[LỖI {name}] {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)


def build_consumers(mods: Modules, sink: Sink, queues: dict[str, queue.Queue]) -> dict[str, Any]:
    def heavy(frame: np.ndarray, ts: float) -> None:
        if "object" in mods.enabled:
            with METRICS.time("object"):
                alert = object_scan.scan_objects(mods.yolo, frame, ts, getattr(mods, "phone_confuser", None))
            sink.add(alert)
        if "face" in mods.enabled:
            with METRICS.time("face"):
                alert, _ = face_scan.scan_stranger_with_masking(mods.face, frame, ts)
            sink.add(alert)

    def paper(frame: np.ndarray, ts: float) -> None:
        with METRICS.time("paper"):
            result = mods.paper_pipeline.process_frame(
                frame, session_id=mods.paper_session,
                frame_id=sink.frame_count, timestamp_ms=int(ts * 1000))
        sink.add(paper_scan.build_alert(result, ts))

    def gaze(frame: np.ndarray, ts: float) -> None:
        with METRICS.time("action"):
            alerts = gaze_scan.scan_frame(mods.stack, frame, ts)
        for alert in alerts:
            sink.add(alert)

    consumers: dict[str, Any] = {}
    if "object" in mods.enabled or "face" in mods.enabled:
        consumers["vision"] = (queues["vision"], heavy)
    if "paper" in mods.enabled:
        consumers["paper"] = (queues["paper"], paper)
    if "action" in mods.enabled:
        consumers["gaze"] = (queues["gaze"], gaze)
    return consumers


def open_video(path: Path, seconds: float, resize: str) -> tuple[cv2.VideoCapture, float, int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"[LỖI] Không mở được video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if seconds <= 0:
        limit = total or 10 ** 9
    else:
        limit = max(1, int(round(seconds * fps))) if not total else min(total, int(round(seconds * fps)))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target = (w, h)
    if resize:
        tw, th = (int(x) for x in resize.lower().split("x"))
        target = (tw, th)
    print(f"[VIDEO] {path.name}: {w}x{h} @ {fps:.1f}fps, chạy {limit}/{total} frame"
          + (f", resize -> {target[0]}x{target[1]}" if resize else ""))
    return cap, fps, limit, target


def push(q: queue.Queue, name: str, item: Any) -> None:
    """Rớt frame cũ nhất khi đầy — y hệt callback của app."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        METRICS.drop(f"queue.{name}")
    q.put(item)


def run(args: argparse.Namespace) -> dict[str, Any]:
    enabled = {m.strip() for m in args.modules.split(",") if m.strip()}
    unknown = enabled - set(ALL_MODULES)
    if unknown:
        raise SystemExit(f"[LỖI] Module không tồn tại: {', '.join(sorted(unknown))}")

    cap, fps, limit, size = open_video(args.video, args.seconds, args.resize)
    mods = Modules(enabled, args)
    sink = Sink()
    queues = {name: queue.Queue(maxsize=QUEUE_SIZE) for name in ("vision", "paper", "gaze")}
    stop = threading.Event()
    workers = []
    for name, (src, fn) in build_consumers(mods, sink, queues).items():
        t = threading.Thread(target=_consumer, args=(name, src, stop, fn), daemon=True, name=name)
        t.start()
        workers.append(t)

    print(f"[RUN] bắt đầu: {'nhịp thật' if not args.fast else 'nhanh nhất'} — Ctrl+C để dừng và in số đo")
    period = 0.0 if args.fast else 1.0 / fps
    gaze_period = 1.0 / GAZE_TARGET_FPS
    vision_period = 1.0 / VISION_TARGET_FPS
    encoder = _make_encoder(size[0], size[1], fps) if args.encode else None
    last_gaze = 0.0
    last_vision = 0.0
    next_at = time.perf_counter()
    started = time.perf_counter()
    frame_id = 0
    try:
        while frame_id < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            sink.frame_count = frame_id
            if size != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, size)

            now = time.time()
            with METRICS.time("callback"):
                # cổng thời gian cho hành vi: 8 fps, không phải mỗi frame (giống app)
                mono = time.perf_counter()
                if mono - last_gaze >= gaze_period:
                    last_gaze = mono
                    push(queues["gaze"], "gaze", (frame.copy(), now))
                if frame_id % PAPER_EVERY == 0:
                    push(queues["paper"], "paper", (frame.copy(), now))
                if mono - last_vision >= vision_period:
                    last_vision = mono
                    push(queues["vision"], "vision", (frame.copy(), now))
                with sink.lock:
                    events.expire_overlays(sink.overlays, now, OVERLAY_TTL)
                    snapshot = list(sink.overlays)
                with METRICS.time("draw"):
                    overlay_draw.draw(frame, snapshot, gaze_model=mods.gaze_model)
                if encoder is not None:
                    with METRICS.time("encode"):
                        encoder.encode(av.VideoFrame.from_ndarray(frame, format="bgr24"))

            if period:
                next_at += period
                lag = next_at - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                else:
                    next_at = time.perf_counter()  # không dồn nợ: giống camera bị tụt nhịp
    except KeyboardInterrupt:
        print("\n[RUN] dừng theo Ctrl+C")
    finally:
        cap.release()
        wall = time.perf_counter() - started
        stop.set()
        for q in queues.values():
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t in workers:
            t.join(timeout=5)
        stack = getattr(mods, "stack", None)
        if stack is not None:
            worker = stack.holistic
            print(f"[GAZE-WORKER] mode={getattr(worker, 'mode', 'in-process')} "
                  f"stats={getattr(worker, 'stats', {})}")
            try:
                stack.close()
            except Exception:
                pass

    snap = METRICS.snapshot()
    print(f"\n[KẾT QUẢ] {frame_id} frame trong {wall:.1f}s "
          f"({frame_id / wall:.1f} fps đầu vào){'' if args.fast else f' / nhịp video {fps:.1f} fps'}")
    print(f"{'module':<30}{'gọi':>7}{'fps':>8}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}{'rớt':>7}{'lỗi':>6}")
    for name, stat in snap.items():
        if name.startswith("queue."):
            continue
        print(f"{LABELS.get(name, name):<30}{stat['runs']:>7}{stat['fps']:>8.1f}{stat['p50_ms']:>9.1f}"
              f"{stat['p95_ms']:>9.1f}{stat['max_ms']:>9.1f}{stat['drops']:>7}{stat['errors']:>6}")
    drops = {name.replace("queue.", ""): stat["drops"]
             for name, stat in snap.items() if name.startswith("queue.") and stat["drops"]}
    print(f"[RỚT] {'không rớt frame nào' if not drops else ', '.join(f'{k}={v}' for k, v in drops.items())}")
    print(f"[CẢNH BÁO] {sink.alerts or 'không có'}")

    out = {"video": str(args.video), "frames": frame_id, "wall_s": round(wall, 2),
           "input_fps": round(frame_id / wall, 1), "stages": snap, "alerts": sink.alerts}
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[JSON] {args.json}")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.video.exists():
        raise SystemExit(f"[LỖI] Không thấy video: {args.video}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
