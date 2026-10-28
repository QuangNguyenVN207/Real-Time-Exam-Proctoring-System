"""Luồng offline: nhận 1 file video -> xuất video đã vẽ cảnh báo + dữ liệu giám sát.

Chạy tuần tự (không thread, không micro, không WebRTC) nhưng gọi đúng các hàm phát hiện
trong package `backend.ai_services.pipeline` mà app_webrtc.py dùng, nên khung cảnh báo,
dòng log và cách cắt câu chữ giống bản realtime.

    python run_offline.py data/IMG_7817.MOV
    python run_offline.py data/IMG_7817.MOV --setup-seconds 5 --max-seconds 20 --stride-heavy 3

Sản phẩm trong --output (mặc định offline_out/<tên video>_<giờ>/):
    annotated.mp4  video đã vẽ khung cảnh báo + bảng cảnh báo góc phải, kèm soundtrack gốc
                   (remux bằng ffmpeg nếu máy có ffmpeg)
    events.jsonl   1 dòng = 1 cảnh báo (t, frame, module, status, actor_id, payload gốc)
    log.txt        dòng thời gian các cảnh báo, mốc [mm:ss] theo video
    report.json    thống kê theo module + theo actor, cấu hình, thời gian xử lý từng module
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

os.environ.setdefault("TORCH_SHOW_CPP_WARNINGS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import av
import cv2
import numpy as np

from backend.ai_services.face_verify.face_verify import FaceVerifier
from backend.ai_services.object_detect.object_detect import ObjectDetector, ObjectDetectModule
from backend.ai_services.object_detect.paper_count_pipeline import PaperCountPipeline
from backend.ai_services.pipeline import audio_scan, events, face_scan, gaze_scan, object_scan, paper_scan
from backend.ai_services.pipeline import overlays as overlay_draw
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.whisper.audio_pipeline import AudioPipeline

ALL_MODULES = ("object", "face", "paper", "pose", "action", "audio")
MAX_LOG_LINES = 20
_SIDEBAR_TIME_RE = re.compile(r"^\*\*\[\d{2}:\d{2}:\d{2}\]\*\*")


def mmss(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes:02d}:{rest:02d}"


def video_relative(line: str, t: float) -> str:
    """Sidebar realtime in giờ đồng hồ thật; với file quay thì mốc [mm:ss] theo video dễ đối chiếu hơn."""
    return _SIDEBAR_TIME_RE.sub(f"**[{mmss(t)}]**", line, count=1)


# ---------------------------------------------------------------- audio
def decode_audio(path: Path, target_sr: int, chunk_samples: int) -> Iterator[np.ndarray]:
    """Track audio của file video -> từng chunk float32 mono ở `target_sr`."""
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        carry = np.array([], dtype=np.float32)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                carry = np.concatenate((carry, out.to_ndarray().reshape(-1).astype(np.float32)))
                while carry.size >= chunk_samples:
                    yield carry[:chunk_samples]
                    carry = carry[chunk_samples:]
        if carry.size:
            yield carry


def transcribe_file_audio(path: Path, pipeline: AudioPipeline, args) -> tuple[list[dict], dict]:
    """Cắt câu bằng đúng DynamicSpeechChunker của luồng realtime -> alert audio theo mốc video."""
    stats = {"chunks": 0, "speech_segments": 0, "error": None}
    chunks = decode_audio(path, args.audio_sample_rate, args.audio_sample_rate // 2)
    first = next(chunks, None)
    if first is None:
        print("[AUDIO] ⚠️ File không có track audio (hoặc giải mã ra rỗng) -> bỏ qua giám sát âm thanh.")
        return [], stats

    chunker = audio_scan.DynamicSpeechChunker(
        args.audio_sample_rate, noise_profile_seconds=args.audio_noise_seconds
    )
    utterances: list[dict[str, Any]] = []
    for chunk in itertools.chain([first], chunks):
        stats["chunks"] += 1
        audio_16k = chunker.feed(chunk)
        if audio_16k is not None:
            _emit_utterance(pipeline, chunker, audio_16k, utterances, stats)
    tail = chunker.flush()  # câu cuối còn dở trong hộp chứa khi hết file
    if tail is not None:
        _emit_utterance(pipeline, chunker, tail, utterances, stats)
    return utterances, stats


def _emit_utterance(pipeline, chunker, audio_16k, utterances, stats) -> None:
    stats["speech_segments"] += 1
    start_s, end_s = chunker.last_span
    try:
        result = pipeline.process_audio(audio_16k, timestamp=time.time(), source="offline-file")
    except Exception as e:
        print(f"[AUDIO] Lỗi xử lý câu tại {mmss(start_s)}: {e}")
        return
    alert = audio_scan.to_alert(result)
    if alert is None:
        return
    alert["timestamp"] = start_s            # realtime dùng giờ đồng hồ, offline dùng mốc video
    alert["span_s"] = [start_s, end_s]
    utterances.append(alert)
    print(f"[AUDIO] {mmss(start_s)}  '{alert.get('transcription', '')}'  risk={alert.get('risk', '')}")


# ---------------------------------------------------------------- sink cảnh báo
@dataclass
class AlertSink:
    """Đóng vai trò video_frame_callback(): nhận alert, lọc overlay, tạo log, vẽ lên frame."""

    ttl: float = 1.5
    overlays: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    frame_id: int = 0
    gaze_model: Optional[Any] = None

    def add(self, alert: Optional[dict[str, Any]], t: float) -> None:
        if not alert:
            return
        alert = dict(alert)
        alert.setdefault("timestamp", t)
        alert["display_timestamp"] = t
        moment = float(alert["timestamp"])

        line = events.to_log_line(alert, now=moment)
        line = video_relative(line, moment) if line else ""
        self.events.append({
            "t": round(moment, 3),
            "frame": self.frame_id,
            "module": alert.get("module"),
            "status": alert.get("status"),
            "actor_id": _resolve_actor(alert, line),
            "log": line,
            "alert": alert,
        })

        if alert.get("module") == "system":
            if line:
                self.logs.insert(0, line)
            return

        events.prune_overlays(self.overlays, alert)
        if line:
            self.logs.insert(0, line)
            del self.logs[MAX_LOG_LINES:]

    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        events.expire_overlays(self.overlays, t, self.ttl)
        return overlay_draw.draw(
            frame, list(self.overlays), gaze_model=self.gaze_model, log_lines=self.logs[:8]
        )


def _resolve_actor(alert: dict[str, Any], line: str) -> Optional[str]:
    """Mọi số liệu giám sát phải khoá theo actor (video, actor_id) — không tính cho cả video.

    Alert hành vi đã có sẵn `actor_id`; alert giấy lấy owner của tờ giấy lạ, rồi mới tới
    tên `student_XX` in trong dòng log (face_verify không có định danh vì mặt lạ chưa đăng ký).
    """
    if alert.get("actor_id"):
        return str(alert["actor_id"])
    details = alert.get("details") or {}
    if details.get("actor_id"):
        return str(details["actor_id"])
    for paper in alert.get("papers") or []:
        if paper.get("status") == "suspicious_new_paper" and paper.get("owner_track_id") is not None:
            return f"student_{int(paper['owner_track_id']):02d}"
    return _actor_from_line(line)


def _actor_from_line(line: str) -> Optional[str]:
    match = re.search(r"(student_\d+)", line)
    return match.group(1) if match else None


# ---------------------------------------------------------------- hệ thống offline
class OfflineProctor:
    """Nạp các model cần dùng và gọi chúng trên từng frame, theo đúng stride của bản realtime."""

    def __init__(self, args, sink: AlertSink) -> None:
        self.args = args
        self.sink = sink
        self.enabled = {m.strip() for m in args.modules.split(",") if m.strip()}
        unknown = self.enabled - set(ALL_MODULES)
        if unknown:
            raise SystemExit(f"[LỖI] Module không tồn tại: {', '.join(sorted(unknown))}")
        self.timings: dict[str, list[float]] = {}
        self.utterances: list[dict[str, Any]] = []
        self.actors: dict[str, dict[str, Any]] = {}
        self.armed = False

    def load_audio(self) -> None:
        """Nạp riêng bộ audio để phiên âm xong là giải phóng trước khi nạp bộ thị giác.

        Đo 2026-10-28: audio một mình ~2.7GB, thị giác ~2.4GB, nạp cùng lúc ~3.1GB đỉnh. Nạp
        tuần tự + `release_audio()` hạ đỉnh xuống ~2.7GB và trả ~2.7GB trước khi chạy frame.
        """
        if "audio" in self.enabled:
            self._timed("audio_init", lambda: setattr(self, "audio_pipeline", AudioPipeline()))

    def load_vision(self) -> None:
        if "object" in self.enabled:
            self._timed("object_init", lambda: setattr(
                self, "yolo", ObjectDetector(model_path=self.args.object_model, device=self.args.device,
                                             confidence_threshold=self.args.confidence)))
        if "face" in self.enabled:
            self._timed("face_init", lambda: setattr(
                self, "face", FaceVerifier(db_path=self.args.faces_dir)))
        if "paper" in self.enabled:
            self._timed("paper_init", self._build_paper)
        # "pose" chỉ còn là bí danh cũ của "action": realtime không còn stack pose thứ hai,
        # nên offline cũng chỉ dựng một stack (nếu cả hai được chọn thì vẫn chỉ chạy một lần).
        if "action" in self.enabled or "pose" in self.enabled:
            self._timed("action_init", self._build_action)

    def _timed(self, name: str, fn):
        started = time.perf_counter()
        out = fn()
        self.timings.setdefault(name, []).append(round(time.perf_counter() - started, 3))
        return out

    def _build_paper(self) -> None:
        self.paper_person = UltralyticsPersonDetector(
            model_path=self.args.person_model, confidence_threshold=0.55,
            device=self.args.device or "auto")
        self.paper_detector = ObjectDetectModule(device=self.args.device or "auto")
        self.paper_pipeline = PaperCountPipeline(
            person_detector=self.paper_person,
            object_detector=self.paper_detector,
            storage_root=Path(self.args.out_dir) / "tracking",
            max_people=2,
        )
        # Pipeline giấy chỉ lo đếm giấy: tắt fallback smartphone y như app realtime
        detector = getattr(self.paper_pipeline, "object_detector", None)
        if detector is not None:
            if hasattr(detector, "_smartphone_fallback_enabled"):
                detector._smartphone_fallback_enabled = False
            if hasattr(detector, "_auxiliary_class_ids"):
                detector._auxiliary_class_ids = []
        if hasattr(self.paper_pipeline, "enable_smartphone_detection"):
            self.paper_pipeline.enable_smartphone_detection = False
        self.paper_session_id = f"offline_session_{int(time.time())}"
        self.paper_pipeline.create_session(self.paper_session_id, restore_existing=False)

    def _build_action(self) -> None:
        self.stack = gaze_scan.build_stack(device=self.args.device or "auto")
        self.sink.gaze_model = self.stack

    def release_audio(self) -> None:
        """Bỏ tham chiếu tới AudioPipeline để GC trả RAM trước khi chạy phần video."""
        import gc
        self.audio_pipeline = None
        gc.collect()

    def process(self, frame: np.ndarray, t: float, frame_id: int) -> None:
        self.sink.frame_id = frame_id
        stride = self.args.stride

        # Cảnh báo audio đã chuyển ngữ từ trước, "nổ" đúng lúc video chạy tới mốc của câu đó
        while self.utterances and self.utterances[0]["timestamp"] <= t:
            self.sink.add(self.utterances.pop(0), t)

        if frame_id % stride["heavy"] == 0:
            if "object" in self.enabled:
                self._detect("object", lambda: object_scan.scan_objects(self.yolo, frame, t), t)
            if "face" in self.enabled:
                self._detect("face", lambda: face_scan.scan_stranger_with_masking(self.face, frame, t)[0], t)

        if "paper" in self.enabled and frame_id % stride["paper"] == 0:
            self._paper(frame, t, frame_id)

        # "pose" là bí danh của "action" (một stack duy nhất); chọn cả hai vẫn chỉ chạy một lần.
        action_stride = stride["action"] if "action" in self.enabled else stride["pose"]
        if ("action" in self.enabled or "pose" in self.enabled) and frame_id % action_stride == 0:
            started = time.perf_counter()
            action_alerts = gaze_scan.scan_frame(self.stack, frame, t)
            self.timings.setdefault("action", []).append(round(time.perf_counter() - started, 4))
            self._record_actor_state(t)
            for alert in action_alerts:
                state = self.actors.get(str(alert.get("actor_id")))
                if state is not None:
                    state["alerts"] += 1
                self.sink.add(alert, t)

    SCORE_KEYS = ("c2_score", "c3_score", "suspicious_activity_score")

    def _record_actor_state(self, t: float) -> None:
        """Giữ bằng chứng của từng actor kể cả khi classifier kết luận c5 (không có alert)."""
        for actor_id, clf in self.stack.last_classifications.items():
            state = self.actors.setdefault(actor_id, {
                "frames": 0, "last_t": 0.0, "predicted_class": None, "alerts": 0, "max_score": {}})
            state["frames"] += 1
            state["last_t"] = round(t, 2)
            state["predicted_class"] = clf.get("predicted_class")
            for key in self.SCORE_KEYS:
                try:
                    value = float(clf.get(key))
                except (TypeError, ValueError):
                    continue  # classifier trả '' khi đặc trưng chưa tính được
                state["max_score"][key] = round(max(state["max_score"].get(key, 0.0), value), 4)

    def actor_report(self) -> dict[str, Any]:
        """Kết luận cuối của classifier theo từng actor (evidence + frame đánh dấu lần đầu)."""
        classifier = getattr(getattr(self, "stack", None), "live_classifier", None)
        final = (classifier.final_decisions() if classifier is not None else None) or {}
        report = {}
        for actor_id, state in self.actors.items():
            decision = final.get(actor_id, {})
            report[actor_id] = {
                **state,
                "final_class": decision.get("predicted_class"),
                "evidence_score": decision.get("evidence_score"),
                "first_flag_frame": decision.get("first_flag_frame_index"),
            }
        return report

    def _detect(self, name: str, fn, t: float) -> None:
        started = time.perf_counter()
        try:
            self.sink.add(fn(), t)
        except Exception as e:
            print(f"[{name}] Lỗi frame {self.sink.frame_id}: {e}")
        self.timings.setdefault(name, []).append(round(time.perf_counter() - started, 4))

    def _paper(self, frame, t, frame_id) -> None:
        if not self.armed and t >= self.args.setup_seconds:
            self.armed = True
            info = paper_scan.arm_baseline(self.paper_pipeline, self.paper_session_id)
            print(f"[PAPER] {info['status_msg']}")
            self.sink.add({"module": "system", "status": "info", "timestamp": t,
                           "message": f"🔒 {info['status_msg']}"}, t)
        try:
            result = self.paper_pipeline.process_frame(
                frame, session_id=self.paper_session_id, frame_id=frame_id, timestamp_ms=int(t * 1000))
        except Exception as e:
            print(f"[paper] Lỗi frame {frame_id}: {e}")
            return
        self.sink.add(paper_scan.build_alert(result, t), t)


# ---------------------------------------------------------------- CLI
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chạy pipeline giám sát offline trên 1 file video.")
    parser.add_argument("video", help="đường dẫn file video (.mp4/.MOV/.mkv)")
    parser.add_argument("--output", default=None, help="thư mục kết quả (mặc định offline_out/<tên>_<giờ>)")
    parser.add_argument("--setup-seconds", type=float, default=5.0,
                        help="tự ARM baseline giấy sau N giây đầu của video (0 = ARM ngay frame đầu)")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="chỉ xử lý N giây đầu (0 = cả file)")
    parser.add_argument("--max-frames", type=int, default=0, help="giới hạn số frame (0 = không giới hạn)")
    parser.add_argument("--modules", default=",".join(ALL_MODULES),
                        help="object,face,paper,pose,action,audio (pose là bí danh của action)")
    parser.add_argument("--stride-heavy", type=int, default=1,
                        help="cứ N frame chạy YOLO vật cấm + verify mặt (realtime: FPS_SKIP=1)")
    parser.add_argument("--stride-paper", type=int, default=5)
    parser.add_argument("--stride-pose", type=int, default=1)
    parser.add_argument("--stride-action", type=int, default=1)
    parser.add_argument("--ttl", type=float, default=1.5, help="thời gian hiển thị overlay, tính theo giây video")
    parser.add_argument("--device", default=None, help="cuda | intel | cpu (mặc định: tự chọn)")
    parser.add_argument("--object-model", default="weights/best (1).pt")
    parser.add_argument("--person-model", default="weights/yolov8n.pt")
    parser.add_argument("--faces-dir", default="data/student_faces/")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-noise-seconds", type=float, default=0.0,
                        help="giây đầu bị ném đi làm noise profile (realtime Ubuntu = 1.0, offline = 0)")
    parser.add_argument("--no-sound", action="store_true", help="không remux audio gốc vào video kết quả")
    parser.add_argument("--quiet", action="store_true", help="không in tiến độ từng frame")
    parser.add_argument("--progress-jsonl", action="store_true",
                        help="in tiến độ dạng JSON (1 dòng/lần, tiền tố @@PROGRESS@@) để tiến trình cha đọc; "
                             "đồng thời ghi offline_out/<run>/preview.jpg mỗi 3s")
    args = parser.parse_args(argv)
    args.stride = {"heavy": max(1, args.stride_heavy), "paper": max(1, args.stride_paper),
                   "pose": max(1, args.stride_pose), "action": max(1, args.stride_action)}
    return args


PROGRESS_PREFIX = "@@PROGRESS@@"
PREVIEW_WIDTH = 640
PREVIEW_INTERVAL_S = 3.0


def _emit(progress, **event) -> None:
    """Gửi tiến độ cho UI (nếu có). Callback lỗi không bao giờ được làm chết luồng chạy."""
    if progress is None:
        return
    try:
        progress(event)
    except Exception as e:  # UI chỉ là người xem, không được làm hỏng phiên chạy
        print(f"[WARN] progress callback lỗi: {e}")


def _jsonl_progress(event: dict[str, Any]) -> None:
    """Chế độ tiến trình con: cha đọc stdout, nên bỏ ảnh (không serialize được) khỏi payload."""
    payload = {k: v for k, v in event.items() if k != "frame"}
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _write_preview(frame: np.ndarray, out_dir: Path) -> None:
    """Ghi JPEG thu nhỏ để tiến trình cha hiển thị mà không cần truyền frame qua pipe."""
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (PREVIEW_WIDTH, max(1, int(height * PREVIEW_WIDTH / width))))
    target = out_dir / "preview.jpg"
    tmp = target.with_suffix(".tmp.jpg")
    if cv2.imwrite(str(tmp), small):
        tmp.replace(target)


def _lower_oom_priority() -> None:
    """Chạy như tiến trình con của UI: nếu kernel buộc phải giết ai, giết job trước Streamlit/IDE.

    Không đặt cho CLI thường — người chạy CLI có thể không muốn bị ưu tiên giết.
    """
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as fh:
            fh.write("500")
    except OSError:
        pass  # không phải Linux hoặc không có quyền: bỏ qua


def main(argv=None, progress=None) -> int:
    args = parse_args(argv)
    if args.progress_jsonl:
        _lower_oom_priority()
    if args.progress_jsonl and progress is None:
        progress = _jsonl_progress
    if args.device:
        os.environ["PROCTOR_DEVICE"] = args.device

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[LỖI] Không thấy file video: {video_path}")
        _emit(progress, stage="error", error=f"Không thấy file video: {video_path}")
        return 2

    out_dir = Path(args.output) if args.output else Path("offline_out") / f"{video_path.stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    _emit(progress, stage="init", out_dir=str(out_dir), video=str(video_path))

    print("=" * 70)
    print(f"[OFFLINE] video   : {video_path}")
    print(f"[OFFLINE] output  : {out_dir}")
    print(f"[OFFLINE] modules : {args.modules}")
    print(f"[OFFLINE] strides : {args.stride}  setup={args.setup_seconds}s  ttl={args.ttl}s")
    print("=" * 70)

    sink = AlertSink(ttl=args.ttl)
    started_at = time.perf_counter()
    _emit(progress, stage="models", detail="Đang nạp bộ âm thanh" if "audio" in args.modules else
          "Đang nạp YOLO / FaceNet / paper / gaze")
    proctor = OfflineProctor(args, sink)

    audio_stats = {"chunks": 0, "speech_segments": 0, "error": None}
    if "audio" in proctor.enabled:
        proctor.load_audio()
        _emit(progress, stage="audio", detail="Đang giải mã audio + PhoWhisper/PhoBERT")
        print("[AUDIO] Đang giải mã track audio + cắt câu + PhoWhisper/PhoBERT ...")
        proctor.utterances, audio_stats = transcribe_file_audio(video_path, proctor.audio_pipeline, args)
        proctor.utterances.sort(key=lambda u: u["timestamp"])
        print(f"[AUDIO] {audio_stats['speech_segments']} câu có tiếng -> {len(proctor.utterances)} câu hiển thị trên UI")
        # VAD + PhoWhisper + PhoBERT chỉ dùng cho lượt phiên âm đầu; giữ lại trong suốt phần video
        # là giữ thêm ~2.7GB RAM vô ích (máy 16GB rất dễ chạm swap).
        proctor.release_audio()
    _emit(progress, stage="models", detail="Đang nạp YOLO / FaceNet / paper / gaze")
    proctor.load_vision()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[LỖI] cv2 không mở được video: {video_path}")
        _emit(progress, stage="error", error=f"Không mở được file video: {video_path}")
        return 2
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames:
        total = min(total, args.max_frames) if total else args.max_frames
    if args.max_seconds:
        capped = int(args.max_seconds * fps)
        total = min(total, capped) if total else capped
    _emit(progress, stage="video", fps=fps, total_frames=total)

    video_only = out_dir / "video_only.mp4"
    writer: Optional[cv2.VideoWriter] = None
    frame_id = 0
    last_preview = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t = frame_id / fps
        if args.max_seconds and t > args.max_seconds:
            break
        if args.max_frames and frame_id >= args.max_frames:
            break
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(video_only), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        proctor.process(frame, t, frame_id)
        annotated = sink.draw(frame, t)
        writer.write(annotated)
        frame_id += 1
        if args.progress_jsonl and time.perf_counter() - last_preview >= PREVIEW_INTERVAL_S:
            _write_preview(annotated, out_dir)
            last_preview = time.perf_counter()
        _emit(progress, stage="frame", frame_id=frame_id, t=t, alerts=len(sink.events),
              logs=list(sink.logs[:6]), frame=annotated)
        if not args.quiet and frame_id % 30 == 0:
            print(f"[FRAME] {frame_id}  t={mmss(t)}  alerts={len(sink.events)}  "
                  f"elapsed={time.perf_counter() - started_at:.1f}s", flush=True)

    capture.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - started_at
    _emit(progress, stage="encode", detail="Đang ghép audio + đóng gói kết quả")
    final_video = remux(video_only, video_path, out_dir / "annotated.mp4", mute=args.no_sound)
    write_outputs(out_dir, sink, proctor, args, video_path, final_video, fps, frame_id, elapsed, audio_stats)
    print_summary(sink, proctor, elapsed, final_video)
    _emit(progress, stage="done", frames=frame_id, video=str(final_video),
          out_dir=str(out_dir), elapsed=round(elapsed, 1), events=list(sink.events))
    return 0


def to_h264(src: Path, target: Path) -> Optional[Path]:
    """Remux sang H.264 để <video> của trình duyệt phát được (annotated.mp4 gốc là mpeg4/mp4v).

    Chỉ dùng cho luồng UI; CLI vẫn giữ nguyên mp4v để không đổi sản phẩm đã kiểm chứng.
    """
    if shutil.which("ffmpeg") is None:
        print("[WARN] Không có ffmpeg -> không tạo được bản H.264 cho trình duyệt.")
        return None
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-c:a", "aac", str(target)]
    try:
        subprocess.run(cmd, check=True)
        return target
    except Exception as e:
        print(f"[WARN] Transcode H.264 thất bại ({e}) -> UI sẽ không phát được video này.")
        return None


def remux(video_only: Path, source: Path, target: Path, mute: bool) -> Path:
    """Ghép video đã vẽ với soundtrack gốc (copy video, không encode lại)."""
    if not video_only.exists():
        return video_only
    if mute or shutil.which("ffmpeg") is None:
        reason = "--no-sound" if mute else "không có ffmpeg trong PATH"
        if not mute:
            print(f"[WARN] {reason} -> video kết quả không kèm audio.")
        video_only.replace(target)
        return target
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(source),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(target)]
    try:
        subprocess.run(cmd, check=True)
        video_only.unlink()
    except Exception as e:
        print(f"[WARN] Remux audio thất bại ({e}) -> dùng video không audio.")
        video_only.replace(target)
    return target


def write_outputs(out_dir, sink, proctor, args, video_path, final_video, fps, frames, elapsed, audio_stats) -> None:
    with open(out_dir / "events.jsonl", "w", encoding="utf-8") as fh:
        for event in sink.events:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    with open(out_dir / "log.txt", "w", encoding="utf-8") as fh:
        fh.write(f"# {video_path.name} — chạy lúc {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"# {frames} frame @ {fps:.2f} fps | {elapsed:.1f}s xử lý | modules={args.modules}\n\n")
        # Chỉ ghi những dòng mà sidebar realtime cũng hiện; frame 'info' của pipeline giấy
        # không có log nhưng vẫn còn nguyên trong events.jsonl.
        for event in sorted(sink.events, key=lambda e: e["t"]):
            if event["log"]:
                fh.write(overlay_draw.plain_text(event["log"]) + "\n")

    by_module: dict[str, int] = {}
    by_actor: dict[str, dict[str, int]] = {}
    for event in sink.events:
        module = event["module"] or "?"
        by_module[module] = by_module.get(module, 0) + 1
        if event.get("actor_id"):
            tally = by_actor.setdefault(event["actor_id"], {})
            tally[module] = tally.get(module, 0) + 1

    report = {
        "video": str(video_path),
        "output_video": str(final_video),
        "fps": fps,
        "frames_processed": frames,
        "wall_time_s": round(elapsed, 2),
        "config": {"setup_seconds": args.setup_seconds, "max_seconds": args.max_seconds,
                   "modules": args.modules, "ttl": args.ttl, "strides": args.stride,
                   "device": args.device, "object_model": args.object_model,
                   "person_model": args.person_model, "faces_dir": args.faces_dir,
                   "audio_noise_seconds": args.audio_noise_seconds},
        "alerts_by_module": by_module,
        "alerts_by_actor": by_actor,
        "actors": proctor.actor_report(),
        "audio": {**audio_stats,
                  "transcriptions": [
                      {"t": e["t"], "transcription": e["alert"].get("transcription"),
                       "risk": e["alert"].get("risk"), "rule_label": e["alert"].get("rule_label"),
                       "ai_label": e["alert"].get("ai_label"),
                       "fusion_reason": e["alert"].get("fusion_reason")}
                      for e in sink.events if "audio" in str(e["module"])]},
        "timings_s": {k: {"count": len(v), "total": round(sum(v), 3), "avg": round(sum(v) / len(v), 4)}
                      for k, v in proctor.timings.items()},
        "log": [e["log"] for e in sink.events if e["log"]],
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)


def print_summary(sink, proctor, elapsed, final_video) -> None:
    counts: dict[str, int] = {}
    for event in sink.events:
        counts[event["module"] or "?"] = counts.get(event["module"] or "?", 0) + 1
    print("=" * 70)
    print(f"[DONE] {sum(counts.values())} cảnh báo trong {elapsed:.1f}s:")
    for module, count in sorted(counts.items()):
        print(f"   - {module:26s} {count}")
    for actor_id, state in sorted(proctor.actor_report().items()):
        print(f"[ACTOR] {actor_id}: class={state['final_class']} frames={state['frames']} "
              f"alerts={state['alerts']} max_score={state['max_score']}")
    print(f"[DONE] video kết quả: {final_video}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
