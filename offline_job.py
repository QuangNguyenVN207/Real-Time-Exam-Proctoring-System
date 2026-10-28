"""Chạy `run_offline.py` ở luồng nền để giao diện Streamlit theo dõi được tiến trình.

Module này **không** import streamlit: nó chỉ giữ trạng thái + một daemon thread, còn UI đọc
`taken qua snapshot()` và vẽ lại bằng `st_autorefresh`. Ảnh preview và video H.264 đều nằm trong
thư mục kết quả của lần chạy.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2

import run_offline

__all__ = ["OfflineJob"]

PREVIEW_WIDTH = 640
# Ngang nhịp tự làm mới của UI (`OFFLINE_REFRESH_MS`): ghi ảnh dày hơn cũng không được hiển thị thêm lần nào.
PREVIEW_INTERVAL_S = 3.0


class OfflineJob:
    """Một lần chạy offline: argv cho `run_offline.main()` + trạng thái đọc được từ luồng UI."""

    def __init__(self, argv: list[str], out_dir: Path, source_video: Path) -> None:
        self.argv = list(argv)
        self.out_dir = Path(out_dir)
        self.source_video = Path(source_video)
        self.preview_path = self.out_dir / "preview.jpg"

        self.status = "queued"          # queued | running | done | error
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
        self._last_preview = 0.0
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "OfflineJob":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.status = "running"
        self._thread = threading.Thread(target=self._run, name="offline-job", daemon=True)
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        """Chờ luồng chạy xong (dùng trong test); UI bình thường không cần gọi."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------ runner
    def _run(self) -> None:
        try:
            code = run_offline.main(self.argv, progress=self._on_progress)
            if code != 0:
                with self._lock:
                    self.status = "error"
                    self.error = self.error or f"run_offline trả mã lỗi {code}"
                return
            self._finalize()
        except BaseException as e:  # thread nền phải luôn kết thúc bằng trạng thái rõ ràng
            with self._lock:
                self.status = "error"
                self.error = f"{type(e).__name__}: {e}"

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

    # ------------------------------------------------------------------ progress
    def _on_progress(self, event: dict[str, Any]) -> None:
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
                frame = event.get("frame")
                if frame is not None:
                    self._write_preview(frame)
            elif stage == "error":
                self.error = str(event.get("error") or "lỗi không rõ")
            elif stage == "done":
                self.elapsed = float(event.get("elapsed") or 0.0)
                self.total_frames = int(event.get("frames") or self.total_frames)
                self.frame_id = self.total_frames

    def _write_preview(self, frame: Any) -> None:
        """Ghi JPEG thu nhỏ của khung hình đã burn cảnh báo — UI đọc file này mỗi lần rerun."""
        now = time.time()
        if now - self._last_preview < PREVIEW_INTERVAL_S:
            return
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (PREVIEW_WIDTH, max(1, int(height * PREVIEW_WIDTH / width))))
        tmp = self.preview_path.with_suffix(".tmp.jpg")
        if cv2.imwrite(str(tmp), small):
            tmp.replace(self.preview_path)
        self._last_preview = now

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
            }
