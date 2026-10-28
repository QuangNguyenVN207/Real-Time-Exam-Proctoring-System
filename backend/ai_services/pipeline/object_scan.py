"""Chạy YOLO vật cấm và lọc bỏ nhãn giấy (giấy đã có pipeline riêng đếm theo người).

Sao y app_webrtc.py:255-302.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["PAPER_LABELS", "scan_objects"]

PAPER_LABELS = {"paper", "document", "cheatsheet", "sheet", "tai_lieu", "giay", "book"}


def scan_objects(yolo_model: Any, frame: Any, timestamp: float) -> Optional[dict[str, Any]]:
    """Trả alert dict `module="object_detect"` khi còn vật cấm sau khi lọc nhãn giấy, ngược lại None."""
    result = yolo_model.process_frame(frame, timestamp)
    if result is None:
        return None

    detections = result.get("detections", [])
    kept = [det for det in detections if str(det.get("label", "")).lower() not in PAPER_LABELS]
    if not kept:
        return None

    result["detections"] = kept
    result["status"] = "alert"
    result["module"] = "object_detect"
    return result
