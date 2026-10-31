"""Chuyển `alert dict` thành dòng log hiển thị + quản lý danh sách overlay.

Logic sao y nguyên từ `video_frame_callback()` và `draw_warning_overlays()` trong
app_webrtc.py để luồng offline in ra đúng chữ/màu/cách lọc như luồng realtime.
"""
from __future__ import annotations

import time
from typing import Any, MutableSequence, Optional

__all__ = ["to_log_line", "prune_overlays", "expire_overlays"]

_AUDIO_MODULES = ("audio_phobert_pipeline", "audio_pipeline")


def to_log_line(alert: dict[str, Any], now: Optional[float] = None) -> str:
    """Trả về dòng markdown cho cột 'Lịch Sử Cảnh Báo', hoặc '' nếu alert không để hiển thị."""
    module = alert.get("module", "")
    status = alert.get("status", "")
    details = alert.get("details", {}) or {}
    timestamp = alert.get("timestamp") or now or time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

    if module == "system":
        return f"**[{time_str}]** {alert.get('message')}"

    if module == "face_verify" and status == "alert":
        return f"**[{time_str}]** 🕵️‍♂️ PHÁT HIỆN NGƯỜI LẠ!"

    if module == "object_detect" and status == "alert":
        detected_labels = [det.get("label", "vật thể") for det in alert.get("detections", [])]
        label_str = ", ".join(detected_labels).upper()
        return f"**[{time_str}]** <span style='color:red'>📱 PHÁT HIỆN VẬT CẤM: {label_str}</span>"

    if module == "paper_count" and status == "alert":
        alerts = alert.get("alerts", [])
        papers = alert.get("papers", [])

        has_true_paper_violation = any(
            p.get("status") == "suspicious_new_paper" for p in papers
        )
        detected_device_label = ""
        for a in alerts:
            label = str(a.get("label", "")).lower()
            source = str(a.get("source", "")).lower()
            if "smartphone" in label or "phone" in label or "device" in label or "object_detect" in source:
                detected_device_label = label.upper() if label else "SMARTPHONE"
            if "paper" in label or "cheat" in label or "sheet" in label or "monitor" in source:
                has_true_paper_violation = True

        if has_true_paper_violation:
            return f"**[{time_str}]** <span style='color:red'>📄 PHÁT HIỆN TÀI LIỆU LẠ (CHEATSHEET)!</span>"
        return ""

    if module == "pose_gaze" and status == "warning":
        action_text = details.get("action", "Bất thường")
        if "EXCHANGE" in action_text or "LOOKING" in action_text or "HOẠT ĐỘNG BẤT THƯỜNG" in action_text:
            return f"**[{time_str}]** <span style='color:red'>🚨 GIAN LẬN HÀNH VI: {action_text}</span>"
        return f"**[{time_str}]** ⚠️ TƯ THẾ: {action_text}"

    if (
        "audio" in module
        or module in _AUDIO_MODULES
        or "transcription" in alert
        or "transcription" in details
    ):
        transcription = str(alert.get("transcription", details.get("transcription", ""))).strip()
        if transcription:
            risk_level = str(alert.get("risk", "")).lower()
            audio_status = str(alert.get("status", "")).lower()
            if risk_level in ["cheating", "high", "medium"] or audio_status == "alert":
                return f"**[{time_str}]** <span style='color:red'>🚨 GIAN LẬN ÂM THANH: '{transcription}'</span>"
            return f"**[{time_str}]** 🎙️ Hội thoại: '{transcription}'"

    return ""


def prune_overlays(overlays: MutableSequence[dict[str, Any]], alert: dict[str, Any]) -> None:
    """Thay overlay cũ cùng module có timestamp cổ hơn bằng alert mới (in-place)."""
    module = alert.get("module", "")
    timestamp = alert.get("timestamp", 0)
    kept = [a for a in overlays if not (a.get("module") == module and a.get("timestamp", 0) < timestamp)]
    overlays[:] = kept
    overlays.append(alert)


def expire_overlays(overlays: MutableSequence[dict[str, Any]], now: float, ttl: float) -> None:
    """Bỏ overlay hết hạn hiển thị (in-place)."""
    overlays[:] = [
        item for item in overlays
        if now - item.get("display_timestamp", now) <= ttl
    ]
