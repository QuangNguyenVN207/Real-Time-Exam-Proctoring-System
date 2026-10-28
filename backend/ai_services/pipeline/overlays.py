"""Vẽ đè cảnh báo lên khung hình — bản dùng chung cho realtime và offline.

Sao y `draw_warning_overlays()` trong app_webrtc.py, thêm tham số `log_lines` để luồng
offline burn cột 'Lịch Sử Cảnh Báo' ngay vào video (realtime hiển thị cái này ở sidebar).
"""
from __future__ import annotations

import re
from typing import Any, MutableSequence, Optional, Sequence

import cv2

__all__ = ["draw", "plain_text"]

_TAG_RE = re.compile(r"<[^>]+>")
# cv2.putText không vẽ được emoji (ra toàn dấu "?"); panel log chỉ giữ lại phần chữ.
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\u20E3]")


def plain_text(markdown_line: str) -> str:
    """Bỏ markdown/HTML khỏi dòng log để draw lên video bằng OpenCV."""
    text = _TAG_RE.sub("", markdown_line)
    return text.replace("**", "").strip()


def draw(
    frame: Any,
    overlays: Sequence[dict[str, Any]],
    *,
    lock: Optional[Any] = None,
    gaze_model: Optional[Any] = None,
    log_lines: Optional[Sequence[str]] = None,
) -> Any:

    if gaze_model is not None and hasattr(gaze_model, "draw_skeleton"):
        gaze_model.draw_skeleton(frame)

    def render(target: MutableSequence[dict[str, Any]]) -> None:
        for alert in target:
            module = alert.get("module", "")
            details = alert.get("details", {}) or {}

            if module == "object_detect":
                for det in alert.get("detections", []):
                    bbox = det.get("bbox")
                    label = det.get("label", "VAT CAM")
                    if isinstance(label, str):
                        label = label.upper()
                    if bbox is not None and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1 + 5, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "face_verify":
                bbox = details.get("unauthorized_bbox")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(frame, "NGUOI LA", (x1 + 5, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elif module == "pose_gaze":
                action = str(details.get("action", "VI PHAM TU THE")).upper()
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 200), -1)
                cv2.putText(frame, f"[CANH BAO TU THE]: {action}", (20, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif module == "paper_count" and alert.get("papers"):
                for paper in alert["papers"]:
                    x1, y1, x2, y2 = paper["bbox_xyxy"]
                    is_suspicious = paper["status"] == "suspicious_new_paper"
                    color = (0, 0, 255) if is_suspicious else (0, 255, 255)
                    label = "CHEAT SHEET (NEW)" if is_suspicious else f"Paper #{paper.get('observation_index', '?')}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if overlays:
        if lock is not None:
            with lock:
                render(overlays)
        else:
            render(overlays)

    if log_lines:
        _draw_log_panel(frame, log_lines)
    return frame


def _draw_log_panel(frame: Any, log_lines: Sequence[str], max_lines: int = 8) -> None:
    """Góc phải trên: vài dòng cảnh báo gần nhất, mới nhất ở trên — giống thứ tự sidebar."""
    h, w = frame.shape[:2]
    lines = [_EMOJI_RE.sub("", plain_text(line)).strip() for line in list(log_lines)[:max_lines]]
    if not lines:
        return

    pad, line_h = 8, 22
    box_w = min(int(w * 0.42), w - 2 * pad)
    box_h = pad * 2 + line_h * len(lines)
    x0, y0 = w - box_w - pad, pad

    roi = frame[y0:y0 + box_h, x0:x0 + box_w]
    if roi.size:
        blackout = roi * 0.35
        cv2.addWeighted(blackout.astype("uint8"), 0.55, roi, 0.45, 0, dst=roi)

    cv2.putText(frame, "CANH BAO", (x0 + pad, y0 + pad + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for index, text in enumerate(lines):
        y = y0 + pad + 14 + line_h * (index + 1)
        if y > h:
            break
        cv2.putText(frame, text[:60], (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
