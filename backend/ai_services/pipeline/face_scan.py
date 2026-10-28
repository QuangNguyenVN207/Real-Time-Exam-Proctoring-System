"""Quét 'người lạ' bằng FaceVerifier + kỹ thuật tô đen các khuôn mặt còn lại.

Sao y `get_largest_stranger_with_masking()` trong app_webrtc.py:156-195.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

__all__ = ["scan_stranger_with_masking", "crop_for_registration"]

_FACE_MASK_MARGIN = 10
_REGISTER_MARGIN = 40


def scan_stranger_with_masking(
    face_model: Any,
    img: np.ndarray,
    timestamp: float,
) -> Tuple[Optional[dict[str, Any]], Optional[Any]]:
    """Trả (alert|None, bbox_khuôn_mặt_lớn_nhất).

    `verify_face()` chỉ báo khuôn mặt khớp kém **đầu tiên** nó gặp, nên khi trong khung hình
    có nhiều người phải che các mặt khác đi thì kết quả mới đúng cho từng người.
    """
    faces = face_model._detect_faces(img)
    if not faces:
        return None, None

    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    h, w = img.shape[:2]

    for i, target_face in enumerate(faces):
        masked_frame = img.copy()
        for j, other_face in enumerate(faces):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = map(int, other_face.bbox)
            cv2.rectangle(
                masked_frame,
                (max(0, ox1 - _FACE_MASK_MARGIN), max(0, oy1 - _FACE_MASK_MARGIN)),
                (min(w, ox2 + _FACE_MASK_MARGIN), min(h, oy2 + _FACE_MASK_MARGIN)),
                (0, 0, 0),
                -1,
            )

        verify_result = face_model.verify_face(masked_frame, timestamp)
        if verify_result and verify_result.get("status") == "alert":
            verify_result["details"]["unauthorized_bbox"] = [int(x) for x in target_face.bbox]
            return verify_result, target_face.bbox

    return None, faces[0].bbox


def crop_for_registration(img: np.ndarray, bbox: Any, margin: int = _REGISTER_MARGIN) -> dict[str, Any]:
    """Nới rộng bbox khuôn mặt và cắt ảnh chuẩn để lưu vào DB + ảnh toàn cảnh cho người xem."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1_ext = max(0, x1 - margin)
    y1_ext = max(0, y1 - margin)
    x2_ext = min(w, x2 + margin)
    y2_ext = min(h, y2 + margin)

    annotated = img.copy()
    cv2.rectangle(annotated, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 255, 0), 2)
    cv2.putText(annotated, "REGISTERED", (x1_ext, y1_ext - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return {
        "face_crop": img[y1_ext:y2_ext, x1_ext:x2_ext],
        "full_view": annotated,
        "bbox": (x1_ext, y1_ext, x2_ext, y2_ext),
    }
