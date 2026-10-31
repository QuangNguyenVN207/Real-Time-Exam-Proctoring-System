"""Quét 'người lạ' trên frame: detect MỘT lần, embed tất cả, so khớp tất cả.

Bản cũ (detect-once có masking) chạy `FaceVerifier._detect_faces` N+1 lần cho N khuôn mặt:
1 lần đầu để lấy danh sách, rồi mỗi khuôn mặt lại tô đen các mặt khác và detect lại. Với 2 người
trong khung hình đó là 3 lần RetinaFace + 4 lần ArcFace mỗi frame, dù chỉ cần 1 + N.

Ở đây mỗi khuôn mặt đã có sẵn `normed_embedding` từ lần detect duy nhất, nên so khớp trực tiếp
với DB là đủ. Thứ tự xét vẫn là diện tích giảm dần (mặt lớn nhất trước) và vẫn chỉ trả về
người lạ ĐẦU TIÊN gặp — giữ nguyên hợp đồng alert của bản cũ.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

__all__ = ["scan_stranger_with_masking", "crop_for_registration"]

_REGISTER_MARGIN = 40


def scan_stranger_with_masking(
    face_model: Any,
    img: np.ndarray,
    timestamp: float,
) -> Tuple[Optional[dict[str, Any]], Optional[Any]]:
    """Trả (alert|None, bbox_khuôn_mặt_lớn_nhất).

    Tên hàm giữ nguyên để app realtime và offline không phải đổi chỗ gọi; bên trong không còn
    tô đen + detect lại (xem docstring module).
    """
    faces = face_model._detect_faces(img)
    if not faces:
        return None, None

    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    for target_face in faces:
        _, best_score = face_model._best_match(target_face.normed_embedding)
        if best_score < face_model.similarity_threshold:
            return {
                "module": "face_verify",
                "status": "alert",
                "timestamp": timestamp,
                "message": "unauthorized_person",
                "details": {
                    "similarity_score": round(best_score, 4),
                    "unauthorized_bbox": [int(x) for x in target_face.bbox],
                },
            }, target_face.bbox

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
