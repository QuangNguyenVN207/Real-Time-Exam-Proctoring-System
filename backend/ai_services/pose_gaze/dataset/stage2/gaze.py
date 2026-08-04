"""
Gaze direction từ iris landmarks — TÁCH BIỆT khỏi head pose.

Chỉ sử dụng khi is_gaze_valid() = True (face quality đủ VÀ iris có mặt).
Khi không hợp lệ, trả về ('unknown', False) — KHÔNG suy diễn giá trị.

Method: normalize iris center theo eye corners (không cần deep learning).
"""

from __future__ import annotations

import numpy as np

from .quality import LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES, is_gaze_valid, GAZE_MIN_QUALITY_SCORE

# Eye corner indices trong MediaPipe Face Mesh (legacy 468-point / Tasks 478-point)
# Left eye  : inner=133, outer=33
# Right eye : inner=362, outer=263
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263

# Ngưỡng phân loại hướng nhìn (tỷ lệ, đã normalize theo eye width)
_GAZE_HORIZONTAL_THRESHOLD: float = 0.15  # |offset_x| > 15% eye width
_GAZE_VERTICAL_THRESHOLD: float = 0.15
# Chất lượng face tối thiểu để dùng eye-corner fallback (không cần iris)
_GAZE_CORNER_QUALITY_MIN: float = 0.5


def extract_gaze(
    face_lm: np.ndarray,                   # (N, 3) crop-normalized
    face_quality: float,
    face_indices: np.ndarray | None = None, # (N,) original MediaPipe indices
) -> tuple[str, bool]:
    """
    Tính hướng nhìn từ iris và eye corner landmarks.

    Parameters
    ----------
    face_lm       : landmark array crop-normalized
    face_quality  : score từ quality.face_quality_score()
    face_indices  : array shape (N,) — index MediaPipe tương ứng

    Returns
    -------
    (direction, is_valid)
        direction : 'center'|'left'|'right'|'up'|'down'|'unknown'
        is_valid  : True nếu gaze thực sự được tính; False nếu fallback 'unknown'
    """
    if face_lm is None or face_lm.size == 0:
        return "unknown", False

    # Kiểm tra iris presence
    if face_indices is not None:
        has_iris = all(idx in face_indices for idx in LEFT_IRIS_INDICES)
    else:
        has_iris = face_lm.shape[0] >= 478

    valid = is_gaze_valid(face_quality, has_iris)
    if not valid:
        return "unknown", False

    def _get_pt(idx: int) -> np.ndarray | None:
        if face_indices is not None:
            m = np.where(face_indices == idx)[0]
            return face_lm[m[0], :2] if len(m) > 0 else None
        else:
            return face_lm[idx, :2] if idx < face_lm.shape[0] else None

    left_inner  = _get_pt(_LEFT_EYE_INNER)
    left_outer  = _get_pt(_LEFT_EYE_OUTER)
    right_inner = _get_pt(_RIGHT_EYE_INNER)
    right_outer = _get_pt(_RIGHT_EYE_OUTER)

    if left_inner is None or left_outer is None or right_inner is None or right_outer is None:
        return "unknown", False

    # Iris points
    l_iris = [_get_pt(i) for i in LEFT_IRIS_INDICES]
    r_iris = [_get_pt(i) for i in RIGHT_IRIS_INDICES]
    if any(p is None for p in l_iris + r_iris):
        return "unknown", False

    left_iris_center  = np.mean(l_iris, axis=0)
    right_iris_center = np.mean(r_iris, axis=0)

    # Eye width (normalized)
    left_eye_w  = float(abs(left_inner[0]  - left_outer[0]))
    right_eye_w = float(abs(right_inner[0] - right_outer[0]))
    eye_w = max(left_eye_w + right_eye_w, 1e-6)

    # Midpoint eye (dùng làm reference)
    left_eye_mid_x  = (left_inner[0]  + left_outer[0])  / 2.0
    right_eye_mid_x = (right_inner[0] + right_outer[0]) / 2.0
    left_eye_mid_y  = (left_inner[1]  + left_outer[1])  / 2.0
    right_eye_mid_y = (right_inner[1] + right_outer[1]) / 2.0

    # Offset iris so với mid eye (normalized theo eye width)
    offset_x = (
        (left_iris_center[0]  - left_eye_mid_x)
        + (right_iris_center[0] - right_eye_mid_x)
    ) / (2.0 * eye_w)
    offset_y = (
        (left_iris_center[1]  - left_eye_mid_y)
        + (right_iris_center[1] - right_eye_mid_y)
    ) / (2.0 * max(left_eye_w + right_eye_w, 1e-6))

    # Phân loại (ưu tiên horizontal)
    if abs(offset_x) >= _GAZE_HORIZONTAL_THRESHOLD:
        direction = "left" if offset_x < 0 else "right"
    elif abs(offset_y) >= _GAZE_VERTICAL_THRESHOLD:
        direction = "up" if offset_y < 0 else "down"
    else:
        direction = "center"

    return direction, True
