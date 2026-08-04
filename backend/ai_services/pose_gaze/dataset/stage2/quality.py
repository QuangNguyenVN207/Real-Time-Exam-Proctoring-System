"""Quality gate cho Face landmarks — kiểm soát khi nào gaze hợp lệ."""

from __future__ import annotations

import numpy as np

from .common import (
    FACE_QUALITY_MIN_AREA_PX,
    FACE_QUALITY_MIN_VISIBILITY,
    GAZE_MIN_QUALITY_SCORE,
)

# 6 key-points mặt dùng để đánh giá visibility:
# mũi tip (1), mắt trái ngoài (33), mắt phải ngoài (263),
# miệng trái (61), miệng phải (291), cằm (199)
_FACE_KEY_VIS_INDICES: tuple[int, ...] = (1, 33, 263, 61, 291, 152)

# Iris indices trong MediaPipe Face Mesh (478-point model — Tasks API với refine)
# Legacy 468-point Holistic KHÔNG có iris; kiểm tra face_lm_indices trước khi dùng.
# Left iris : 468 (center) + 469, 470, 471, 472 (ring)
# Right iris: 473 (center) + 474, 475, 476, 477 (ring)
LEFT_IRIS_INDICES: tuple[int, ...] = (468, 469, 470, 471, 472)
RIGHT_IRIS_INDICES: tuple[int, ...] = (473, 474, 475, 476, 477)


def face_quality_score(
    face_lm: np.ndarray | None,            # (N, 3) crop-normalized
    face_vis: np.ndarray | None,           # (N,) visibility [0,1]
    crop_h: int,
    crop_w: int,
    face_indices: np.ndarray | None = None, # (N,) MediaPipe landmark indices
) -> float:
    """
    Tính quality score [0,1] của face detection trong crop.

    Score = trung bình có trọng số của 3 thành phần:
      1. landmark_present  (0 hoặc 1) — weight 0.4
      2. area_score        [0,1]      — weight 0.3
      3. visibility_score  [0,1]      — weight 0.3

    area_score  = 1 nếu bbox mặt >= FACE_QUALITY_MIN_AREA_PX px², giảm tuyến tính
    visibility  = mean visibility của 6 key-points
    """
    if face_lm is None or face_lm.size == 0:
        return 0.0

    # 1. Landmark present
    landmark_present = 1.0

    # 2. Diện tích mặt (ước tính từ bounding box của landmarks)
    xs = face_lm[:, 0] * crop_w
    ys = face_lm[:, 1] * crop_h
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    face_area_px = max(0.0, (x_max - x_min) * (y_max - y_min))
    area_score = min(1.0, face_area_px / max(1.0, float(FACE_QUALITY_MIN_AREA_PX)))

    # 3. Mean visibility của key-points
    if face_vis is not None and face_vis.size > 0:
        if face_indices is not None:
            v_list = []
            for idx in _FACE_KEY_VIS_INDICES:
                m = np.where(face_indices == idx)[0]
                if len(m) > 0:
                    v_list.append(float(face_vis[m[0]]))
            visibility_score = float(np.clip(np.mean(v_list), 0.0, 1.0)) if v_list else area_score
        elif len(face_vis) > max(_FACE_KEY_VIS_INDICES):
            vis_values = face_vis[list(_FACE_KEY_VIS_INDICES)]
            visibility_score = float(np.clip(vis_values.mean(), 0.0, 1.0))
        else:
            visibility_score = area_score
    else:
        # Tasks API không có visibility trên face landmarks → dùng area làm proxy
        visibility_score = area_score

    score = (
        0.4 * landmark_present
        + 0.3 * area_score
        + 0.3 * visibility_score
    )
    return float(np.clip(score, 0.0, 1.0))


def iris_present(face_lm: np.ndarray | None) -> bool:
    """
    Kiểm tra xem model có trả về iris landmarks không.

    MediaPipe legacy Holistic (468 points) KHÔNG có iris.
    MediaPipe Tasks HolisticLandmarker (478 points) CÓ iris (indices 468-477).
    """
    if face_lm is None:
        return False
    # Iris chỉ có nếu model trả về ít nhất 478 points
    return face_lm.shape[0] >= 478


def is_gaze_valid(quality_score: float, has_iris: bool) -> bool:
    """
    Gaze hợp lệ khi VÀ CHỈ KHI:
    - face_quality_score >= GAZE_MIN_QUALITY_SCORE
    - iris landmarks có mặt (model Tasks API)

    Khi trả False, gaze_direction phải được set = 'unknown'.
    """
    return quality_score >= GAZE_MIN_QUALITY_SCORE and has_iris
