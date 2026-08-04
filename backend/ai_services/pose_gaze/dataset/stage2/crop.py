"""Crop frame theo bbox với padding, clamp về biên ảnh."""

from __future__ import annotations

import numpy as np

from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox
from .common import CROP_PADDING


def crop_with_padding(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding: float = CROP_PADDING,
) -> tuple[np.ndarray, BoundingBox]:
    """
    Cắt vùng person từ frame với padding% theo cả 4 phía.

    Parameters
    ----------
    frame   : BGR frame (H, W, 3)
    bbox    : BoundingBox từ IoUPersonTracker (pixel tuyệt đối)
    padding : hệ số padding (0.15 = 15% chiều dài bbox mỗi phía)

    Returns
    -------
    (crop_bgr, actual_bbox)
        crop_bgr   : mảng numpy đã cắt — KHÔNG resize (giữ aspect ratio thật)
        actual_bbox: BoundingBox pixel tuyệt đối sau khi clamp về biên ảnh
    """
    h, w = frame.shape[:2]

    pad_x = (bbox.x2 - bbox.x1) * padding
    pad_y = (bbox.y2 - bbox.y1) * padding

    x1 = int(max(0.0, bbox.x1 - pad_x))
    y1 = int(max(0.0, bbox.y1 - pad_y))
    x2 = int(min(float(w), bbox.x2 + pad_x))
    y2 = int(min(float(h), bbox.y2 + pad_y))

    # Đảm bảo crop có diện tích dương
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)

    crop = frame[y1:y2, x1:x2]
    actual_bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
    return crop, actual_bbox


def landmark_crop_to_frame(
    lm_x_crop: float,
    lm_y_crop: float,
    crop_bbox: BoundingBox,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float]:
    """
    Chuyển tọa độ landmark normalized theo crop sang normalized theo frame gốc.

    MediaPipe trả tọa độ x,y ∈ [0,1] tính trong ảnh input (= crop).
    Hàm này map ngược về tọa độ [0,1] tính trong frame gốc.

    Parameters
    ----------
    lm_x_crop, lm_y_crop : tọa độ normalized trong crop [0,1]
    crop_bbox             : bbox crop (pixel tuyệt đối trong frame)
    frame_w, frame_h      : kích thước frame gốc

    Returns
    -------
    (x_frame_norm, y_frame_norm) ∈ [0,1] tương đối frame gốc
    """
    crop_w = crop_bbox.x2 - crop_bbox.x1
    crop_h = crop_bbox.y2 - crop_bbox.y1

    # Pixel tuyệt đối trong frame
    px = crop_bbox.x1 + lm_x_crop * crop_w
    py = crop_bbox.y1 + lm_y_crop * crop_h

    return px / frame_w, py / frame_h
