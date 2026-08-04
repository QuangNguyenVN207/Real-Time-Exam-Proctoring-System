"""Data contracts cho Stage 2 — không phụ thuộc vào MediaPipe hay OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Metadata về một crop (1 track trong 1 frame)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class CropMeta:
    """Metadata định vị crop trong không gian frame gốc."""

    clip_id: str
    window_id: str
    frame_idx: int          # index trong window (0..N-1)
    track_id: int

    # bbox person từ YOLO → IoUPersonTracker (pixel tuyệt đối trong frame)
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    # bbox sau khi clamp + padding (pixel tuyệt đối trong frame)
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float

    frame_h: int
    frame_w: int
    timestamp_ms: float        # millisecond từ đầu clip (theo stage1 parquet)
    tracking_confidence: float # detector confidence từ IoUPersonTracker


# ─────────────────────────────────────────────────────────────────────────────
# Raw output của MediaPipe Holistic trên 1 crop × 1 track × 1 frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HolisticRaw:
    """
    Landmark thô từ MediaPipe Holistic cho một crop.

    Tọa độ lưu theo 2 hệ:
    - crop_lm  : normalized về kích thước crop (x,y ∈ [0,1] tương đối crop)
    - frame_lm : normalized về kích thước frame gốc (x,y ∈ [0,1] tương đối frame)

    z giữ nguyên theo MediaPipe (relative depth, cùng thang với x).

    KHÔNG lưu ảnh RGB; tất cả chỉ là số.

    Khi MediaPipe không phát hiện được một nhóm landmark, mảng tương ứng là None
    và flag missing_* = True.
    """

    meta: CropMeta

    # ── Pose (33 points, chỉ thân trên — xem POSE_KEY_INDICES) ──────────────
    # shape (33, 3): [x_crop, y_crop, z]
    pose_crop_lm: np.ndarray | None = None
    # shape (33, 3): [x_frame, y_frame, z]
    pose_frame_lm: np.ndarray | None = None
    # shape (33,): visibility score [0,1]
    pose_vis: np.ndarray | None = None
    # shape (33,): presence score [0,1]
    pose_presence: np.ndarray | None = None

    # ── Face (468 points, chỉ giữ selected_face_indices) ────────────────────
    # shape (468, 3): [x_crop, y_crop, z]
    face_crop_lm: np.ndarray | None = None
    # shape (468, 3): [x_frame, y_frame, z]
    face_frame_lm: np.ndarray | None = None
    # shape (468,): visibility
    face_vis: np.ndarray | None = None

    # shape (N,): original MediaPipe landmark indices
    face_lm_indices: np.ndarray | None = None

    # ── Left hand (21 points, chỉ HAND_KEY_INDICES) ──────────────────────────
    # shape (21, 3): [x_crop, y_crop, z]
    left_hand_crop_lm: np.ndarray | None = None
    # shape (21, 3): [x_frame, y_frame, z]
    left_hand_frame_lm: np.ndarray | None = None

    # ── Right hand (21 points, chỉ HAND_KEY_INDICES) ─────────────────────────
    right_hand_crop_lm: np.ndarray | None = None
    right_hand_frame_lm: np.ndarray | None = None

    # ── Quality & validity flags ─────────────────────────────────────────────
    face_quality_score: float = 0.0  # [0,1]: tổng hợp từ area + visibility
    gaze_valid: bool = False          # chỉ True khi face đủ rõ VÀ iris có đủ

    missing_pose: bool = True
    missing_face: bool = True
    missing_left_hand: bool = True
    missing_right_hand: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Một hàng trong features/frames.parquet (đã giải mã thành scalar)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameFeatureRow:
    """
    Bảng feature được tính từ HolisticRaw — 1 hàng / frame / track.

    Thiết kế để dễ thêm/bỏ cột mà không chạy lại MediaPipe:
    nếu cần feature mới, chạy lại feature engineering từ .npz thay vì video.
    """

    # ── Keys định danh ───────────────────────────────────────────────────────
    clip_id: str
    window_id: str
    frame_idx: int
    track_id: int
    timestamp_ms: float
    tracking_confidence: float

    # ── Bounding box (pixel tuyệt đối trong frame) ───────────────────────────
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    crop_x1: float
    crop_y1: float
    crop_x2: float
    crop_y2: float
    frame_h: int
    frame_w: int

    # ── Head pose (từ solvePnP trên Face Mesh) ───────────────────────────────
    # None được serialize thành NaN trong parquet
    head_yaw: float | None = None    # độ; âm=trái, dương=phải
    head_pitch: float | None = None  # độ; âm=lên, dương=xuống
    head_roll: float | None = None   # độ; âm=nghiêng trái, dương=phải
    # Phân loại: 'forward'|'left'|'right'|'down'|'unknown'
    head_direction: str = "unknown"

    # ── Gaze (TÁCH BIỆT khỏi head_direction) ────────────────────────────────
    # 'center'|'left'|'right'|'up'|'down'|'unknown'
    gaze_direction: str = "unknown"
    gaze_valid: bool = False
    face_quality_score: float = 0.0

    # ── Body pose ────────────────────────────────────────────────────────────
    shoulder_slope_deg: float | None = None  # góc nghiêng 2 vai (landmark 11,12)
    torso_lean_deg: float | None = None      # góc nghiêng thân (vai → hông)

    # ── Missing flags ────────────────────────────────────────────────────────
    missing_pose: bool = True
    missing_face: bool = True
    missing_left_hand: bool = True
    missing_right_hand: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
