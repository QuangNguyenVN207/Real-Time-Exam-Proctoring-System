"""Hằng số và tiện ích dùng chung trong Stage 2."""

from __future__ import annotations

from pathlib import Path


# ─── Crop ───────────────────────────────────────────────────────────────────
# Padding 15% quanh bbox person trước khi feed vào MediaPipe Holistic
CROP_PADDING: float = 0.15

# ─── Face / gaze quality ────────────────────────────────────────────────────
# Diện tích tối thiểu (pixel²) của vùng mặt trong crop để gaze hợp lệ
FACE_QUALITY_MIN_AREA_PX: int = 3600          # 60×60 px

# Mean visibility trung bình tối thiểu của 6 key-points mặt
FACE_QUALITY_MIN_VISIBILITY: float = 0.5

# Ngưỡng quality score [0,1] để đánh dấu gaze_valid=True
GAZE_MIN_QUALITY_SCORE: float = 0.65

# ─── YOLO person detection ──────────────────────────────────────────────────
YOLO_CONF_THRESHOLD: float = 0.40
YOLO_PERSON_CLASS_NAME: str = "person"

# ─── Tracker ────────────────────────────────────────────────────────────────
# Số người tối đa theo dõi cùng lúc trong một window
MAX_TRACKS: int = 2

# IoU tối thiểu để associate detection với track cũ
TRACKER_MIN_IOU: float = 0.10

# Số frame bị miss trước khi xoá track (~ 3 giây ở 10 FPS)
TRACKER_MAX_MISSED: int = 30

# ─── MediaPipe ──────────────────────────────────────────────────────────────
# model_complexity 1 = cân bằng tốc độ / độ chính xác (0=nhanh, 2=chính xác)
HOLISTIC_MODEL_COMPLEXITY: int = 1

# Offline dataset: mỗi frame độc lập, không có temporal smoothing từ webcam
HOLISTIC_STATIC_IMAGE_MODE: bool = True

# ─── Hand landmarks ─────────────────────────────────────────────────────────
# Chỉ giữ các điểm quan trọng để nhận biết ký hiệu tay:
# Wrist (0), ngón cái tip (4), ngón trỏ tip (8), ngón giữa tip (12),
# ngón áp út tip (16), ngón út tip (20) + MCP joints (1,5,9,13,17)
HAND_KEY_INDICES: frozenset[int] = frozenset(
    {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20}
)

# ─── Pose landmarks ─────────────────────────────────────────────────────────
# Chỉ giữ thân trên: vai (11,12), khuỷu (13,14), cổ tay (15,16),
# hông (23,24), mũi (0), mắt (1,2,3,4,5,6), tai (7,8)
POSE_KEY_INDICES: frozenset[int] = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24}
)

# ─── Paths ──────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def stage2_root(output_root: Path | None = None) -> Path:
    """Trả về thư mục gốc output Stage 2."""
    if output_root is not None:
        return output_root
    return _repo_root() / "data" / "processed" / "stage2_landmarks"
