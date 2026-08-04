"""
Body pose features từ MediaPipe Pose landmarks.

Chỉ dùng thân trên (shoulder, elbow, wrist, hip).
Trả None khi landmark bị che hoặc visibility thấp.
"""

from __future__ import annotations

import numpy as np

# MediaPipe Pose landmark indices (thân trên)
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW     = 13
_RIGHT_ELBOW    = 14
_LEFT_HIP       = 23
_RIGHT_HIP      = 24

_MIN_VIS: float = 0.5  # visibility tối thiểu để landmark tin cậy


def _vis_ok(vis: np.ndarray, idx: int, threshold: float = _MIN_VIS) -> bool:
    """Kiểm tra visibility của một landmark index."""
    return len(vis) > idx and float(vis[idx]) >= threshold


def shoulder_slope_deg(
    pose_lm: np.ndarray,    # (33, 3) crop-normalized
    pose_vis: np.ndarray,   # (33,)
    min_visibility: float = _MIN_VIS,
) -> float | None:
    """
    Góc nghiêng của đường vai (landmark 11 → 12) so với trục ngang.

    Dương: vai phải cao hơn vai trái (thân nghiêng phải).
    Âm:   vai trái cao hơn vai phải (thân nghiêng trái).
    Trả None nếu một trong 2 vai không đủ visibility.
    """
    if not (_vis_ok(pose_vis, _LEFT_SHOULDER, min_visibility)
            and _vis_ok(pose_vis, _RIGHT_SHOULDER, min_visibility)):
        return None

    ls = pose_lm[_LEFT_SHOULDER,  :2]   # (x, y) crop-normalized
    rs = pose_lm[_RIGHT_SHOULDER, :2]

    # Delta pixel (không cần chuyển pixel vì đây là tỷ lệ)
    dx = float(rs[0] - ls[0])
    dy = float(rs[1] - ls[1])   # y tăng xuống dưới trong ảnh

    angle = float(np.degrees(np.arctan2(-dy, dx)))  # âm dy → dương angle khi vai phải cao hơn
    return angle


def torso_lean_deg(
    pose_lm: np.ndarray,
    pose_vis: np.ndarray,
    min_visibility: float = _MIN_VIS,
) -> float | None:
    """
    Góc nghiêng thân trên: từ trung điểm vai đến trung điểm hông.

    0°   = thẳng đứng.
    Dương: thân nghiêng về phía trước/phải.
    Âm:   thân nghiêng về phía sau/trái.
    Trả None nếu thiếu bất kỳ landmark nào trong 4 điểm.
    """
    required = [_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]
    if not all(_vis_ok(pose_vis, idx, min_visibility) for idx in required):
        return None

    shoulder_mid = (pose_lm[_LEFT_SHOULDER, :2] + pose_lm[_RIGHT_SHOULDER, :2]) / 2.0
    hip_mid      = (pose_lm[_LEFT_HIP,      :2] + pose_lm[_RIGHT_HIP,      :2]) / 2.0

    dx = float(hip_mid[0] - shoulder_mid[0])
    dy = float(hip_mid[1] - shoulder_mid[1])   # y tăng xuống trong ảnh

    # Góc so với trục dọc (0 = thẳng đứng)
    angle = float(np.degrees(np.arctan2(dx, dy)))
    return angle
