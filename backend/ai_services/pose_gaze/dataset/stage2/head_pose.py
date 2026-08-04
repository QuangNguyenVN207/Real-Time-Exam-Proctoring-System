"""
Head pose estimation từ MediaPipe Face Mesh landmarks.

Dùng cv2.solvePnP với 6 anchor points để tính góc Euler (yaw, pitch, roll).
Camera intrinsics được xấp xỉ từ kích thước crop (không cần calibration data).

TÁCH BIỆT khỏi gaze — hàm này chỉ tính hướng đầu từ face mesh,
KHÔNG dùng iris.
"""

from __future__ import annotations
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

# ─── 6 anchor points trên mô hình 3D mặt chuẩn (mm) ────────────────────────
# Thứ tự: mũi tip (1), cằm (152), mắt trái ngoài (33), mắt phải ngoài (263),
#          miệng trái (61), miệng phải (291)
_ANCHOR_FACE_INDICES: list[int] = [1, 152, 33, 263, 61, 291]

# Tọa độ 3D model (mm) tương ứng với các index trên — hệ tọa độ OpenCV
_MODEL_POINTS: np.ndarray = np.array(
    [
        [0.0,    0.0,    0.0   ],   # mũi tip
        [0.0,  330.0,   65.0   ],   # cằm (152) (y dương, z dương)
        [-225.0, -170.0, 135.0  ],  # mắt trái ngoài (33) (nằm bên trái ảnh -> x âm, y âm, z dương)
        [225.0,  -170.0, 135.0  ],  # mắt phải ngoài (263) (nằm bên phải ảnh -> x dương, y âm, z dương)
        [-150.0, 150.0,  125.0  ],  # miệng trái (61) (nằm bên trái ảnh -> x âm, y dương, z dương)
        [150.0,  150.0,  125.0  ],  # miệng phải (291) (nằm bên phải ảnh -> x dương, y dương, z dương)
    ],
    dtype=np.float64,
)

# Ngưỡng phân loại hướng đầu (độ)
_YAW_THRESHOLD: float = 15.0   # > 15° sang trái/phải
_PITCH_THRESHOLD: float = 15.0  # > 15° xuống dưới


def estimate_head_pose(
    face_lm: np.ndarray,                   # (N, 3) crop-normalized (x,y ∈ [0,1], z relative)
    crop_w: int,
    crop_h: int,
    face_indices: np.ndarray | None = None, # (N,) MediaPipe landmark indices tương ứng
) -> tuple[float, float, float] | None:
    """
    Ước tính góc Euler đầu từ face landmarks.

    Parameters
    ----------
    face_lm      : array shape (N, 3) — tọa độ crop-normalized từ MediaPipe
    crop_w       : chiều rộng crop (pixel)
    crop_h       : chiều cao crop (pixel)
    face_indices : array shape (N,) — index MediaPipe tương ứng của mỗi hàng trong face_lm

    Returns
    -------
    (yaw_deg, pitch_deg, roll_deg) hoặc None nếu solvePnP thất bại.
      yaw  : âm = đầu quay trái, dương = quay phải
      pitch: âm = đầu ngẩng lên, dương = cúi xuống
      roll : âm = nghiêng trái, dương = nghiêng phải
    """
    try:
        import cv2
    except ImportError:
        return None

    if face_lm is None or face_lm.size == 0:
        return None

    # Tìm vị trí của 6 anchor points trong face_lm
    img_pts = []
    for idx in _ANCHOR_FACE_INDICES:
        if face_indices is not None:
            matches = np.where(face_indices == idx)[0]
            if len(matches) == 0:
                LOGGER.debug("head_pose: anchor index %d không có trong face_lm_indices (N=%d)", idx, len(face_indices))
                return None
            row_idx = matches[0]
        else:
            if idx >= face_lm.shape[0]:
                LOGGER.debug("head_pose: anchor index %d vượt kích thước face_lm (%d)", idx, face_lm.shape[0])
                return None
            row_idx = idx
        px = face_lm[row_idx, 0] * crop_w
        py = face_lm[row_idx, 1] * crop_h
        img_pts.append([px, py])

    image_points = np.array(img_pts, dtype=np.float64)

    # Kiểm tra image_points hợp lệ (không phải NaN/Inf, không quá xa crop)
    if not np.isfinite(image_points).all():
        LOGGER.debug("head_pose: image_points chứa NaN/Inf → bỏ qua")
        return None
    # Nếu tất cả các điểm nằm ngoài crop (có thể do letterbox padding)
    in_bounds = (
        (image_points[:, 0] >= -crop_w) & (image_points[:, 0] <= 2 * crop_w)
        & (image_points[:, 1] >= -crop_h) & (image_points[:, 1] <= 2 * crop_h)
    )
    if not in_bounds.any():
        LOGGER.debug("head_pose: tất cả anchor points nằm ngoài crop bounds → bỏ qua")
        return None

    # Camera intrinsics đơn giản: focal ≈ crop_w, principal = tâm crop
    focal_length = float(crop_w)
    camera_matrix = np.array(
        [
            [focal_length, 0.0,          crop_w / 2.0],
            [0.0,          focal_length, crop_h / 2.0],
            [0.0,          0.0,          1.0          ],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([0.0, 0.0, 1000.0], dtype=np.float64).reshape(3, 1)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    LOGGER.info("head_pose: image_points:\n%s", image_points)
    LOGGER.info("head_pose: success=%s, rvec=%s, tvec=%s", success, rvec.ravel() if rvec is not None else None, tvec.ravel() if tvec is not None else None)
    if not success:
        return None

    # Chuyển rotation vector sang ma trận rồi sang góc Euler
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Góc Euler từ rotation matrix (ZYX convention)
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
        yaw   = float(np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])))
        roll  = float(np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])))
    else:
        pitch = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
        yaw   = 0.0
        roll  = float(np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])))

    return yaw, pitch, roll


def classify_head_direction(yaw: float, pitch: float) -> str:
    """
    Phân loại hướng đầu dựa trên góc Euler.

    Returns
    -------
    'forward' | 'left' | 'right' | 'down' | 'unknown'

    'unknown' không được dùng ở đây (head pose luôn có kết quả nếu solvePnP thành công).
    Caller phải truyền None khi estimate_head_pose trả None.
    """
    if pitch > _PITCH_THRESHOLD:
        return "down"
    if yaw < -_YAW_THRESHOLD:
        return "left"
    if yaw > _YAW_THRESHOLD:
        return "right"
    return "forward"
