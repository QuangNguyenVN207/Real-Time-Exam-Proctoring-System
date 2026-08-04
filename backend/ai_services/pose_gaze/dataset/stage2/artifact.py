"""
Lưu và tải raw landmark artifact dạng nén .npz.

Thiết kế:
- 1 file .npz per window per track_id
- Array được stack theo axis=0 (dim 0 = frame index)
- Frame không có landmark → NaN row thay vì bỏ qua (giữ alignment)
- Không lưu ảnh RGB

Sau khi có .npz, có thể tính lại feature mà KHÔNG cần chạy lại video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import CropMeta, HolisticRaw


# Tên các mảng lưu trong .npz
_KEYS = [
    "pose_crop_lm",
    "pose_frame_lm",
    "pose_vis",
    "pose_presence",
    "face_crop_lm",
    "face_frame_lm",
    "face_vis",
    "left_hand_crop_lm",
    "left_hand_frame_lm",
    "right_hand_crop_lm",
    "right_hand_frame_lm",
]

# Kích thước mặc định của từng array (dùng để tạo NaN row)
_ARRAY_SHAPES: dict[str, tuple[int, ...]] = {
    "pose_crop_lm":        (33, 3),
    "pose_frame_lm":       (33, 3),
    "pose_vis":            (33,),
    "pose_presence":       (33,),
    "face_crop_lm":        (109, 3),  # 109 selected face landmarks
    "face_frame_lm":       (109, 3),
    "face_vis":            (109,),
    "left_hand_crop_lm":   (21, 3),
    "left_hand_frame_lm":  (21, 3),
    "right_hand_crop_lm":  (21, 3),
    "right_hand_frame_lm": (21, 3),
}

# Scalar metadata lưu cùng npz
_SCALAR_KEYS = [
    "face_quality_score",
    "gaze_valid",
    "missing_pose",
    "missing_face",
    "missing_left_hand",
    "missing_right_hand",
    # CropMeta scalars
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    "frame_h", "frame_w",
    "timestamp_ms",
    "tracking_confidence",
    "track_id",
    "frame_idx",
]


def _nan_row(key: str, n_frames_so_far: int) -> np.ndarray:
    """Tạo một hàng NaN có shape đúng với key."""
    shape = _ARRAY_SHAPES.get(key)
    if shape is None:
        return np.array([np.nan])
    return np.full(shape, np.nan, dtype=np.float32)


def save_raw_npz(
    window_dir: Path,
    track_id: int,
    frames: list[HolisticRaw],
) -> Path:
    """
    Lưu danh sách HolisticRaw của một track trong một window vào .npz.

    Parameters
    ----------
    window_dir : thư mục chứa window (sẽ được tạo nếu chưa có)
    track_id   : định danh track
    frames     : danh sách HolisticRaw theo thứ tự frame (có thể rỗng)

    Returns
    -------
    Path tới file .npz đã ghi
    """
    window_dir.mkdir(parents=True, exist_ok=True)
    out_path = window_dir / f"track_{track_id:02d}.npz"

    if not frames:
        np.savez_compressed(out_path)
        return out_path

    # Stack mảng theo frame axis
    stacked: dict[str, np.ndarray] = {}
    for key in _KEYS:
        # Tự động xác định shape thực tế của key từ các frame non-None
        target_shape = None
        for raw in frames:
            arr = getattr(raw, key, None)
            if arr is not None:
                target_shape = arr.shape
                break
        if target_shape is None:
            target_shape = _ARRAY_SHAPES.get(key, (0, 3))

        rows = []
        for raw in frames:
            arr = getattr(raw, key, None)
            if arr is not None and arr.shape == target_shape:
                rows.append(arr.astype(np.float32))
            else:
                rows.append(np.full(target_shape, np.nan, dtype=np.float32))
        stacked[key] = np.stack(rows, axis=0)

    # Stack scalar metadata
    scalar_arrays: dict[str, np.ndarray] = {}
    for sk in _SCALAR_KEYS:
        values = []
        for raw in frames:
            if sk in ("track_id", "frame_idx", "frame_h", "frame_w",
                      "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                      "crop_x1", "crop_y1", "crop_x2", "crop_y2",
                      "timestamp_ms", "tracking_confidence"):
                val = getattr(raw.meta, sk, np.nan)
            else:
                val = getattr(raw, sk, np.nan)
            # bool → float
            values.append(float(val) if val is not None else np.nan)
        scalar_arrays[sk] = np.array(values, dtype=np.float32)

    np.savez_compressed(out_path, **stacked, **scalar_arrays)
    return out_path


def load_raw_npz(path: Path) -> dict[str, np.ndarray]:
    """
    Tải lại .npz thành dict mảng numpy.

    Returns
    -------
    dict với keys = _KEYS + _SCALAR_KEYS.
    Dùng để tính lại feature mà không chạy lại MediaPipe.
    """
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k] for k in data.files}
