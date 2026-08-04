"""
Stage 2 build script — CLI để trích xuất landmark từ stage1 frames.

Luồng:
    windows.parquet + selected_frames.parquet (Stage 1)
    └─ for each clip (theo source_frame_index tăng dần):
        │   tracker = IoUPersonTracker()  ← reset khi đổi clip, không reset mỗi window
        └─ for each source frame duy nhất:
            │   frame = cv2.imread(...)
            │   detections = yolo_detector.detect(frame)
            │   tracks = tracker.update(detections)
            └─ for each track (is_present=True):
                │   crop, crop_bbox = crop_with_padding(frame, track.bbox)
                │   raw = holistic_runner.process_crop(meta, crop, crop_bbox)
                │   [tính quality, head_pose, gaze, body_pose]
                │   → FrameFeatureRow
        └─ save_raw_npz(window_dir, track_id, raw_list)
    └─ concat feature_rows → features/frames.parquet
    └─ aggregate_windows(frames_df) → features/windows.parquet

Lệnh chạy (đưa cho người dùng tự thực thi):

    # Dry run 3 window đầu
    python -m backend.ai_services.pose_gaze.dataset.stage2.build ^
        --stage1-root data/processed/stage1_merged ^
        --output-root data/processed/stage2_landmarks ^
        --fallback-weights weights/yolov8n.pt ^
        --device cpu --limit 3

    # Build riêng từng split; không dùng chung output-root
    foreach ($split in @("train", "val", "test")) {
        python -m backend.ai_services.pose_gaze.dataset.stage2.build `
            --stage1-root "data/processed/$split" `
            --output-root "data/processed/stage2_landmarks/$split" `
            --fallback-weights weights/yolov8n.pt `
            --device cpu --skip-existing
    }

    # Kiểm tra split không chồng lấn trước khi build
    python -m backend.ai_services.pose_gaze.dataset.stage2.validate_splits `
        --processed-root data/processed --splits train val test
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox
from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker

from .artifact import save_raw_npz
from .body_pose import shoulder_slope_deg, torso_lean_deg
from .common import (
    CROP_PADDING,
    MAX_TRACKS,
    TRACKER_MAX_MISSED,
    TRACKER_MIN_IOU,
    YOLO_CONF_THRESHOLD,
    YOLO_PERSON_CLASS_NAME,
    stage2_root,
)
from .crop import crop_with_padding
from .gaze import extract_gaze
from .head_pose import classify_head_direction, estimate_head_pose
from .holistic_runner import DatasetHolisticRunner
from .quality import face_quality_score, iris_present, is_gaze_valid
from .schemas import CropMeta, FrameFeatureRow, HolisticRaw
from ...holistic_landmarks import EYE_CONNECTIONS, HAND_CONNECTIONS, LIP_CONNECTIONS, POSE_CONNECTIONS

LOGGER = logging.getLogger(__name__)

# Số giây tối thiểu chênh lệch timestamp để log warning
_WARN_TIMESTAMP_GAP_MS: float = 5000.0


def _write_landmark_check(frame: np.ndarray, raws: dict[int, list[HolisticRaw]], output: Path) -> None:
    """Ghi preview kiểm tra pose + tay trên frame gốc, không vẽ face."""
    canvas = frame.copy()
    colors = [(40, 210, 40), (255, 255, 255)]

    for index, track_id in enumerate(sorted(raws)[:2]):
        raw = raws[track_id][-1]
        color = colors[index]
        meta = raw.meta
        x1, y1, x2, y2 = (int(value) for value in (meta.bbox_x1, meta.bbox_y1, meta.bbox_x2, meta.bbox_y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"track {track_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for points, edges in (
            (raw.pose_frame_lm, POSE_CONNECTIONS),
            (raw.left_hand_frame_lm, HAND_CONNECTIONS),
            (raw.right_hand_frame_lm, HAND_CONNECTIONS),
        ):
            if points is None:
                continue
            valid = np.isfinite(points[:, :2]).all(axis=1)
            pixels = np.zeros((len(points), 2), dtype=np.int32)
            pixels[valid] = np.clip(points[valid, :2], [0, 0], [canvas.shape[1] - 1, canvas.shape[0] - 1]).astype(np.int32)
            for start, end in edges:
                if start < len(points) and end < len(points) and valid[start] and valid[end]:
                    cv2.line(canvas, tuple(pixels[start]), tuple(pixels[end]), color, 2)
            for point in pixels[valid]:
                cv2.circle(canvas, tuple(point), 3, color, -1)

        if raw.face_frame_lm is not None and raw.face_lm_indices is not None:
            positions = {int(index): position for position, index in enumerate(raw.face_lm_indices)}
            face_edges = [
                (positions[start], positions[end])
                for start, end in (*EYE_CONNECTIONS, *LIP_CONNECTIONS, (10, 1), (1, 152))
                if start in positions and end in positions
            ]
            points = raw.face_frame_lm
            valid = np.isfinite(points[:, :2]).all(axis=1)
            pixels = np.zeros((len(points), 2), dtype=np.int32)
            pixels[valid] = np.clip(points[valid, :2], [0, 0], [canvas.shape[1] - 1, canvas.shape[0] - 1]).astype(np.int32)
            for start, end in face_edges:
                if valid[start] and valid[end]:
                    cv2.line(canvas, tuple(pixels[start]), tuple(pixels[end]), color, 2)
            for index in {point for edge in face_edges for point in edge}:
                cv2.circle(canvas, tuple(pixels[index]), 3, color, -1)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


# ─── YOLO loader với fallback ────────────────────────────────────────────────

def _load_yolo_detector(
    primary: Path | None,
    fallback: Path | None,
    device: str,
) -> UltralyticsPersonDetector:
    """
    Load YOLO. Kiểm tra xem model có class 'person'. Nếu không → fallback.
    Nếu fallback cũng không tồn tại → tự download yolov8n.pt qua ultralytics.
    """
    from ultralytics import YOLO

    def _has_person(model_path: Path) -> bool:
        m = YOLO(str(model_path))
        names = m.names or {}
        has = any(str(v).lower() == YOLO_PERSON_CLASS_NAME for v in names.values())
        if not has:
            LOGGER.warning(
                "Model %s không có class '%s'. Classes: %s",
                model_path.name, YOLO_PERSON_CLASS_NAME, list(names.values()),
            )
        return has

    # Thử primary trước
    if primary is not None and primary.exists():
        if _has_person(primary):
            LOGGER.info("YOLO: dùng %s", primary)
            return UltralyticsPersonDetector(
                primary, confidence_threshold=YOLO_CONF_THRESHOLD, device=device
            )
        LOGGER.warning("Primary weights '%s' không có class person → chuyển fallback.", primary)

    # Fallback
    if fallback is not None:
        if not fallback.exists():
            LOGGER.info("Fallback weights '%s' chưa có — ultralytics sẽ tự tải khi load.", fallback)
        LOGGER.info("YOLO: dùng fallback %s", fallback)
        return UltralyticsPersonDetector(
            fallback, confidence_threshold=YOLO_CONF_THRESHOLD, device=device
        )

    # Không có gì → dùng yolov8n.pt mặc định (ultralytics auto-download)
    LOGGER.warning("Không tìm thấy weights nào — dùng yolov8n.pt (ultralytics sẽ tải).")
    return UltralyticsPersonDetector(
        "yolov8n.pt", confidence_threshold=YOLO_CONF_THRESHOLD, device=device
    )


# ─── Đọc windows.parquet ─────────────────────────────────────────────────────

def _load_stage1_inputs(stage1_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc và kiểm tra hai artifact Stage 1 mà Stage 2 thực sự sử dụng.

    ``selected_frames`` là danh sách frame đã được Stage 1 chọn. Không suy ra
    timestamp hay nhãn từ tên file/window vì các window có overlap.
    """
    windows_path = stage1_root / "windows.parquet"
    selected_path = stage1_root / "selected_frames.parquet"
    for path in (windows_path, selected_path):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy artifact Stage 1: {path}")

    windows = pd.read_parquet(windows_path)
    selected = pd.read_parquet(selected_path)
    window_required = {
        "window_id", "clip_id", "class_code", "split", "split_status",
        "window_label", "include_in_training", "review_required",
    }
    selected_required = {
        "window_id", "clip_id", "source_frame_index", "window_local_index",
        "frame_path", "timestamp_s", "frame_quality_score",
    }
    missing_windows = window_required - set(windows.columns)
    missing_selected = selected_required - set(selected.columns)
    if missing_windows or missing_selected:
        details = []
        if missing_windows:
            details.append(f"windows.parquet thiếu {sorted(missing_windows)}")
        if missing_selected:
            details.append(f"selected_frames.parquet thiếu {sorted(missing_selected)}")
        raise ValueError("; ".join(details))
    if windows["window_id"].duplicated().any():
        raise ValueError("windows.parquet có window_id trùng lặp")

    # ``label`` là tên legacy của temporal phase. Chuẩn hóa cục bộ, không tạo
    # lại cột label trong Stage 0 và không dùng nó làm nhãn hành vi.
    windows = windows.copy()
    windows["window_phase"] = windows["window_label"]
    selected = selected.merge(
        windows[
            ["window_id", "class_code", "split", "split_status", "window_phase",
             "include_in_training", "review_required", "label_confidence"]
        ],
        on="window_id",
        how="left",
        validate="many_to_one",
    )
    if selected["class_code"].isna().any():
        raise ValueError("selected_frames.parquet chứa window_id không có trong windows.parquet")
    return windows, selected


def _load_windows(stage1_root: Path) -> pd.DataFrame:
    """Ưu tiên nhãn derived đã duyệt, không rơi về temporal label cũ."""
    ready_path = stage1_root / "windows_training_ready.parquet"
    windows_path = ready_path if ready_path.exists() else stage1_root / "windows.parquet"
    if not windows_path.exists():
        raise FileNotFoundError(f"Không tìm thấy windows.parquet: {windows_path}")
    windows = pd.read_parquet(windows_path)
    required = {"window_id", "clip_id", "frame_dir", "class_code", "split"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Stage 1 windows thiếu cột: {sorted(missing)}")
    if windows_path == ready_path and "binary_label" not in windows.columns:
        raise ValueError("windows_training_ready.parquet thiếu binary_label")
    return windows


def _iter_frame_paths(frames_root: Path, frame_dir: str) -> Iterator[Path]:
    """Yield frame PNG của một window theo thứ tự local index."""
    window_dir = frames_root / frame_dir
    if not window_dir.exists():
        LOGGER.warning("Thư mục frame không tồn tại: %s", window_dir)
        return
    yield from sorted(window_dir.glob("frame_*.png"))


# ─── Tính feature từ HolisticRaw ─────────────────────────────────────────────

def _build_feature_row(raw: HolisticRaw) -> FrameFeatureRow:
    """Chuyển HolisticRaw thành FrameFeatureRow (scalar features)."""
    meta = raw.meta

    row = FrameFeatureRow(
        clip_id=meta.clip_id,
        window_id=meta.window_id,
        frame_idx=meta.frame_idx,
        track_id=meta.track_id,
        timestamp_ms=meta.timestamp_ms,
        tracking_confidence=meta.tracking_confidence,
        bbox_x1=meta.bbox_x1,
        bbox_y1=meta.bbox_y1,
        bbox_x2=meta.bbox_x2,
        bbox_y2=meta.bbox_y2,
        crop_x1=meta.crop_x1,
        crop_y1=meta.crop_y1,
        crop_x2=meta.crop_x2,
        crop_y2=meta.crop_y2,
        frame_h=meta.frame_h,
        frame_w=meta.frame_w,
        face_quality_score=raw.face_quality_score,
        gaze_valid=raw.gaze_valid,
        missing_pose=raw.missing_pose,
        missing_face=raw.missing_face,
        missing_left_hand=raw.missing_left_hand,
        missing_right_hand=raw.missing_right_hand,
    )

    # ── Head pose ──────────────────────────────────────────────────────────
    if raw.face_crop_lm is not None:
        crop_h = int(meta.crop_y2 - meta.crop_y1)
        crop_w = int(meta.crop_x2 - meta.crop_x1)
        result = estimate_head_pose(raw.face_crop_lm, crop_w, crop_h, face_indices=raw.face_lm_indices)
        if result is not None:
            yaw, pitch, roll = result
            row.head_yaw   = round(yaw,   2)
            row.head_pitch = round(pitch, 2)
            row.head_roll  = round(roll,  2)
            row.head_direction = classify_head_direction(yaw, pitch)

    # ── Gaze (TÁCH BIỆT — chỉ khi face valid + iris) ──────────────────────
    if raw.face_crop_lm is not None:
        direction, valid = extract_gaze(raw.face_crop_lm, raw.face_quality_score, face_indices=raw.face_lm_indices)
        row.gaze_direction = direction
        row.gaze_valid     = valid

    # ── Body pose ──────────────────────────────────────────────────────────
    if raw.pose_crop_lm is not None and raw.pose_vis is not None:
        row.shoulder_slope_deg = shoulder_slope_deg(raw.pose_crop_lm, raw.pose_vis)
        row.torso_lean_deg     = torso_lean_deg(raw.pose_crop_lm, raw.pose_vis)

    return row


# ─── Aggregate theo window ────────────────────────────────────────────────────

def _aggregate_windows(frames_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các thống kê aggregate theo (window_id, track_id).

    Mỗi feature dạng float → mean, std, min, max.
    Cột categorical (head_direction, gaze_direction) → mode.
    """
    if frames_df.empty:
        return pd.DataFrame()

    float_cols = [
        "head_yaw", "head_pitch", "head_roll",
        "shoulder_slope_deg", "torso_lean_deg",
        "face_quality_score", "tracking_confidence",
    ]
    cat_cols = ["head_direction", "gaze_direction"]
    flag_cols = ["missing_pose", "missing_face", "missing_left_hand", "missing_right_hand", "gaze_valid"]

    group = frames_df.groupby(["window_id", "track_id", "clip_id", "frame_idx"], sort=False)
    # Khôi phục groupby đúng
    group = frames_df.groupby(["window_id", "track_id"], sort=False)

    agg_dict: dict = {}
    for col in float_cols:
        if col in frames_df.columns:
            agg_dict[col] = ["mean", "std", "min", "max"]
    for col in flag_cols:
        if col in frames_df.columns:
            agg_dict[col] = "mean"  # tỷ lệ frame có flag=True

    if not agg_dict:
        return pd.DataFrame()

    agg = group.agg(agg_dict)
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg = agg.reset_index()

    # Mode cho categorical columns
    for col in cat_cols:
        if col in frames_df.columns:
            mode_series = (
                frames_df.groupby(["window_id", "track_id"])[col]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
            )
            agg = agg.merge(
                mode_series.rename(f"{col}_mode").reset_index(),
                on=["window_id", "track_id"],
                how="left",
            )

    # Thêm metadata của window (lấy giá trị đầu tiên)
    meta_cols = ["clip_id", "class_code", "label", "frame_w", "frame_h"]
    available_meta = [c for c in meta_cols if c in frames_df.columns]
    if available_meta:
        meta_df = frames_df.groupby(["window_id", "track_id"])[available_meta].first().reset_index()
        agg = agg.merge(meta_df, on=["window_id", "track_id"], how="left")

    return agg


# ─── Main build function ──────────────────────────────────────────────────────

def build_stage2(
    *,
    stage1_root_path: Path,
    output_root: Path,
    primary_weights: Path | None,
    fallback_weights: Path | None,
    device: str = "cpu",
    limit: int | None = None,
    skip_existing: bool = False,
    visualize_check: bool = False,
    visualize_root: Path | None = None,
) -> None:
    """
    Chạy toàn bộ Stage 2 pipeline.

    Không tự gọi trong code — chỉ dùng qua CLI hoặc test.
    """
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    windows_df = _load_windows(stage1_root_path)
    frames_root = stage1_root_path / "frames"

    raw_root     = output_root / "raw"
    features_dir = output_root / "features"
    raw_root.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Tải YOLO detector...")
    detector = _load_yolo_detector(primary_weights, fallback_weights, device)

    if limit is not None:
        windows_df = windows_df.head(limit)

    all_feature_rows: list[dict] = []
    # Track ID chỉ có ý nghĩa trong một clip; không reset khi chuyển window.
    trackers: dict[str, IoUPersonTracker] = {}

    iterator = windows_df.iterrows()
    if _tqdm_available:
        from tqdm import tqdm
        iterator = tqdm(windows_df.iterrows(), total=len(windows_df), desc="Stage2 windows")

    with DatasetHolisticRunner() as holistic_runner:
        for _, win_row in iterator:
            window_id = str(win_row["window_id"])
            clip_id   = str(win_row["clip_id"])
            frame_dir = str(win_row["frame_dir"])
            label     = str(win_row.get("label", "unknown"))
            class_code = str(win_row.get("class_code", "unknown"))
            binary_label = str(win_row.get("binary_label", "unknown"))
            window_phase = str(win_row.get("window_phase", win_row.get("window_label", label)))
            split = str(win_row.get("split", "unknown"))
            window_start_ms = float(win_row.get("window_start_s", 0.0)) * 1000.0
            target_fps = float(win_row.get("target_fps", 10.0))

            # Thư mục output cho window này
            window_raw_dir = raw_root / clip_id / window_id

            # Skip nếu đã có output (resume sau crash)
            if skip_existing and any(window_raw_dir.glob("track_*.npz")):
                LOGGER.debug("Skip (đã có): %s", window_id)
                continue

            frame_paths = list(_iter_frame_paths(frames_root, frame_dir))
            if not frame_paths:
                LOGGER.warning("Không có frame trong: %s", frame_dir)
                continue

            tracker = trackers.setdefault(
                clip_id,
                IoUPersonTracker(
                    max_tracks=MAX_TRACKS,
                    min_iou=TRACKER_MIN_IOU,
                    max_missed_frames=TRACKER_MAX_MISSED,
                ),
            )

            # Lưu raw per track_id
            track_raws: dict[int, list[HolisticRaw]] = {}
            last_frame: np.ndarray | None = None

            for frame_idx, frame_path in enumerate(frame_paths):
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    LOGGER.warning("Không đọc được: %s", frame_path)
                    continue

                frame_h, frame_w = frame_bgr.shape[:2]
                last_frame = frame_bgr

                # YOLO detect → tracker update
                detections = detector.detect(frame_bgr)
                tracked = tracker.update(detections)

                # Timestamp Stage 1 là timestamp nguồn từ đầu clip, không phải
                # local index của window chồng lấp.
                timestamp_ms = window_start_ms + (frame_idx * 1000.0 / target_fps)

                for person in tracked:
                    if not person.is_present:
                        continue

                    # Crop với 15% padding
                    crop_bgr, crop_bbox = crop_with_padding(frame_bgr, person.bbox, CROP_PADDING)

                    meta = CropMeta(
                        clip_id=clip_id,
                        window_id=window_id,
                        frame_idx=frame_idx,
                        track_id=person.track_id,
                        bbox_x1=person.bbox.x1,
                        bbox_y1=person.bbox.y1,
                        bbox_x2=person.bbox.x2,
                        bbox_y2=person.bbox.y2,
                        crop_x1=crop_bbox.x1,
                        crop_y1=crop_bbox.y1,
                        crop_x2=crop_bbox.x2,
                        crop_y2=crop_bbox.y2,
                        frame_h=frame_h,
                        frame_w=frame_w,
                        timestamp_ms=timestamp_ms,
                        tracking_confidence=person.confidence,
                    )

                    # MediaPipe Holistic
                    raw = holistic_runner.process_crop(meta, crop_bgr, crop_bbox, full_frame=frame_bgr)

                    # Quality score
                    crop_h_px = int(crop_bbox.y2 - crop_bbox.y1)
                    crop_w_px = int(crop_bbox.x2 - crop_bbox.x1)
                    raw.face_quality_score = face_quality_score(
                        raw.face_crop_lm, raw.face_vis, crop_h_px, crop_w_px, face_indices=raw.face_lm_indices
                    )
                    has_iris = raw.face_lm_indices is not None and all(idx in raw.face_lm_indices for idx in (474, 475, 476, 477))
                    raw.gaze_valid = is_gaze_valid(raw.face_quality_score, has_iris)

                    track_raws.setdefault(person.track_id, []).append(raw)

                    # Feature row
                    feat_row = _build_feature_row(raw)
                    d = feat_row.to_dict()
                    d["label"]      = label
                    d["class_code"] = class_code
                    d["binary_label"] = binary_label
                    d["window_phase"] = window_phase
                    d["split"] = split
                    all_feature_rows.append(d)

            # Lưu raw .npz cho mỗi track
            for track_id, raws in track_raws.items():
                npz_path = save_raw_npz(window_raw_dir, track_id, raws)
                LOGGER.debug("Đã lưu: %s", npz_path)

            if visualize_check and last_frame is not None:
                check_root = visualize_root or (output_root / "visual_checks")
                check_path = check_root / split / f"{clip_id}_{window_id}.png"
                _write_landmark_check(last_frame, track_raws, check_path)
                LOGGER.info("Landmark check: %s", check_path)

    # ── Ghi feature parquet ────────────────────────────────────────────────
    if not all_feature_rows:
        LOGGER.warning("Không có feature row nào được tạo ra.")
        return

    frames_df = pd.DataFrame(all_feature_rows)
    frames_path = features_dir / "frames.parquet"
    frames_df.to_parquet(frames_path, index=False)
    LOGGER.info("Đã ghi %d hàng → %s", len(frames_df), frames_path)

    windows_agg = _aggregate_windows(frames_df)
    if not windows_agg.empty:
        windows_path = features_dir / "windows.parquet"
        windows_agg.to_parquet(windows_path, index=False)
        LOGGER.info("Đã ghi %d windows → %s", len(windows_agg), windows_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 2: YOLO tracking + MediaPipe Holistic landmark extraction"
    )
    parser.add_argument(
        "--stage1-root", type=Path, required=True,
        help="Thư mục stage1_merged chứa windows.parquet và frames/",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Thư mục output stage2_landmarks (mặc định: data/processed/stage2_landmarks)",
    )
    parser.add_argument(
        "--weights", type=Path, default=None, dest="primary_weights",
        help="Path tới YOLO weights chính (vd: weights/yolov8_finetuned.pt)",
    )
    parser.add_argument(
        "--fallback-weights", type=Path, default=None,
        help="Fallback weights nếu primary không có class person (vd: weights/yolov8n.pt)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device cho YOLO: 'cpu' hoặc 'cuda' (MediaPipe luôn chạy CPU trên Windows)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Chỉ xử lý N window đầu (dry run)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Bỏ qua window đã có file .npz (resume sau crash)",
    )
    parser.add_argument(
        "--visualize-check", action="store_true",
        help="Xuất preview pose + tay sau mỗi window để kiểm tra trước khi resume",
    )
    parser.add_argument(
        "--visualize-root", type=Path, default=None,
        help="Thư mục preview kiểm tra (mặc định: output-root/visual_checks)",
    )

    args = parser.parse_args()
    out_root = stage2_root(args.output_root)

    LOGGER.info("=== Stage 2 build ===")
    LOGGER.info("stage1-root  : %s", args.stage1_root)
    LOGGER.info("output-root  : %s", out_root)
    LOGGER.info("primary      : %s", args.primary_weights)
    LOGGER.info("fallback     : %s", args.fallback_weights)
    LOGGER.info("device       : %s", args.device)
    LOGGER.info("limit        : %s", args.limit)
    LOGGER.info("skip-existing: %s", args.skip_existing)
    LOGGER.info("visualize-check: %s", args.visualize_check)

    build_stage2(
        stage1_root_path=args.stage1_root,
        output_root=out_root,
        primary_weights=args.primary_weights,
        fallback_weights=args.fallback_weights,
        device=args.device,
        limit=args.limit,
        skip_existing=args.skip_existing,
        visualize_check=args.visualize_check,
        visualize_root=args.visualize_root,
    )
    LOGGER.info("=== Stage 2 hoàn thành ===")


if __name__ == "__main__":
    main()
