"""
Debug script — in chi tiết HolisticRaw của 1 window / 1 frame
để chẩn đoán tại sao head_direction và gaze_direction = 'unknown'.

Chạy từ repo root:
    python -m backend.ai_services.pose_gaze.dataset.stage2.debug_pipeline \
        --stage1-root data/processed/stage1_merged \
        --fallback-weights weights/yolov8n.pt \
        --device cpu
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox

from .common import (
    CROP_PADDING, MAX_TRACKS, TRACKER_MAX_MISSED, TRACKER_MIN_IOU,
    YOLO_CONF_THRESHOLD, YOLO_PERSON_CLASS_NAME,
)
from .crop import crop_with_padding
from .holistic_runner import DatasetHolisticRunner
from .quality import face_quality_score, is_gaze_valid
from .head_pose import estimate_head_pose, classify_head_direction
from .gaze import extract_gaze
from .schemas import CropMeta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Stage 2 pipeline (1 window)")
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--fallback-weights", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=3,
                        help="Số frame tối đa để xử lý")
    args = parser.parse_args()

    frames_root = args.stage1_root / "frames"

    # Lấy window đầu tiên
    import pandas as pd
    windows_df = pd.read_parquet(args.stage1_root / "windows.parquet")
    win_row = windows_df.iloc[0]
    frame_dir = str(win_row["frame_dir"])
    clip_id   = str(win_row["clip_id"])
    window_id = str(win_row["window_id"])

    frame_paths = sorted((frames_root / frame_dir).glob("frame_*.png"))[:args.max_frames]
    if not frame_paths:
        LOGGER.error("Không có frame nào trong: %s", frames_root / frame_dir)
        return

    LOGGER.info("Window: %s | Frames: %d", window_id, len(frame_paths))

    # Load YOLO
    from ultralytics import YOLO as _YOLO
    fw = args.fallback_weights or Path("yolov8n.pt")
    detector = UltralyticsPersonDetector(fw, confidence_threshold=YOLO_CONF_THRESHOLD, device=args.device)

    tracker = IoUPersonTracker(
        max_tracks=MAX_TRACKS, min_iou=TRACKER_MIN_IOU,
        max_missed_frames=TRACKER_MAX_MISSED,
    )

    with DatasetHolisticRunner() as runner:
        LOGGER.info("Backend: %s", runner._extractor.backend)
        LOGGER.info("Selected face indices count: %d", len(runner._extractor.selected_face_indices))
        LOGGER.info("Selected face indices: %s", sorted(runner._extractor.selected_face_indices))

        for frame_idx, frame_path in enumerate(frame_paths):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue

            frame_h, frame_w = frame_bgr.shape[:2]
            detections = detector.detect(frame_bgr)
            tracked = tracker.update(detections)
            timestamp_ms = frame_idx * 100.0

            LOGGER.info("\n=== Frame %d (%s) — %d detections, %d tracks ===",
                        frame_idx, frame_path.name, len(detections), len(tracked))

            for person in tracked:
                if not person.is_present:
                    continue

                crop_bgr, crop_bbox = crop_with_padding(frame_bgr, person.bbox, CROP_PADDING)
                meta = CropMeta(
                    clip_id=clip_id, window_id=window_id, frame_idx=frame_idx,
                    track_id=person.track_id,
                    bbox_x1=person.bbox.x1, bbox_y1=person.bbox.y1,
                    bbox_x2=person.bbox.x2, bbox_y2=person.bbox.y2,
                    crop_x1=crop_bbox.x1, crop_y1=crop_bbox.y1,
                    crop_x2=crop_bbox.x2, crop_y2=crop_bbox.y2,
                    frame_h=frame_h, frame_w=frame_w,
                    timestamp_ms=timestamp_ms,
                    tracking_confidence=person.confidence,
                )

                raw = runner.process_crop(meta, crop_bgr, crop_bbox, full_frame=frame_bgr)

                LOGGER.info("\n  Track %d:", person.track_id)
                LOGGER.info("    missing_face: %s", raw.missing_face)
                LOGGER.info("    missing_pose: %s", raw.missing_pose)

                if raw.face_crop_lm is not None:
                    LOGGER.info("    face_crop_lm shape: %s", raw.face_crop_lm.shape)
                    LOGGER.info("    face_lm_indices: %s", sorted(raw.face_lm_indices.tolist()) if raw.face_lm_indices is not None else None)
                    LOGGER.info("    face_lm count: %d", len(raw.face_lm_indices) if raw.face_lm_indices is not None else 0)

                    # Kiểm tra các anchor index có mặt không
                    anchor_indices = [1, 152, 33, 263, 61, 291]
                    if raw.face_lm_indices is not None:
                        present = [idx for idx in anchor_indices if idx in raw.face_lm_indices]
                        missing_anchors = [idx for idx in anchor_indices if idx not in raw.face_lm_indices]
                        LOGGER.info("    Head pose anchors present: %s", present)
                        LOGGER.info("    Head pose anchors MISSING: %s", missing_anchors)

                    # Kiểm tra iris
                    iris_indices = [469, 470, 471, 472, 474, 475, 476, 477]
                    if raw.face_lm_indices is not None:
                        iris_present_list = [idx for idx in iris_indices if idx in raw.face_lm_indices]
                        LOGGER.info("    Iris indices present: %s", iris_present_list)

                    # Tính quality
                    crop_h_px = int(crop_bbox.y2 - crop_bbox.y1)
                    crop_w_px = int(crop_bbox.x2 - crop_bbox.x1)
                    q = face_quality_score(
                        raw.face_crop_lm, raw.face_vis, crop_h_px, crop_w_px,
                        face_indices=raw.face_lm_indices
                    )
                    LOGGER.info("    face_quality_score: %.4f", q)

                    # Thử head pose
                    from backend.ai_services.pose_gaze.dataset.stage2.head_pose import _MODEL_POINTS
                    LOGGER.info("    head_pose _MODEL_POINTS[1]: %s", _MODEL_POINTS[1])
                    result = estimate_head_pose(raw.face_crop_lm, crop_w_px, crop_h_px, face_indices=raw.face_lm_indices)
                    if result is not None:
                        yaw, pitch, roll = result
                        direction = classify_head_direction(yaw, pitch)
                        LOGGER.info("    head_pose: yaw=%.1f pitch=%.1f roll=%.1f → %s", yaw, pitch, roll, direction)
                    else:
                        LOGGER.warning("    head_pose: estimate_head_pose trả None!")

                    # Thử gaze
                    has_iris = raw.face_lm_indices is not None and all(
                        idx in raw.face_lm_indices for idx in (474, 475, 476, 477)
                    )
                    LOGGER.info("    has_iris (474-477): %s", has_iris)
                    gaze_dir, gaze_valid = extract_gaze(raw.face_crop_lm, q, face_indices=raw.face_lm_indices)
                    LOGGER.info("    gaze: %s (valid=%s)", gaze_dir, gaze_valid)
                else:
                    LOGGER.info("    face_crop_lm: None (không phát hiện mặt)")

                if raw.pose_crop_lm is not None:
                    LOGGER.info("    pose_crop_lm shape: %s", raw.pose_crop_lm.shape)
                else:
                    LOGGER.info("    pose_crop_lm: None")


if __name__ == "__main__":
    main()
