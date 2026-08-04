from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .common import processed_root
from .manifest import build_manifest, parse_video_filename


@lru_cache(maxsize=4)
def _load_person_detector(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


@lru_cache(maxsize=4)
def _load_object_detector(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def _load_mediapipe_models():
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, face_mesh, pose


def _expand_bbox(bbox_xyxy: list[int] | tuple[int, int, int, int], frame_shape: tuple[int, int, int], pad_ratio: float = 0.12) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    pad_x = int(round((x2 - x1) * pad_ratio))
    pad_y = int(round((y2 - y1) * pad_ratio))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    return left, top, right, bottom


def _crop_frame(frame: np.ndarray, bbox_xyxy: list[int] | tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    left, top, right, bottom = _expand_bbox(bbox_xyxy, frame.shape)
    crop = frame[top:bottom, left:right]
    return crop, (left, top, right, bottom)


def _rotation_vector_to_euler(rotation_vector: np.ndarray) -> tuple[float, float, float]:
    import cv2

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = 0.0

    return float(np.degrees(y_angle)), float(np.degrees(x_angle)), float(np.degrees(z_angle))


def _estimate_head_pose(face_landmarks, crop_shape: tuple[int, int, int]) -> tuple[float, float, float] | None:
    import cv2

    if not face_landmarks:
        return None

    height, width = crop_shape[:2]
    image_points = np.array(
        [
            (face_landmarks.landmark[1].x * width, face_landmarks.landmark[1].y * height),
            (face_landmarks.landmark[152].x * width, face_landmarks.landmark[152].y * height),
            (face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height),
            (face_landmarks.landmark[263].x * width, face_landmarks.landmark[263].y * height),
            (face_landmarks.landmark[61].x * width, face_landmarks.landmark[61].y * height),
            (face_landmarks.landmark[291].x * width, face_landmarks.landmark[291].y * height),
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    return _rotation_vector_to_euler(rotation_vector)


def _infer_gaze_direction(yaw: float | None, pitch: float | None, quality_ok: bool) -> str:
    if not quality_ok or yaw is None or pitch is None or np.isnan(yaw) or np.isnan(pitch):
        return "unknown"
    if pitch > 15:
        return "down"
    if pitch < -15:
        return "up"
    if yaw > 15:
        return "right"
    if yaw < -15:
        return "left"
    return "center"


def _score_quality(face_landmarks, pose_landmarks, crop: np.ndarray) -> tuple[bool, float]:
    import cv2

    crop_area = float(crop.shape[0] * crop.shape[1]) if crop.size else 0.0
    if crop_area <= 0.0:
        return False, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    landmark_count = len(face_landmarks.landmark) if face_landmarks else 0
    face_score = min(1.0, landmark_count / 120.0)
    blur_score_norm = min(1.0, blur_score / 120.0)
    pose_score = 1.0 if pose_landmarks else 0.0
    crop_score = min(1.0, crop_area / 50000.0)
    quality_score = (face_score + blur_score_norm + pose_score + crop_score) / 4.0
    return quality_score >= 0.45, quality_score


def _detect_objects(frame: np.ndarray, model_paths: tuple[str, str]) -> tuple[list[str], int]:
    labels: set[str] = set()
    total = 0
    for model_path in model_paths:
        try:
            model = _load_object_detector(model_path)
            result = model(frame, verbose=False)[0]
        except Exception:
            continue
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names[class_id])
            if class_name == "person":
                continue
            labels.add(class_name)
            total += 1
    return sorted(labels), total


def _assign_students(track_rows: list[dict[str, object]], subject_ids: tuple[str, ...]) -> None:
    track_rows.sort(key=lambda row: float(row["bbox_xyxy"][0]) + float(row["bbox_xyxy"][2]))
    if not subject_ids:
        return
    if len(subject_ids) == 1:
        for row in track_rows:
            row["student_id"] = subject_ids[0]
            row["peer_student_id"] = ""
        return

    for index, row in enumerate(track_rows):
        student_id = subject_ids[min(index, len(subject_ids) - 1)]
        peer_student_id = subject_ids[1 - min(index, 1)] if len(subject_ids) > 1 else ""
        row["student_id"] = student_id
        row["peer_student_id"] = peer_student_id


def build_frames(
    *,
    manifest_path: Path | None = None,
    output_path: Path | None = None,
    frame_stride: int = 3,
    limit: int | None = None,
    person_model_path: str = "weights/yolov8n.pt",
    object_model_paths: tuple[str, str] = ("weights/yolov8n.pt", "weights/yolov8_finetuned.pt"),
) -> pd.DataFrame:
    import cv2
    from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
    from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox, PersonDetection
    from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker

    manifest_path = manifest_path or (processed_root() / "manifest.parquet")
    output_path = output_path or (processed_root() / "frames.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)
    if manifest.empty:
        raise ValueError("Manifest is empty; run build_manifest first")

    person_detector = UltralyticsPersonDetector(person_model_path)
    try:
        mp, face_mesh, pose = _load_mediapipe_models()
    except Exception:
        mp = None
        face_mesh = None
        pose = None

    records: list[dict[str, object]] = []
    processed_videos = 0

    for _, row in manifest.iterrows():
        if limit is not None and processed_videos >= limit:
            break

        video_path = Path(str(row["video_path"]))
        metadata = parse_video_filename(video_path)
        tracker = IoUPersonTracker(max_tracks=2)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_index = 0
        video_rows: list[dict[str, object]] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            detections = person_detector.detect(frame)
            tracks = tracker.update(detections)
            if not tracks:
                frame_index += 1
                continue

            object_labels, object_count = _detect_objects(frame, object_model_paths)
            timestamp_ms = int(round((frame_index / float(row["fps"] or 1.0)) * 1000.0))
            track_rows: list[dict[str, object]] = []

            for track in sorted(tracks, key=lambda item: (item.bbox.x1 + item.bbox.x2) / 2.0):
                bbox_xyxy = track.bbox.to_list()
                crop, crop_bounds = _crop_frame(frame, bbox_xyxy)
                face_landmarks = None
                pose_landmarks = None
                yaw = pitch = roll = float("nan")
                quality_ok = False
                quality_score = 0.0

                if crop.size and face_mesh is not None:
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    face_result = face_mesh.process(rgb_crop)
                    if face_result.multi_face_landmarks:
                        face_landmarks = face_result.multi_face_landmarks[0]
                        pose_result = pose.process(rgb_crop) if pose is not None else None
                        pose_landmarks = pose_result.pose_landmarks if pose_result and pose_result.pose_landmarks else None
                        head_pose = _estimate_head_pose(face_landmarks, crop.shape)
                        if head_pose is not None:
                            yaw, pitch, roll = head_pose
                        quality_ok, quality_score = _score_quality(face_landmarks, pose_landmarks, crop)

                gaze_direction = _infer_gaze_direction(yaw, pitch, quality_ok)
                track_rows.append(
                    {
                        "video_stem": metadata.stem,
                        "video_path": str(video_path),
                        "class_code": metadata.class_code,
                        "subject_ids": ",".join(metadata.subject_ids),
                        "frame_id": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "track_id": int(track.track_id),
                        "bbox_xyxy": bbox_xyxy,
                        "track_confidence": float(track.confidence),
                        "student_id": "",
                        "peer_student_id": "",
                        "action": metadata.class_code,
                        "quality_ok": bool(quality_ok),
                        "quality_score": float(quality_score),
                        "gaze_direction": gaze_direction,
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "delta_yaw": float("nan"),
                        "delta_pitch": float("nan"),
                        "delta_roll": float("nan"),
                        "crop_bounds": crop_bounds,
                        "object_labels": json.dumps(object_labels, ensure_ascii=False),
                        "object_count": int(object_count),
                        "face_landmark_count": len(face_landmarks.landmark) if face_landmarks else 0,
                        "pose_landmark_count": len(pose_landmarks.landmark) if pose_landmarks else 0,
                    }
                )

            _assign_students(track_rows, metadata.subject_ids)
            if len(metadata.subject_ids) == 1:
                for track_row in track_rows:
                    track_row["peer_student_id"] = ""

            records.extend(track_rows)
            video_rows.extend(track_rows)
            frame_index += 1

        capture.release()
        processed_videos += 1

        if not video_rows:
            continue

        video_frame = pd.DataFrame(video_rows)
        for track_id, group in video_frame.groupby("track_id"):
            baseline_cutoff = int(group["timestamp_ms"].min()) + 1500
            baseline = group.loc[group["timestamp_ms"] <= baseline_cutoff]
            if baseline.empty:
                baseline = group.head(1)

            baseline_yaw = float(pd.to_numeric(baseline["yaw"], errors="coerce").median())
            baseline_pitch = float(pd.to_numeric(baseline["pitch"], errors="coerce").median())
            baseline_roll = float(pd.to_numeric(baseline["roll"], errors="coerce").median())

            mask = (video_frame["track_id"] == track_id)
            video_frame.loc[mask, "delta_yaw"] = pd.to_numeric(video_frame.loc[mask, "yaw"], errors="coerce") - baseline_yaw
            video_frame.loc[mask, "delta_pitch"] = pd.to_numeric(video_frame.loc[mask, "pitch"], errors="coerce") - baseline_pitch
            video_frame.loc[mask, "delta_roll"] = pd.to_numeric(video_frame.loc[mask, "roll"], errors="coerce") - baseline_roll

        video_frame.loc[~video_frame["quality_ok"].astype(bool), "gaze_direction"] = "unknown"
        records = records[:-len(video_rows)] + video_frame.to_dict("records")

    frames = pd.DataFrame(records)
    if frames.empty:
        raise ValueError("No frame observations were extracted")

    frames.loc[~frames["quality_ok"].astype(bool), "gaze_direction"] = "unknown"
    frames.to_parquet(output_path, index=False)

    report = (
        frames.groupby("class_code")
        .agg(rows=("frame_id", "size"), tracks=("track_id", "nunique"), quality_rate=("quality_ok", "mean"), mean_quality=("quality_score", "mean"))
        .reset_index()
    )
    report_path = output_path.with_name("extract_report.csv")
    report.to_csv(report_path, index=False)

    if mp is not None:
        face_mesh.close()
        pose.close()

    return frames
