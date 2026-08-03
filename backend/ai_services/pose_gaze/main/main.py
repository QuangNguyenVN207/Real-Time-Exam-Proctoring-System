"""Final realtime webcam demo: tracking, Holistic, and trained XGBoost."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from backend.ai_services.pose_gaze.holistic.feature_csv import (
    MODEL_FEATURE_COLUMNS,
    model_features_from_result,
)
from backend.ai_services.pose_gaze.holistic.landmark import (
    HolisticLandmarkExtractor,
    TrackHolisticResult,
)
from backend.ai_services.pose_gaze.settings import (
    DEFAULT_HOLISTIC_CONFIDENCE,
    DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    DEFAULT_MAX_MISSED_FRAMES,
    DEFAULT_MIN_IOU,
    DEFAULT_PERSON_CONFIDENCE,
)
from backend.ai_services.pose_gaze.tracking.manager import TrackingManager
from backend.ai_services.pose_gaze.tracking.webcam import (
    PersonTrackingConfig,
    PersonTrackingModule,
    ProcessingRateController,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODULE_DIR / "model" / "xgboost_model.ubj"
DEFAULT_METADATA_PATH = MODULE_DIR / "model" / "model_metadata.json"
WINDOW_NAME = "Exam Proctoring - Realtime XGBoost Classification"


@dataclass(frozen=True, slots=True)
class Prediction:
    class_code: str
    label: str
    confidence: float
    probabilities: tuple[float, ...]


def decode_prediction(raw_prediction: Any, class_count: int) -> tuple[float, ...]:
    """Normalize binary, multiclass softprob, or multiclass softmax output."""

    if class_count < 1:
        raise ValueError("class_count must be at least 1")
    values = raw_prediction.tolist() if hasattr(raw_prediction, "tolist") else raw_prediction
    while isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if not values:
            raise ValueError("XGBoost returned an empty prediction")
        if len(values) == 1 and isinstance(values[0], Sequence):
            values = values[0]
            continue
        break
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = [values]
    flat = [float(value) for value in values]

    if class_count == 1:
        return (1.0,)
    if len(flat) == class_count:
        if any(value < 0.0 for value in flat):
            raise ValueError("XGBoost probabilities must not be negative")
        total = sum(flat)
        if total <= 0.0:
            raise ValueError("XGBoost probabilities sum to zero")
        return tuple(value / total for value in flat)
    if len(flat) == 1 and class_count == 2 and 0.0 <= flat[0] <= 1.0:
        return (1.0 - flat[0], flat[0])
    if len(flat) == 1 and flat[0].is_integer():
        class_index = int(flat[0])
        if 0 <= class_index < class_count:
            return tuple(
                1.0 if index == class_index else 0.0
                for index in range(class_count)
            )
    raise ValueError(
        f"Unexpected XGBoost output shape: {len(flat)} values for "
        f"{class_count} classes"
    )


class PredictionSmoother:
    """Per-track exponential smoothing to reduce frame-to-frame label flicker."""

    def __init__(self, alpha: float = 0.30) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: dict[int, tuple[float, ...]] = {}

    def update(self, track_id: int, probabilities: Sequence[float]) -> tuple[float, ...]:
        current = tuple(float(value) for value in probabilities)
        previous = self._state.get(track_id)
        if previous is None:
            smoothed = current
        else:
            if len(previous) != len(current):
                raise ValueError("Prediction class count changed during runtime")
            smoothed = tuple(
                (1.0 - self.alpha) * old + self.alpha * new
                for old, new in zip(previous, current)
            )
        total = sum(smoothed)
        if total <= 0.0:
            raise ValueError("Smoothed probabilities sum to zero")
        normalized = tuple(value / total for value in smoothed)
        self._state[track_id] = normalized
        return normalized

    def retain(self, track_ids: set[int]) -> None:
        self._state = {
            track_id: values
            for track_id, values in self._state.items()
            if track_id in track_ids
        }


class XGBoostRuntime:
    """Load a native XGBoost model and enforce the shared feature contract."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"XGBoost model was not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"XGBoost metadata was not found: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("model_metadata.json must contain an object")
        raw_codes = metadata.get("class_codes")
        if not isinstance(raw_codes, list) or not raw_codes:
            raise ValueError("model metadata requires a non-empty class_codes list")
        self.class_codes = tuple(str(code) for code in raw_codes)
        if len(set(self.class_codes)) != len(self.class_codes):
            raise ValueError("class_codes must be unique and ordered like training")
        raw_labels = metadata.get("class_labels", {})
        if not isinstance(raw_labels, dict):
            raise ValueError("class_labels must be an object")
        self.class_labels = {
            str(code): str(label) for code, label in raw_labels.items()
        }
        raw_features = metadata.get("feature_columns", MODEL_FEATURE_COLUMNS)
        if tuple(raw_features) != MODEL_FEATURE_COLUMNS:
            raise ValueError(
                "Model metadata feature_columns does not match the current "
                "Holistic CSV schema"
            )

        try:
            import numpy as np
            import xgboost as xgb
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Install numpy and xgboost for classification") from error
        self._np = np
        self._xgb = xgb
        self._booster = xgb.Booster()
        self._booster.load_model(str(model_path))
        booster_features = self._booster.feature_names
        if booster_features is not None and tuple(booster_features) != MODEL_FEATURE_COLUMNS:
            raise ValueError(
                "XGBoost model feature_names do not match MODEL_FEATURE_COLUMNS"
            )

    def predict_probabilities(
        self, result: TrackHolisticResult
    ) -> tuple[float, ...]:
        features = model_features_from_result(result)
        vector = [
            self._np.nan if features[column] is None else float(features[column])
            for column in MODEL_FEATURE_COLUMNS
        ]
        matrix = self._np.asarray([vector], dtype=self._np.float32)
        dmatrix = self._xgb.DMatrix(
            matrix,
            feature_names=list(MODEL_FEATURE_COLUMNS),
            missing=self._np.nan,
        )
        raw = self._booster.predict(dmatrix)
        return decode_prediction(raw, len(self.class_codes))

    def describe(self, probabilities: Sequence[float]) -> Prediction:
        if len(probabilities) != len(self.class_codes):
            raise ValueError("Probability count does not match class_codes")
        class_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        class_code = self.class_codes[class_index]
        return Prediction(
            class_code=class_code,
            label=self.class_labels.get(class_code, class_code),
            confidence=float(probabilities[class_index]),
            probabilities=tuple(float(value) for value in probabilities),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights")
    parser.add_argument("--device", default=None, help="Ultralytics device: cpu, 0, ...")
    parser.add_argument("--confidence", type=float, default=DEFAULT_PERSON_CONFIDENCE)
    parser.add_argument("--target-fps", type=int, default=8)
    parser.add_argument("--max-tracks", type=int, default=2)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument(
        "--max-missed-frames", type=int, default=DEFAULT_MAX_MISSED_FRAMES
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--holistic-model", type=Path, default=None)
    parser.add_argument("--holistic-input-size", type=int, default=512)
    parser.add_argument(
        "--holistic-confidence",
        type=float,
        default=DEFAULT_HOLISTIC_CONFIDENCE,
    )
    parser.add_argument(
        "--soft-landmark-confidence",
        type=float,
        default=DEFAULT_HOLISTIC_SOFT_CONFIDENCE,
    )
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--xgboost-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--smoothing-alpha", type=float, default=0.30)
    parser.add_argument("--display-threshold", type=float, default=0.50)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tracks < 1:
        raise ValueError("--max-tracks must be at least 1")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0, 1]")
    if not 0.0 <= args.soft_landmark_confidence <= args.holistic_confidence <= 1.0:
        raise ValueError(
            "Holistic confidence must be in [0, 1] and soft confidence <= it"
        )
    if not 0.0 <= args.display_threshold <= 1.0:
        raise ValueError("--display-threshold must be in [0, 1]")


def _draw_predictions(
    cv2_module: Any,
    frame: Any,
    results: tuple[TrackHolisticResult, ...],
    predictions: dict[int, Prediction],
    *,
    display_threshold: float,
) -> None:
    for result in results:
        prediction = predictions.get(result.track_id)
        if prediction is None:
            continue
        x1, y1, _, _ = result.bbox.to_list()
        label = (
            f"T{result.track_id} {prediction.class_code} "
            f"{prediction.confidence:.0%} | {prediction.label}"
            if prediction.confidence >= display_threshold
            else f"T{result.track_id} uncertain {prediction.confidence:.0%}"
        )
        cv2_module.putText(
            frame,
            label,
            (x1, max(20, y1 - 12)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2_module.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime = XGBoostRuntime(
        args.xgboost_model.resolve(),
        args.model_metadata.resolve(),
    )
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install opencv-python to run realtime demo") from error

    session_id = TrackingManager.generate_session_id("webcam_xgboost")
    tracking = PersonTrackingModule(
        PersonTrackingConfig(
            model_path=args.model,
            session_id=session_id,
            confidence_threshold=args.confidence,
            device=args.device,
            max_tracks=args.max_tracks,
            min_iou=args.min_iou,
            max_missed_frames=args.max_missed_frames,
        )
    )
    capture = tracking.open_webcam(
        args.camera,
        width=args.width,
        height=args.height,
    )
    rate = ProcessingRateController(args.target_fps)
    smoother = PredictionSmoother(args.smoothing_alpha)
    print(f"Session ID: {session_id}")
    print("Press Q or Esc to quit")

    try:
        with HolisticLandmarkExtractor(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=args.holistic_confidence,
            min_tracking_confidence=args.holistic_confidence,
            soft_landmark_confidence=args.soft_landmark_confidence,
            crop_padding=args.crop_padding,
            task_model_path=args.holistic_model,
            task_input_size=args.holistic_input_size,
        ) as holistic:
            while True:
                frame_started_at = monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                inference_started_at = monotonic()
                packet = tracking.process_frame(frame)
                results = holistic.process_packet(frame, packet)
                visible_ids = {result.track_id for result in results}
                smoother.retain(visible_ids)
                predictions: dict[int, Prediction] = {}
                for result in results:
                    valid_ratio = model_features_from_result(result)[
                        "all_landmarks_valid_ratio"
                    ]
                    if not valid_ratio:
                        continue
                    probabilities = runtime.predict_probabilities(result)
                    probabilities = smoother.update(result.track_id, probabilities)
                    predictions[result.track_id] = runtime.describe(probabilities)
                inference_ms = (monotonic() - inference_started_at) * 1000.0

                holistic.draw_results(frame, results)
                tracking.draw_tracks(frame, packet)
                _draw_predictions(
                    cv2,
                    frame,
                    results,
                    predictions,
                    display_threshold=args.display_threshold,
                )
                cv2.putText(
                    frame,
                    (
                        f"Limit {rate.target_fps} FPS | inference "
                        f"{inference_ms:.0f} ms | Q/Esc quit"
                    ),
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, frame)
                finished_at = monotonic()
                rate.mark_processed(frame_started_at, finished_at)
                wait_ms = max(
                    1,
                    round(rate.remaining_seconds(frame_started_at) * 1000.0),
                )
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        output_path = tracking.manager.generate_final_output(session_id)
        capture.release()
        cv2.destroyAllWindows()
        print(f"Tracking JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
