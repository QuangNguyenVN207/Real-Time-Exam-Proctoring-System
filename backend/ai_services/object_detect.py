import json
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.config import settings

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ObjectDetectModule:
    def __init__(self):
        print(f"[object_detect] Loading YOLO model from {settings.yolo_model_path}...")
        print(f"[object_detect] Inference device: {_DEVICE}"
              + (f" ({torch.cuda.get_device_name(0)})" if _DEVICE == "cuda" else " (không tìm thấy CUDA)"))
        self._model = YOLO(settings.yolo_model_path)
        self._model.to(_DEVICE)
        self._flagged = set(settings.flagged_classes)

        model_class_names = set(self._model.names.values())
        missing = self._flagged - model_class_names
        if missing:
            print(
                f"[object_detect][WARNING] flagged_classes không khớp model: {missing}\n"
                f"  Model thực có các class: {sorted(model_class_names)}\n"
                "  Các class trên sẽ KHÔNG BAO GIỜ được detect — sửa lại config.py."
            )

        # đếm frame theo session, để "mỗi N frame" tính độc lập từng session
        self._frames_seen: dict[str, int] = {}
        # lưu kết quả lần inference gần nhất, để frame bị skip vẫn trả
        # về kết quả hợp lý thay vì None
        self._last_result: dict[str, dict] = {}
        # class nào đã confirm ở lần inference trước, để biết class nào
        # MỚI chuyển sang confirm ở lần này (chỉ chụp ảnh lúc đó)
        self._previously_confirmed: dict[str, set[str]] = {}
        # đếm số frame liên tiếp mỗi class được detect, theo từng session
        self._consecutive: dict[str, dict[str, int]] = {}

    def process(self, frame: np.ndarray, session_id: str, frame_id: int) -> dict | None:
        seen = self._frames_seen.get(session_id, 0) + 1
        self._frames_seen[session_id] = seen

        if seen % settings.object_detect_every_n_frames != 0:
            return self._last_result.get(session_id)

        result = self._process_sync(frame, session_id, frame_id)
        self._last_result[session_id] = result
        return result

    def _process_sync(self, frame: np.ndarray, session_id: str, frame_id: int) -> dict:
        predictions = self._model(
            frame,
            imgsz=640,
            device=_DEVICE,
            conf=settings.yolo_confidence_threshold,
            verbose=False,
        )
        detected_this_frame, boxes_this_frame = self._extract_flagged_classes(predictions)
        return self._evaluate(detected_this_frame, boxes_this_frame, frame, session_id, frame_id)

    def _extract_flagged_classes(self, predictions) -> tuple[dict[str, float], dict[str, list[int]]]:
        detected: dict[str, float] = {}
        boxes: dict[str, list[int]] = {}
        if not predictions:
            return detected, boxes

        result = predictions[0]
        if result.boxes is None:
            return detected, boxes

        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = names.get(class_id, str(class_id))
            if class_name not in self._flagged:
                continue
            confidence = float(box.conf[0])
            if confidence > detected.get(class_name, 0.0):
                detected[class_name] = confidence
                boxes[class_name] = [int(v) for v in box.xyxy[0].tolist()]

        return detected, boxes

    def _evaluate(
        self,
        detected_this_frame: dict[str, float],
        boxes_this_frame: dict[str, list[int]],
        raw_frame: np.ndarray,
        session_id: str,
        frame_id: int,
    ) -> dict:
        counters = self._consecutive.setdefault(session_id, {})
        for class_name in self._flagged:
            if class_name in detected_this_frame:
                counters[class_name] = counters.get(class_name, 0) + 1
            else:
                counters[class_name] = 0

        confirmed = [c for c, n in counters.items() if n >= settings.object_confirm_frames]

        prev_confirmed = self._previously_confirmed.get(session_id, set())
        newly_confirmed = [c for c in confirmed if c not in prev_confirmed]
        self._previously_confirmed[session_id] = set(confirmed)

        if newly_confirmed:
            self._capture_evidence(
                newly_confirmed, detected_this_frame, boxes_this_frame,
                raw_frame, session_id, frame_id,
            )

        if not confirmed:
            return {
                "label": "clear",
                "risk_score": 0.0,
                "confirmed_classes": [],
                "raw_detections": _with_display_names(detected_this_frame),
                "raw_boxes": boxes_this_frame,
                "frame_id": frame_id,
            }

        best_class = max(confirmed, key=lambda c: detected_this_frame.get(c, 0.0))
        best_confidence = detected_this_frame.get(best_class, settings.yolo_confidence_threshold)
        risk = min(0.6 + 0.15 * (len(confirmed) - 1), 1.0)

        return {
            "label": f"{_normalize_label(_display_name(best_class))}_detected",
            "risk_score": risk,
            "confidence": best_confidence,
            "confirmed_classes": [_display_name(c) for c in confirmed],
            "raw_detections": _with_display_names(detected_this_frame),
            "raw_boxes": boxes_this_frame,
            "frame_id": frame_id,
        }

    def _capture_evidence(
        self,
        newly_confirmed: list[str],
        detected_this_frame: dict[str, float],
        boxes_this_frame: dict[str, list[int]],
        raw_frame: np.ndarray,
        session_id: str,
        frame_id: int,
    ) -> None:
        session_dir = settings.session_log_dir / session_id
        snapshot_dir = session_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.time()
        display_tag = "_".join(_normalize_label(_display_name(c)) for c in newly_confirmed)
        snapshot_filename = f"object_detect_{display_tag}_{frame_id}_{int(timestamp)}.jpg"
        snapshot_path = snapshot_dir / snapshot_filename

        annotated = raw_frame.copy()
        detections_payload = []
        for class_name in newly_confirmed:
            confidence = detected_this_frame.get(class_name, 0.0)
            bbox = boxes_this_frame.get(class_name)
            label = _display_name(class_name)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated, f"{label} {confidence:.2f}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            else:
                bbox_xywh = None

            detections_payload.append({
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": bbox_xywh,
            })

        cv2.imwrite(str(snapshot_path), annotated)

        log_entry = {
            "module": "object_detect",
            "status": "alert",
            "detections": detections_payload,
            "timestamp": timestamp,
            "session_id": session_id,
            "frame_id": frame_id,
            "snapshot_file": snapshot_filename,
        }

        log_path = session_dir / "object_detect_log.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[object_detect][ERROR] Không ghi được log: {e}")

    def cleanup_session(self, session_id: str):
        self._frames_seen.pop(session_id, None)
        self._last_result.pop(session_id, None)
        self._previously_confirmed.pop(session_id, None)
        self._consecutive.pop(session_id, None)


def _display_name(raw_class_name: str) -> str:
    return settings.object_class_display_names.get(raw_class_name, raw_class_name)


def _with_display_names(detected: dict[str, float]) -> dict[str, float]:
    return {_display_name(k): v for k, v in detected.items()}


def _normalize_label(class_name: str) -> str:
    return class_name.replace(" ", "_")