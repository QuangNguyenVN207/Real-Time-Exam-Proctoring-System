"""YOLO object detection with paper detections routed to identity tracking."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.config import settings


class ObjectDetector:
    """Small server-facing YOLO adapter for prohibited exam objects.

    The larger :class:`ObjectDetectModule` below is retained for the existing
    identity-aware video and paper-tracking pipeline.  This adapter implements
    the simpler ``process_frame(frame, timestamp)`` contract used by the
    multi-module server.
    """

    BANNED_ITEMS = [
        # "cheat_sheet",
        "earphone",
        "smartwatch",
        "smartphone",
    ]
    INPUT_SIZE = 640

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.5,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Load the YOLO checkpoint once and allocate it to CUDA or CPU."""

        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device or self._automatic_device()
        self._suppress_ultralytics_logging()

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:  # pragma: no cover - dependency setup
                raise RuntimeError(
                    "Install ultralytics before creating ObjectDetector"
                ) from error
            model = YOLO(self.model_path)

        self.model = model
        if hasattr(self.model, "to"):
            try:
                self.model.to(self.device)
            except Exception:
                pass  # Bỏ qua nếu mô hình là OpenVINO/ONNX không hỗ trợ .to()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> dict[str, Any] | None:
        """Return prohibited objects in original-frame coordinates without forcing 640x640 resize."""
        # print("\n[DEBUG YOLO] ---> BẮT ĐẦU VÀO HÀM process_frame")
        try:
            if (
                not isinstance(frame, np.ndarray)
                or frame.ndim not in (2, 3)
                or frame.size == 0
            ):
                # print(f"[DEBUG YOLO] Lỗi: Frame đầu vào không hợp lệ! Type: {type(frame)}")
                return None

            original_height, original_width = frame.shape[:2]
            # print(f"[DEBUG YOLO] Kích thước frame: {original_width}x{original_height}")

            # GIỮ NGUYÊN KÍCH THƯỚC GỐC: Không resize về 640x640 nữa để giữ chi tiết vật thể nhỏ (smartphone)
            # print(f"[DEBUG YOLO] Đưa vào model với device={self.device}, conf={self.confidence_threshold}...")
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                # device=self.device,
                verbose=False,
            )
            # print(f"[DEBUG YOLO] Kết quả thô từ model.predict: {results}")
            # Vì không resize nên tỉ lệ scale = 1.0
            # print("[DEBUG YOLO] Đang chạy hàm trích xuất _extract_server_detections_no_resize...")
            detections = self._extract_server_detections_no_resize(
                results,
                original_frame=frame,
                original_width=original_width,
                original_height=original_height,
            )
            if not detections:
                # print("[DEBUG YOLO] Cảnh báo: Không có vật thể nào (hoặc đã bị lọc hết ở hàm _extract). Trả về None.")
                return None
            # print(f"[DEBUG YOLO] THÀNH CÔNG: Trích xuất được {len(detections)} vật thể!")

            details = {
                "detections": detections,
                "original_frame_size": [
                    original_width,
                    original_height,
                ],
            }
            return {
                "module": "object_detect",
                "status": "alert",
                "timestamp": float(timestamp),
                "details": details,
                "detections": detections,
            }
        except Exception as e:
            # SỬA LỖI LOGIC: Không được giấu lỗi. Phải in ra toàn bộ stack trace để bắt tận gốc
            import traceback
            print(f"\n[DEBUG YOLO] ❌ CÓ LỖI CRASH (EXCEPTION) BÊN TRONG HÀM process_frame: {e}")
            print(traceback.format_exc())
            return None

    def _extract_server_detections_no_resize(
        self,
        results: Any,
        *,
        original_frame: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = self._names_dict(
            getattr(result, "names", getattr(self.model, "names", {}))
        )
        
        detections: list[dict[str, Any]] = []

        for box in boxes:
            class_id = int(self._first_scalar(box.cls))
            raw_name = names.get(class_id, class_id)

            # PHÒNG HỜI OPENVINO MẤT FILE .NAMES (Trả về số nguyên)
            if isinstance(raw_name, (int, float)) or str(raw_name).isdigit():
                # Nếu model của bạn đặt smartphone ở index 0 (hoặc chỉnh lại theo index thực tế của bạn)
                id_mapping = {0: "smartphone", 1: "cheat_sheet", 2: "earphone", 3: "smartwatch"}
                raw_name = id_mapping.get(int(class_id), str(class_id))

            label = self._canonical_class_name(raw_name)

            # label = self._canonical_class_name(names.get(class_id, class_id))
            confidence = float(self._first_scalar(box.conf))
            # 🛑 CHẶN TUYỆT ĐỐI KHÔNG CHO PHÉP MODEL CŨ BẮT GIẤY / CHEATSHEET
            if label in ["cheat_sheet", "paper", "document", "sheet", "test_paper"]:
                continue

            if (
                label not in self.BANNED_ITEMS
                or confidence <= self.confidence_threshold
            ):
                continue

            # Lấy trực tiếp tọa độ thực tế trên khung hình gốc (vì không resize)
            coordinates = self._first_row(box.xyxy)
            model_x1, model_y1, model_x2, model_y2 = (
                float(value) for value in coordinates
            )
            bbox = [
                max(0, min(original_width, round(model_x1))),
                max(0, min(original_height, round(model_y1))),
                max(0, min(original_width, round(model_x2))),
                max(0, min(original_height, round(model_y2))),
            ]
            
            # Bộ lọc phụ (nếu cần thiết có thể bật/tắt trong settings)
            if (
                label == "smartphone"
                and (
                    (
                        settings.smartphone_calculator_grid_filter_enabled
                        and _looks_like_calculator(original_frame, bbox)
                    )
                    or (
                        settings.smartphone_pen_shape_filter_enabled
                        and _looks_like_pen(original_frame, bbox)
                    )
                )
            ):
                continue
                
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )
        return detections
    
    def _extract_server_detections(
        self,
        results: Any,
        *,
        original_frame: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = self._names_dict(
            getattr(result, "names", getattr(self.model, "names", {}))
        )
        scale_x = original_width / self.INPUT_SIZE
        scale_y = original_height / self.INPUT_SIZE
        detections: list[dict[str, Any]] = []

        for box in boxes:
            class_id = int(self._first_scalar(box.cls))
            label = self._canonical_class_name(names.get(class_id, class_id))
            confidence = float(self._first_scalar(box.conf))
            # 🛑 CHẶN TUYỆT ĐỐI KHÔNG CHO PHÉP MODEL CŨ BẮT GIẤY / CHEATSHEET
            if label in ["cheat_sheet", "paper", "document", "sheet", "test_paper"]:
                continue
            
            if (
                label not in self.BANNED_ITEMS
                or confidence <= self.confidence_threshold
            ):
                continue

            coordinates = self._first_row(box.xyxy)
            model_x1, model_y1, model_x2, model_y2 = (
                float(value) for value in coordinates
            )
            bbox = [
                max(0, min(original_width, round(model_x1 * scale_x))),
                max(0, min(original_height, round(model_y1 * scale_y))),
                max(0, min(original_width, round(model_x2 * scale_x))),
                max(0, min(original_height, round(model_y2 * scale_y))),
            ]
            if (
                label == "smartphone"
                and (
                    (
                        settings.smartphone_calculator_grid_filter_enabled
                        and _looks_like_calculator(original_frame, bbox)
                    )
                    or (
                        settings.smartphone_pen_shape_filter_enabled
                        and _looks_like_pen(original_frame, bbox)
                    )
                )
            ):
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )
        return detections

    @staticmethod
    def _automatic_device() -> str:
        try:
            import torch
        except ImportError:  # pragma: no cover - ultralytics installs torch
            return "gpu"
        return "cuda" if torch.cuda.is_available() else "gpu"

    @staticmethod
    def _suppress_ultralytics_logging() -> None:
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        try:
            from ultralytics.utils import LOGGER

            LOGGER.setLevel(logging.ERROR)
        except (ImportError, AttributeError):
            pass

    @staticmethod
    def _names_dict(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {
                index: str(value)
                for index, value in enumerate(names)
            }
        return {}

    @staticmethod
    def _canonical_class_name(class_name: Any) -> str:
        normalized = (
            str(class_name)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        return settings.object_class_aliases.get(normalized, normalized)

    @staticmethod
    def _first_scalar(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
        return array.reshape(-1)[0]

    @staticmethod
    def _first_row(value: Any) -> list[float]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=float).reshape(-1, 4)
        return array[0].tolist()


class ObjectDetectModule:
    """Detect prohibited objects while exposing every paper box to the tracker.

    ``cheat_sheet`` is deliberately not confirmed here. A paper classifier can
    flicker between ``cheat_sheet`` and ``test_paper`` for the same physical
    sheet, so paper decisions belong to the pose/gaze paper-tracking pipeline.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        smartphone_model: Any | None = None,
        device: str | None = None,
        enable_smartphone_fallback: bool | None = None,
        detect_every_n_frames: int | None = None,
        phone_confidence_floor: float | None = None,
        enable_person_roi: bool | None = None,
        enable_custom_paper_roi: bool | None = None,
        confirm_frames_by_class: dict[str, int] | None = None,
    ):
        self._detect_every_n_frames = (
            settings.object_detect_every_n_frames
            if detect_every_n_frames is None
            else int(detect_every_n_frames)
        )
        if self._detect_every_n_frames < 1:
            raise ValueError("detect_every_n_frames must be at least 1")
        self._phone_confidence_floor = (
            None
            if phone_confidence_floor is None
            else float(phone_confidence_floor)
        )
        self._person_roi_enabled = (
            settings.person_roi_object_enabled
            if enable_person_roi is None
            else bool(enable_person_roi)
        )
        self._custom_paper_roi_enabled = (
            settings.person_roi_custom_paper_enabled
            if enable_custom_paper_roi is None
            else bool(enable_custom_paper_roi)
        )
        self._confirm_frames_by_class = {
            str(class_name): int(required_frames)
            for class_name, required_frames in (
                confirm_frames_by_class or {}
            ).items()
        }
        if any(
            required_frames < 1
            for required_frames in self._confirm_frames_by_class.values()
        ):
            raise ValueError("confirm frame counts must be at least 1")
        load_default_models = model is None
        if model is None:
            try:
                import torch
                from ultralytics import YOLO
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Install torch and ultralytics before creating ObjectDetectModule"
                ) from error

            # --- BẮT ĐẦU ĐOẠN SỬA ---
            detected_device = "cpu"
            if torch.cuda.is_available():
                detected_device = "cuda"
            else:
                try:
                    import openvino as ov
                    # Nếu tìm thấy chữ GPU trong danh sách thiết bị OpenVINO, đó là Intel iGPU
                    if "GPU" in ov.Core().available_devices:
                        detected_device = "GPU"
                except ImportError:
                    pass
            
            self._device = device or detected_device

            # Tự động trỏ sang thư mục OpenVINO nếu đang dùng Intel iGPU
            model_path = str(settings.yolo_model_path)
            if self._device == "GPU" and model_path.endswith("best (1).pt"):
                model_path = model_path.replace("best (1).pt", "best_openvino_model")
            elif self._device == "GPU" and model_path.endswith(".pt"):
                model_path = model_path.replace(".pt", "_openvino_model")
            
            print(f"[object_detect] Loading YOLO model from {model_path}...")
            self._model = YOLO(model_path, task="detect")
            # --- KẾT THÚC ĐOẠN SỬA ---

            # self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            # print(
            #     f"[object_detect] Loading YOLO model from "
            #     f"{settings.yolo_model_path}..."
            # )
            # self._model = YOLO(settings.yolo_model_path)
            try:
                self._model.to(self._device)
            except Exception:
                pass  # Bỏ qua lỗi .to() đối với OpenVINO
            if self._device == "cuda":
                print(
                    f"[object_detect] Inference device: cuda "
                    f"({torch.cuda.get_device_name(0)})"
                )
            elif self._device == "GPU":
                print("[object_detect] Inference device: Intel iGPU (OpenVINO)")
            else:
                print("[object_detect] Inference device: cpu")
        else:
            self._model = model
            self._device = device or "cpu"
            if hasattr(self._model, "to"):
                self._model.to(self._device)

        fallback_enabled = (
            settings.smartphone_fallback_enabled
            if enable_smartphone_fallback is None
            else bool(enable_smartphone_fallback)
        )
        self._smartphone_model = smartphone_model
        if (
            self._smartphone_model is None
            and fallback_enabled
            and load_default_models
        ):  
            # --- BẮT ĐẦU ĐOẠN SỬA ---
            fallback_path = str(settings.smartphone_fallback_model_path)
            if self._device == "GPU" and fallback_path.endswith(".pt"):
                fallback_path = fallback_path.replace(".pt", "_openvino_model")

            print(
                "[object_detect] Loading small-object smartphone fallback "
                f"from {fallback_path}..."
            )
            self._smartphone_model = YOLO(fallback_path, task="detect")
            # --- KẾT THÚC ĐOẠN SỬA ---
            try:
                self._smartphone_model.to(self._device)
            except Exception:
                pass

            # print(
            #     "[object_detect] Loading small-object smartphone fallback "
            #     f"from {settings.smartphone_fallback_model_path}..."
            # )
            # self._smartphone_model = YOLO(
            #     settings.smartphone_fallback_model_path
            # )
            # self._smartphone_model.to(self._device)
        elif self._smartphone_model is not None and hasattr(
            self._smartphone_model,
            "to",
        ):
            try:
                self._smartphone_model.to(self._device)
            except Exception:
                pass
        self._smartphone_fallback_enabled = (
            fallback_enabled and self._smartphone_model is not None
        )
        fallback_names = self._names_dict(
            getattr(self._smartphone_model, "names", {})
        )
        self._fallback_model_names = fallback_names
        self._smartphone_class_ids = [
            class_id
            for class_id, class_name in fallback_names.items()
            if self._canonical_class_name(class_name) == "smartphone"
        ]
        self._book_class_ids = [
            class_id
            for class_id, class_name in fallback_names.items()
            if (
                settings.book_as_cheatsheet_enabled
                and str(class_name).strip().lower().replace(
                    "-", "_"
                ).replace(" ", "_") == "book"
            )
        ]
        confuser_names = {
            str(name).strip().lower().replace("-", "_").replace(" ", "_")
            for name in settings.smartphone_confuser_class_names
        }
        self._smartphone_confuser_class_ids = [
            class_id
            for class_id, class_name in fallback_names.items()
            if (
                str(class_name)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
                in confuser_names
            )
        ]
        self._auxiliary_class_ids = (
            self._smartphone_class_ids
            + self._book_class_ids
            + self._smartphone_confuser_class_ids
        )
        if self._smartphone_fallback_enabled:
            print(
                "[object_detect] Tiled COCO fallback active: "
                "cell phone -> smartphone, book -> cheat_sheet"
            )

        self._paper_classes = set(settings.paper_class_names)
        # Paper classes never use the old class-level alert counter.
        self._flagged = set(settings.flagged_classes) - self._paper_classes

        raw_model_names = self._names_dict(getattr(self._model, "names", {}))
        self._model_names = {
            class_id: self._canonical_class_name(class_name)
            for class_id, class_name in raw_model_names.items()
        }
        self._custom_paper_class_ids = [
            class_id
            for class_id, class_name in self._model_names.items()
            if class_name in self._paper_classes
        ]
        model_class_names = set(self._model_names.values())
        self.supports_test_paper = "test_paper" in model_class_names

        missing = self._flagged - model_class_names
        if missing:
            print(
                f"[object_detect][WARNING] flagged_classes do not match model: "
                f"{sorted(missing)}. Model classes: {sorted(model_class_names)}"
            )
        if not self.supports_test_paper:
            print(
                "[object_detect][INFO] This checkpoint has no test_paper class. "
                "Paper labels will be exposed as paper_unknown and decisions "
                "will be made by the configured downstream paper policy."
            )

        self._frames_seen: dict[str, int] = {}
        self._last_result: dict[str, dict[str, Any]] = {}
        self._previously_confirmed: dict[str, set[str]] = {}
        self._recent_presence: dict[
            str,
            dict[str, deque[list[list[int]]]],
        ] = {}

    def process(
        self,
        frame: np.ndarray,
        session_id: str,
        frame_id: int,
        *,
        person_rois: list[dict[str, Any]] | None = None,
        pose_suspicious_activity: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        seen = self._frames_seen.get(session_id, 0) + 1
        self._frames_seen[session_id] = seen

        if seen % self._detect_every_n_frames != 0:
            cached = self._last_result.get(session_id)
            if cached is None:
                return None
            skipped_result = dict(cached)
            skipped_result["inference_ran"] = False
            skipped_result["requested_frame_id"] = frame_id
            skipped_result["pose_gate"] = pose_suspicious_activity
            return skipped_result

        result = self._process_sync(
            frame,
            session_id,
            frame_id,
            person_rois=person_rois,
            pose_suspicious_activity=pose_suspicious_activity,
            **kwargs,
        )
        result["inference_ran"] = True
        result["requested_frame_id"] = frame_id
        result["pose_gate"] = pose_suspicious_activity
        self._last_result[session_id] = result
        return result

    def _process_sync(
        self,
        frame: np.ndarray,
        session_id: str,
        frame_id: int,
        *,
        person_rois: list[dict[str, Any]] | None = None,
        pose_suspicious_activity: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        predictions = self._model(
            frame,
            imgsz=settings.object_inference_size,
            # device=self._device,
            conf=min(
                settings.yolo_confidence_threshold,
                settings.paper_detection_confidence_threshold,
            ),
            # Ultralytics keeps predictor arguments between calls. The custom
            # paper ROI pass below sets ``classes`` to paper-only, so reset it
            # explicitly before the next full-frame inference.
            classes=None,
            verbose=False,
        )
        raw_objects = self._extract_detections(predictions)
        raw_objects.extend(
            self._detect_custom_paper_rois(
                frame,
                person_rois=person_rois or [],
            )
        )
        fallback_objects = self._detect_auxiliary_fallback(
            frame,
            person_rois=person_rois,
        )
        if self._smartphone_fallback_enabled:
            # The custom checkpoint currently produces oversized smartphone
            # boxes on classroom background. Use the tiled COCO detector as the
            # authoritative smartphone source while keeping every other custom
            # flagged class unchanged. COCO books are additive paper candidates;
            # they do not replace cheat-sheet boxes from the custom checkpoint.
            raw_objects = [
                item
                for item in raw_objects
                if item["class_name"] != "smartphone"
            ]
            raw_objects.extend(fallback_objects)
            raw_objects = self._nms(
                raw_objects,
                iou_threshold=settings.smartphone_fallback_nms_iou,
            )
        detected, boxes = self._best_direct_alert_detections(raw_objects)
        result = self._evaluate(
            detected,
            boxes,
            frame,
            session_id,
            frame_id,
            candidate_boxes_this_frame=(
                self._direct_alert_candidate_boxes(raw_objects)
            ),
        )
        result["raw_objects"] = raw_objects
        result["paper_detections"] = [
            {
                "bbox_xyxy": item["bbox_xyxy"],
                "confidence": item["confidence"],
                "class_name": item["class_name"],
            }
            for item in raw_objects
            if item["is_paper_candidate"]
        ]
        result["model_capabilities"] = {
            "supports_test_paper": self.supports_test_paper,
            "paper_decision_mode": (
                "identity_plus_semantics"
                if self.supports_test_paper
                else "identity_and_count_only"
            ),
        }
        result["pose_gate"] = pose_suspicious_activity
        return result

    def _extract_detections(self, predictions: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if not predictions:
            return detections

        result = predictions[0]
        if result.boxes is None:
            return detections
        names = self._names_dict(getattr(result, "names", self._model_names))

        for box in result.boxes:
            class_id = int(box.cls[0])
            raw_class_name = names.get(
                class_id,
                self._model_names.get(class_id, str(class_id)),
            )
            class_name = self._canonical_class_name(raw_class_name)
            is_paper = class_name in self._paper_classes
            if class_name not in self._flagged and not is_paper:
                continue

            confidence = float(box.conf[0])
            threshold = (
                settings.paper_detection_confidence_threshold
                if is_paper
                else settings.object_class_confidence_thresholds.get(
                    class_name,
                    settings.yolo_confidence_threshold,
                )
            )
            if confidence < threshold:
                continue

            detections.append(
                {
                    "class_name": class_name,
                    "display_name": _display_name(class_name),
                    "confidence": confidence,
                    "bbox_xyxy": [
                        int(value) for value in box.xyxy[0].tolist()
                    ],
                    "is_paper_candidate": is_paper,
                    "source": "custom_flagged_model",
                }
            )
        return detections

    def _detect_custom_paper_rois(
        self,
        frame: np.ndarray,
        *,
        person_rois: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Recover small/folded cheat sheets with the domain model on ROIs."""

        if (
            not self._custom_paper_roi_enabled
            or not self._custom_paper_class_ids
            or not person_rois
            or not callable(self._model)
        ):
            return []

        frame_height, frame_width = frame.shape[:2]
        roi_specs, owner_hints = self._person_roi_specs(
            frame_width=frame_width,
            frame_height=frame_height,
            person_rois=person_rois,
            include_book_context=False,
        )
        if not roi_specs:
            return []

        # predictions = self._model(
        #     [
        #         frame[y1:y2, x1:x2]
        #         for x1, y1, x2, y2 in roi_specs
        #     ],
        #     imgsz=settings.person_roi_custom_paper_inference_size,
        #     # device=self._device,
        #     conf=settings.person_roi_custom_paper_confidence_threshold,
        #     classes=self._custom_paper_class_ids,
        #     verbose=False,
        # )
        predictions = []
        for x1, y1, x2, y2 in roi_specs:
            pred = self._model(
                frame[y1:y2, x1:x2],
                imgsz=settings.person_roi_custom_paper_inference_size,
                conf=settings.person_roi_custom_paper_confidence_threshold,
                classes=self._custom_paper_class_ids,
                verbose=False,
            )
            predictions.append(pred[0]) # Gộp kết quả lại để tương thích với phần code bên dưới
        detections: list[dict[str, Any]] = []
        for (offset_x, offset_y, _, _), result, owner_hint in zip(
            roi_specs,
            predictions,
            owner_hints,
            strict=True,
        ):
            if result.boxes is None:
                continue
            names = self._names_dict(
                getattr(result, "names", self._model_names)
            )
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self._canonical_class_name(
                    names.get(
                        class_id,
                        self._model_names.get(class_id, str(class_id)),
                    )
                )
                confidence = float(box.conf[0])
                if (
                    class_name not in self._paper_classes
                    or confidence
                    < settings.person_roi_custom_paper_confidence_threshold
                ):
                    continue
                x1, y1, x2, y2 = (
                    float(value) for value in box.xyxy[0].tolist()
                )
                detections.append(
                    {
                        "class_name": class_name,
                        "display_name": _display_name(class_name),
                        "confidence": confidence,
                        "bbox_xyxy": [
                            round(x1 + offset_x),
                            round(y1 + offset_y),
                            round(x2 + offset_x),
                            round(y2 + offset_y),
                        ],
                        "is_paper_candidate": True,
                        "source": "custom_person_roi_paper_model",
                        "owner_track_id_hint": owner_hint.get("track_id"),
                        "owner_person_id_hint": owner_hint.get("person_id"),
                    }
                )
        return detections

    def _detect_auxiliary_fallback(
        self,
        frame: np.ndarray,
        *,
        person_rois: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect small phones/books globally and inside tracked-person ROIs."""

        if (
            not self._smartphone_fallback_enabled
            or not self._auxiliary_class_ids
        ):
            return []
        height, width = frame.shape[:2]
        if max(width, height) <= 960:
            tile_specs = [(0, 0, width, height)]
        else:
            tile_specs = [
                (0, 0, round(0.58 * width), round(0.68 * height)),
                (
                    round(0.42 * width),
                    0,
                    width,
                    round(0.68 * height),
                ),
                (
                    0,
                    round(0.32 * height),
                    round(0.58 * width),
                    height,
                ),
                (
                    round(0.42 * width),
                    round(0.32 * height),
                    width,
                    height,
                ),
            ]
        crops = [
            frame[y1:y2, x1:x2]
            for x1, y1, x2, y2 in tile_specs
        ]
        predictions = self._smartphone_model(
            crops,
            imgsz=settings.object_inference_size,
            # device=self._device,
            conf=min(
                settings.smartphone_fallback_confidence_threshold,
                settings.book_fallback_confidence_threshold,
            ),
            classes=self._auxiliary_class_ids,
            verbose=False,
        )
        detections = self._fallback_detections_from_predictions(
            predictions,
            tile_specs,
            frame_width=width,
            frame_height=height,
        )

        roi_specs, roi_owner_hints = self._person_roi_specs(
            frame_width=width,
            frame_height=height,
            person_rois=person_rois or [],
            include_book_context=bool(
                self._book_class_ids and max(width, height) > 960
            ),
        )
        if (
            self._person_roi_enabled
            and roi_specs
            and self._auxiliary_class_ids
        ):
            roi_predictions = self._smartphone_model(
                [
                    frame[y1:y2, x1:x2]
                    for x1, y1, x2, y2 in roi_specs
                ],
                imgsz=settings.person_roi_object_inference_size,
                # device=self._device,
                conf=min(
                    settings.person_roi_phone_confidence_threshold,
                    settings.person_roi_book_confidence_threshold,
                ),
                classes=self._auxiliary_class_ids,
                verbose=False,
            )
            detections.extend(
                self._fallback_detections_from_predictions(
                    roi_predictions,
                    roi_specs,
                    frame_width=width,
                    frame_height=height,
                    phone_threshold=(
                        settings.person_roi_phone_confidence_threshold
                    ),
                    book_threshold=(
                        settings.person_roi_book_confidence_threshold
                    ),
                    source_scope="person_roi",
                    owner_hints=roi_owner_hints,
                    require_phone_shape=True,
                )
            )

        # Wide classroom tiles preserve context for a small phone, but an
        # upright notebook can become only a few pixels wide after resizing.
        # When no tracked people are available, retain three fixed columns as a
        # fallback. Normally the person ROIs above replace these columns, using
        # fewer crops while focusing more pixels on hands, lap and desk.
        if (
            self._book_class_ids
            and max(width, height) > 960
            and not roi_specs
        ):
            detail_specs = [
                (
                    round(0.05 * width),
                    round(0.20 * height),
                    round(0.45 * width),
                    round(0.95 * height),
                ),
                (
                    round(0.30 * width),
                    round(0.20 * height),
                    round(0.70 * width),
                    round(0.95 * height),
                ),
                (
                    round(0.55 * width),
                    round(0.20 * height),
                    round(0.95 * width),
                    round(0.95 * height),
                ),
            ]
            detail_predictions = self._smartphone_model(
                [
                    frame[y1:y2, x1:x2]
                    for x1, y1, x2, y2 in detail_specs
                ],
                imgsz=settings.book_fallback_detail_inference_size,
                # device=self._device,
                conf=settings.book_fallback_confidence_threshold,
                classes=self._book_class_ids,
                verbose=False,
            )
            detections.extend(
                self._fallback_detections_from_predictions(
                    detail_predictions,
                    detail_specs,
                    frame_width=width,
                    frame_height=height,
                )
            )
        detections = self._remove_smartphone_confusers(
            frame,
            detections,
        )
        return self._nms(
            detections,
            iou_threshold=settings.smartphone_fallback_nms_iou,
        )

    def _fallback_detections_from_predictions(
        self,
        predictions: Any,
        tile_specs: list[tuple[int, int, int, int]],
        *,
        frame_width: int,
        frame_height: int,
        phone_threshold: float | None = None,
        book_threshold: float | None = None,
        source_scope: str = "tiled",
        owner_hints: list[dict[str, Any] | None] | None = None,
        require_phone_shape: bool = False,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        resolved_owner_hints = owner_hints or [None] * len(tile_specs)
        for (offset_x, offset_y, _, _), result, owner_hint in zip(
            tile_specs,
            predictions,
            resolved_owner_hints,
            strict=True,
        ):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                raw_class_name = self._fallback_model_names.get(
                    class_id,
                    str(class_id),
                )
                class_name = self._canonical_class_name(raw_class_name)
                confidence = float(box.conf[0])
                if class_id in self._smartphone_confuser_class_ids:
                    threshold = (
                        settings.smartphone_confuser_confidence_threshold
                    )
                    detection_kind = "phone_confuser"
                elif class_name == "cheat_sheet":
                    threshold = (
                        book_threshold
                        if book_threshold is not None
                        else settings.book_fallback_confidence_threshold
                    )
                    detection_kind = "book"
                elif class_name == "smartphone":
                    threshold = (
                        phone_threshold
                        if phone_threshold is not None
                        else settings.smartphone_fallback_confidence_threshold
                    )
                    if self._phone_confidence_floor is not None:
                        threshold = max(
                            threshold,
                            self._phone_confidence_floor,
                        )
                    detection_kind = "phone"
                else:
                    continue
                if confidence < threshold:
                    continue
                x1, y1, x2, y2 = (
                    float(value) for value in box.xyxy[0].tolist()
                )
                bbox_xyxy = [
                    round(x1 + offset_x),
                    round(y1 + offset_y),
                    round(x2 + offset_x),
                    round(y2 + offset_y),
                ]
                interaction_bbox = (
                    owner_hint.get("interaction_bbox_xyxy")
                    if owner_hint is not None
                    else None
                )
                if (
                    interaction_bbox is not None
                    and not _bbox_center_inside(
                        bbox_xyxy,
                        interaction_bbox,
                    )
                ):
                    continue
                if detection_kind == "phone_confuser":
                    detections.append(
                        {
                            "class_name": class_name,
                            "display_name": class_name,
                            "confidence": confidence,
                            "bbox_xyxy": bbox_xyxy,
                            "is_paper_candidate": False,
                            "is_phone_confuser": True,
                            "source": f"coco_{source_scope}_phone_confuser",
                        }
                    )
                    continue
                if (
                    class_name == "cheat_sheet"
                    and not self._is_plausible_book_bbox(
                        bbox_xyxy,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                ):
                    continue
                if (
                    class_name == "smartphone"
                    and require_phone_shape
                    and not self._is_plausible_roi_phone_bbox(
                        bbox_xyxy,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                ):
                    continue
                is_paper = class_name in self._paper_classes
                payload = {
                    "class_name": class_name,
                    "display_name": _display_name(class_name),
                    "confidence": confidence,
                    "bbox_xyxy": bbox_xyxy,
                    "is_paper_candidate": is_paper,
                    "source": (
                        f"coco_{source_scope}_book_fallback"
                        if is_paper
                        else f"coco_{source_scope}_phone_fallback"
                    ),
                }
                if owner_hint is not None:
                    payload["owner_track_id_hint"] = owner_hint.get(
                        "track_id"
                    )
                    payload["owner_person_id_hint"] = owner_hint.get(
                        "person_id"
                    )
                detections.append(payload)
        return detections

    def _remove_smartphone_confusers(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop remote/calculator/pen-like candidates before confirmation."""

        confusers = [
            item
            for item in detections
            if item.get("is_phone_confuser", False)
        ]
        accepted: list[dict[str, Any]] = []
        for item in detections:
            if item.get("is_phone_confuser", False):
                continue
            if item.get("class_name") != "smartphone":
                accepted.append(item)
                continue

            bbox = item["bbox_xyxy"]
            overlaps_pretrained_confuser = any(
                (
                    _bbox_intersection_over_smaller(
                        bbox,
                        negative["bbox_xyxy"],
                    )
                    >= settings.smartphone_confuser_overlap_threshold
                )
                for negative in confusers
            )
            has_calculator_grid = (
                settings.smartphone_calculator_grid_filter_enabled
                and _looks_like_calculator(frame, bbox)
            )
            has_pen_shape = (
                settings.smartphone_pen_shape_filter_enabled
                and _looks_like_pen(frame, bbox)
            )
            if (
                not overlaps_pretrained_confuser
                and not has_calculator_grid
                and not has_pen_shape
            ):
                accepted.append(item)
        return accepted

    @staticmethod
    def _person_roi_specs(
        *,
        frame_width: int,
        frame_height: int,
        person_rois: list[dict[str, Any]],
        include_book_context: bool = False,
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[dict[str, Any] | None],
    ]:
        specs: list[tuple[int, int, int, int]] = []
        owner_hints: list[dict[str, Any] | None] = []
        for payload in person_rois[: settings.person_roi_max_people]:
            values = payload.get("bbox_xyxy")
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in values)
            width = x2 - x1
            height = y2 - y1
            if width <= 1 or height <= 1:
                continue

            if (
                height / width
                >= settings.person_roi_full_body_aspect_ratio
            ):
                roi_bottom = (
                    y1
                    + settings.person_roi_full_body_bottom_ratio * height
                )
            else:
                roi_bottom = (
                    y2
                    + settings.person_roi_torso_bottom_expansion * height
                )
            roi = (
                max(
                    0,
                    round(
                        x1
                        - settings.person_roi_horizontal_expansion * width
                    ),
                ),
                max(
                    0,
                    round(
                        y1 - settings.person_roi_top_expansion * height
                    ),
                ),
                min(
                    frame_width,
                    round(
                        x2
                        + settings.person_roi_horizontal_expansion * width
                    ),
                ),
                min(frame_height, round(roi_bottom)),
            )
            if roi[2] - roi[0] < 32 or roi[3] - roi[1] < 32:
                continue
            specs.append(roi)
            owner_hint = {
                "track_id": payload.get("track_id"),
                "person_id": payload.get("person_id"),
                "interaction_bbox_xyxy": roi,
            }
            owner_hints.append(owner_hint)

            if include_book_context:
                context_width = min(
                    frame_width,
                    round(
                        settings.person_roi_context_width_ratio
                        * frame_width
                    ),
                )
                context_height = min(
                    frame_height,
                    round(
                        settings.person_roi_context_height_ratio
                        * frame_height
                    ),
                )
                center_x = (x1 + x2) / 2.0
                center_y = (
                    y1
                    + settings.person_roi_context_vertical_center_ratio
                    * height
                )
                context_left = min(
                    max(0, round(center_x - context_width / 2.0)),
                    frame_width - context_width,
                )
                context_top = min(
                    max(0, round(center_y - context_height / 2.0)),
                    frame_height - context_height,
                )
                context_roi = (
                    context_left,
                    context_top,
                    context_left + context_width,
                    context_top + context_height,
                )
                if context_roi != roi:
                    specs.append(context_roi)
                    owner_hints.append(owner_hint)
        return specs, owner_hints

    @staticmethod
    def _is_plausible_book_bbox(
        bbox_xyxy: list[int],
        *,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        width = max(0, bbox_xyxy[2] - bbox_xyxy[0])
        height = max(0, bbox_xyxy[3] - bbox_xyxy[1])
        short_side = min(width, height)
        long_side = max(width, height)
        minimum_short_side = max(
            10,
            round(
                settings.book_fallback_min_short_side_ratio
                * max(frame_width, frame_height)
            ),
        )
        aspect_ratio = long_side / max(short_side, 1)
        return (
            short_side >= minimum_short_side
            and aspect_ratio <= settings.book_fallback_max_aspect_ratio
        )

    @staticmethod
    def _is_plausible_roi_phone_bbox(
        bbox_xyxy: list[int],
        *,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        width = max(0, bbox_xyxy[2] - bbox_xyxy[0])
        height = max(0, bbox_xyxy[3] - bbox_xyxy[1])
        short_side = min(width, height)
        long_side = max(width, height)
        minimum_short_side = max(
            10,
            round(
                settings.person_roi_phone_min_short_side_ratio
                * max(frame_width, frame_height)
            ),
        )
        aspect_ratio = long_side / max(short_side, 1)
        return (
            short_side >= minimum_short_side
            and aspect_ratio
            >= settings.person_roi_phone_min_aspect_ratio
            and aspect_ratio
            <= settings.person_roi_phone_max_aspect_ratio
        )

    @staticmethod
    def _nms(
        detections: list[dict[str, Any]],
        *,
        iou_threshold: float,
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for candidate in sorted(
            detections,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        ):
            duplicate = next(
                (
                    existing
                    for existing in kept
                    if (
                        candidate.get("class_name")
                        == existing.get("class_name")
                        and _bbox_iou(
                            candidate["bbox_xyxy"],
                            existing["bbox_xyxy"],
                        )
                        > iou_threshold
                    )
                ),
                None,
            )
            if duplicate is None:
                kept.append(candidate)
            elif (
                duplicate.get("owner_track_id_hint") is None
                and candidate.get("owner_track_id_hint") is not None
            ):
                duplicate["owner_track_id_hint"] = candidate[
                    "owner_track_id_hint"
                ]
                duplicate["owner_person_id_hint"] = candidate.get(
                    "owner_person_id_hint"
                )
        return kept

    def _best_direct_alert_detections(
        self,
        raw_objects: list[dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, list[int]]]:
        detected: dict[str, float] = {}
        boxes: dict[str, list[int]] = {}
        for item in raw_objects:
            class_name = item["class_name"]
            if class_name not in self._flagged:
                continue
            confidence = float(item["confidence"])
            if confidence > detected.get(class_name, 0.0):
                detected[class_name] = confidence
                boxes[class_name] = list(item["bbox_xyxy"])
        return detected, boxes

    def _direct_alert_candidate_boxes(
        self,
        raw_objects: list[dict[str, Any]],
    ) -> dict[str, list[list[int]]]:
        """Keep every candidate so parallel phones get separate histories."""

        candidates: dict[str, list[list[int]]] = {}
        for item in raw_objects:
            class_name = item["class_name"]
            if class_name not in self._flagged:
                continue
            candidates.setdefault(class_name, []).append(
                list(item["bbox_xyxy"])
            )
        return candidates

    def _evaluate(
        self,
        detected_this_frame: dict[str, float],
        boxes_this_frame: dict[str, list[int]],
        raw_frame: np.ndarray,
        session_id: str,
        frame_id: int,
        *,
        candidate_boxes_this_frame: (
            dict[str, list[list[int]]] | None
        ) = None,
    ) -> dict[str, Any]:
        histories = self._recent_presence.setdefault(session_id, {})
        counters: dict[str, int] = {}
        resolved_candidates = candidate_boxes_this_frame or {
            class_name: [bbox]
            for class_name, bbox in boxes_this_frame.items()
        }
        for class_name in self._flagged:
            history = histories.setdefault(
                class_name,
                deque(maxlen=settings.object_confirm_window),
            )
            history.append(resolved_candidates.get(class_name, []))
            counters[class_name] = _max_spatially_consistent_detections(
                history
            )

        confirmed = sorted(
            class_name
            for class_name, count in counters.items()
            if count
            >= self._confirm_frames_by_class.get(
                class_name,
                settings.object_confirm_frames,
            )
        )
        previous = self._previously_confirmed.get(session_id, set())
        newly_confirmed = [
            class_name for class_name in confirmed if class_name not in previous
        ]
        self._previously_confirmed[session_id] = set(confirmed)

        if newly_confirmed:
            self._capture_evidence(
                newly_confirmed,
                detected_this_frame,
                boxes_this_frame,
                raw_frame,
                session_id,
                frame_id,
            )

        base = {
            "confirmed_classes": [_display_name(name) for name in confirmed],
            "raw_detections": _with_display_names(detected_this_frame),
            "raw_boxes": boxes_this_frame,
            "frame_id": frame_id,
        }
        if not confirmed:
            return {"label": "clear", "risk_score": 0.0, **base}

        best_class = max(
            confirmed,
            key=lambda name: detected_this_frame.get(name, 0.0),
        )
        best_confidence = detected_this_frame.get(
            best_class,
            settings.yolo_confidence_threshold,
        )
        return {
            "label": f"{_normalize_label(_display_name(best_class))}_detected",
            "risk_score": min(0.6 + 0.15 * (len(confirmed) - 1), 1.0),
            "confidence": best_confidence,
            **base,
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
        try:
            import cv2
        except ImportError:  # pragma: no cover - dependency error handled at runtime
            print("[object_detect][WARNING] OpenCV unavailable; evidence not saved")
            return

        session_dir = settings.session_log_dir / session_id
        snapshot_dir = session_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.time()
        display_tag = "_".join(
            _normalize_label(_display_name(name)) for name in newly_confirmed
        )
        snapshot_filename = (
            f"object_detect_{display_tag}_{frame_id}_{int(timestamp)}.jpg"
        )
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
                    annotated,
                    f"{label} {confidence:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            else:
                bbox_xywh = None
            detections_payload.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": bbox_xywh,
                }
            )

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
            with log_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except OSError as error:
            print(f"[object_detect][ERROR] Could not write log: {error}")

    def cleanup_session(self, session_id: str) -> None:
        self._frames_seen.pop(session_id, None)
        self._last_result.pop(session_id, None)
        self._previously_confirmed.pop(session_id, None)
        self._recent_presence.pop(session_id, None)

    @staticmethod
    def _names_dict(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    @staticmethod
    def _canonical_class_name(class_name: str) -> str:
        normalized = str(class_name).strip().lower().replace("-", "_").replace(" ", "_")
        return settings.object_class_aliases.get(normalized, normalized)


def _display_name(raw_class_name: str) -> str:
    return settings.object_class_display_names.get(raw_class_name, raw_class_name)


def _with_display_names(detected: dict[str, float]) -> dict[str, float]:
    return {_display_name(name): confidence for name, confidence in detected.items()}


def _normalize_label(class_name: str) -> str:
    return class_name.replace(" ", "_")


def _bbox_iou(first: list[int], second: list[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(
        0,
        first[3] - first[1],
    )
    second_area = max(0, second[2] - second[0]) * max(
        0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_intersection_over_smaller(
    first: list[int],
    second: list[int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(
        0,
        first[3] - first[1],
    )
    second_area = max(0, second[2] - second[0]) * max(
        0,
        second[3] - second[1],
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0


def _same_physical_object(
    first: list[int],
    second: list[int],
) -> bool:
    """Match two recent detections without requiring pixel-perfect IoU.

    Hand-held objects can move between inference frames. An overlap match is
    preferred, while the normalized centre-distance fallback tolerates that
    motion without joining detections from different desks or hands.
    """

    if (
        _bbox_intersection_over_smaller(first, second)
        >= settings.object_spatial_match_overlap_threshold
    ):
        return True

    first_center = (
        (first[0] + first[2]) / 2.0,
        (first[1] + first[3]) / 2.0,
    )
    second_center = (
        (second[0] + second[2]) / 2.0,
        (second[1] + second[3]) / 2.0,
    )
    center_distance = float(
        np.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )
    )
    reference_size = max(
        first[2] - first[0],
        first[3] - first[1],
        second[2] - second[0],
        second[3] - second[1],
        1,
    )
    return (
        center_distance
        <= settings.object_spatial_match_center_distance_ratio
        * reference_size
    )


def _max_spatially_consistent_detections(
    history: deque[list[list[int]]],
) -> int:
    """Count matching frames, never multiple boxes from the same frame."""

    frames = list(history)
    anchors = [bbox for frame_boxes in frames for bbox in frame_boxes]
    if not anchors:
        return 0
    return max(
        sum(
            any(
                _same_physical_object(anchor, candidate)
                for candidate in frame_boxes
            )
            for frame_boxes in frames
        )
        for anchor in anchors
    )


def _looks_like_pen(
    frame: np.ndarray,
    bbox_xyxy: list[int],
) -> bool:
    """Conservatively reject a bright narrow pen/marker phone false positive.

    The guard intentionally applies only to moderately elongated detection
    boxes containing a much thinner, low-saturation bright component. A real
    phone that fills its box, including one with a bright screen or case, does
    not satisfy the component-area and width constraints.
    """

    try:
        import cv2

        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(frame_width, int(bbox_xyxy[0])))
        y1 = max(0, min(frame_height, int(bbox_xyxy[1])))
        x2 = max(0, min(frame_width, int(bbox_xyxy[2])))
        y2 = max(0, min(frame_height, int(bbox_xyxy[3])))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            return False

        crop_height, crop_width = crop.shape[:2]
        if crop_height <= crop_width:
            return False
        bbox_aspect = max(crop_width, crop_height) / max(
            min(crop_width, crop_height),
            1,
        )
        # Pens in the observed false positives form a portrait-ish proposal.
        # More elongated landscape proposals are usually actual phones.
        if not 1.30 <= bbox_aspect <= 2.10:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bright_neutral = (
            (gray >= 130) & (hsv[:, :, 1] <= 110)
        ).astype(np.uint8) * 255
        bright_neutral_ratio = (
            np.count_nonzero(bright_neutral) / bright_neutral.size
        )
        # A tight YOLO proposal around a white marker can contain more bright
        # pixels than the older, looser proposals.  Keep the upper bound below
        # a bright phone that fills almost its complete box, while allowing the
        # observed tight marker crop (about 0.43 bright-neutral coverage).
        if not 0.10 <= bright_neutral_ratio <= 0.50:
            return False
        bright_neutral = cv2.morphologyEx(
            bright_neutral,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(
            bright_neutral,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        crop_area = crop_width * crop_height
        short_crop_side = min(crop_width, crop_height)
        long_crop_side = max(crop_width, crop_height)
        for contour in contours:
            area_ratio = cv2.contourArea(contour) / max(crop_area, 1)
            component_width, component_height = cv2.minAreaRect(contour)[1]
            component_short = min(component_width, component_height)
            component_long = max(component_width, component_height)
            if component_short <= 0:
                continue
            component_aspect = component_long / component_short
            short_side_ratio = component_short / short_crop_side
            long_side_ratio = component_long / long_crop_side
            if (
                0.035 <= area_ratio <= 0.48
                and component_aspect >= 2.30
                and short_side_ratio <= 0.62
                and 0.52 <= long_side_ratio <= 1.06
            ):
                return True
        return False
    except Exception:
        # This is an optional post-filter; a malformed crop must never stop the
        # long-lived camera worker.
        return False


def _looks_like_calculator(
    frame: np.ndarray,
    bbox_xyxy: list[int],
) -> bool:
    """Recognize a physical calculator's repeated button grid.

    COCO does not contain a calculator class. This conservative visual guard
    only rejects a phone candidate when at least three rows and three columns
    of similarly sized rectangular buttons are visible.
    """

    try:
        import cv2

        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(frame_width, int(bbox_xyxy[0])))
        y1 = max(0, min(frame_height, int(bbox_xyxy[1])))
        x2 = max(0, min(frame_width, int(bbox_xyxy[2])))
        y2 = max(0, min(frame_height, int(bbox_xyxy[3])))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 18:
            return False

        gray = (
            crop
            if crop.ndim == 2
            else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        )
        scale = min(6.0, max(1.0, 180.0 / min(gray.shape[:2])))
        if scale > 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        height, width = gray.shape[:2]
        image_area = width * height
        button_boxes: list[list[int]] = []
        for contour in contours:
            x, y, button_width, button_height = cv2.boundingRect(contour)
            area = button_width * button_height
            aspect_ratio = button_width / max(button_height, 1)
            if not (
                0.025 <= button_width / width <= 0.30
                and 0.02 <= button_height / height <= 0.18
                and 0.0007 <= area / image_area <= 0.05
                and 0.45 <= aspect_ratio <= 2.2
            ):
                continue
            candidate = [
                x,
                y,
                x + button_width,
                y + button_height,
            ]
            if any(
                _bbox_iou(candidate, existing) > 0.70
                for existing in button_boxes
            ):
                continue
            button_boxes.append(candidate)

        if len(button_boxes) < 8:
            return False
        centers = [
            (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
                box[2] - box[0],
                box[3] - box[1],
            )
            for box in button_boxes
        ]
        median_width = float(np.median([item[2] for item in centers]))
        median_height = float(np.median([item[3] for item in centers]))

        def grouped_counts(
            values: list[float],
            tolerance: float,
        ) -> list[int]:
            groups: list[list[float]] = []
            for value in sorted(values):
                for group in groups:
                    if abs(value - sum(group) / len(group)) <= tolerance:
                        group.append(value)
                        break
                else:
                    groups.append([value])
            return [len(group) for group in groups]

        row_counts = grouped_counts(
            [item[1] for item in centers],
            max(5.0, median_height * 0.8),
        )
        column_counts = grouped_counts(
            [item[0] for item in centers],
            max(5.0, median_width * 0.8),
        )
        rows_with_three_buttons = sum(count >= 3 for count in row_counts)
        columns_with_three_buttons = sum(
            count >= 3 for count in column_counts
        )
        return (
            rows_with_three_buttons >= 3
            and columns_with_three_buttons >= 3
        )
    except Exception:
        # A visual guard must never make the detector worker fail.
        return False


def _bbox_center_inside(
    bbox_xyxy: list[int],
    container_xyxy: list[int] | tuple[int, int, int, int],
) -> bool:
    center_x = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
    center_y = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
    return (
        container_xyxy[0] <= center_x <= container_xyxy[2]
        and container_xyxy[1] <= center_y <= container_xyxy[3]
    )
