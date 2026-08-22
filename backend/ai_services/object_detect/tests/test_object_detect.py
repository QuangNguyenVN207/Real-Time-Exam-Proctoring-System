from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from backend.ai_services.object_detect.object_detect import (
    ObjectDetectModule,
    ObjectDetector,
    _bbox_center_inside,
    _looks_like_calculator,
)
from backend.ai_services.webcam_utils import resize_live_frame


class _FakeModel:
    names = {
        0: "cheat_sheet",
        1: "smartphone",
        2: "earphone",
        3: "smartwatch",
    }

    def to(self, _device: str) -> None:
        return None


class _FakeFallbackModel:
    names = {67: "cell phone", 73: "book"}

    def to(self, _device: str) -> None:
        return None

    def __call__(self, crops, **kwargs):
        is_person_roi_call = (
            int(kwargs["imgsz"]) == 960
            and set(kwargs["classes"]) == {67, 73}
        )
        return [
            SimpleNamespace(
                boxes=(
                    [_box(67, 0.40, [50, 100, 80, 155])]
                    if is_person_roi_call and index == 0
                    else []
                )
            )
            for index, _crop in enumerate(crops)
        ]


class _FakeCustomRoiPaperModel(_FakeModel):
    def __call__(self, crops, **kwargs):
        return [
            SimpleNamespace(
                names=self.names,
                boxes=(
                    [_box(0, 0.42, [30, 120, 100, 180])]
                    if index == 0
                    else []
                ),
            )
            for index, _crop in enumerate(crops)
        ]


class _FakeCountingObjectModel(_FakeModel):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _frame, **_kwargs):
        self.calls += 1
        return [SimpleNamespace(names=self.names, boxes=[])]


def _box(class_id: int, confidence: float, xyxy: list[int]):
    return SimpleNamespace(
        cls=np.array([class_id]),
        conf=np.array([confidence]),
        xyxy=np.array([xyxy]),
    )


class _FakeServerModel:
    names = {
        0: "smartphone",
        1: "book",
        2: "pen",
        3: "earphone",
    }

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.device = None
        self.received_frame = None
        self.received_kwargs = None

    def to(self, device: str) -> None:
        self.device = device

    def __call__(self, frame, **kwargs):
        if self.fail:
            raise RuntimeError("simulated inference failure")
        self.received_frame = frame
        self.received_kwargs = kwargs
        return [
            SimpleNamespace(
                names=self.names,
                boxes=[
                    _box(0, 0.95, [64, 128, 320, 512]),
                    _box(1, 0.80, [320, 64, 600, 320]),
                    _box(2, 0.99, [10, 10, 20, 20]),
                    _box(3, 0.50, [100, 100, 200, 200]),
                ],
            )
        ]


class ObjectDetectorContractTests(unittest.TestCase):
    def test_contract_resize_filter_scale_and_suppress_logging(self) -> None:
        model = _FakeServerModel()
        detector = ObjectDetector(
            "unused.pt",
            model=model,
            device="cpu",
            confidence_threshold=0.5,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = detector.process_frame(frame, 1_700_000_000)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            set(("module", "status", "timestamp", "details")) - set(result),
            set(),
        )
        self.assertEqual(result["module"], "object_detect")
        self.assertEqual(result["status"], "alert")
        self.assertIsInstance(result["timestamp"], float)
        self.assertIsInstance(result["details"], dict)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.received_frame.shape[:2], (640, 640))
        self.assertEqual(model.received_kwargs["imgsz"], 640)
        self.assertEqual(model.received_kwargs["device"], "cpu")
        self.assertFalse(model.received_kwargs["verbose"])

        detections = result["details"]["detections"]
        self.assertEqual(
            [item["label"] for item in detections],
            ["smartphone", "cheat_sheet"],
        )
        self.assertEqual(
            detections[0]["bbox"],
            [128, 144, 640, 576],
        )
        self.assertEqual(
            detections[1]["bbox"],
            [640, 72, 1200, 360],
        )
        self.assertIs(result["detections"], detections)

    def test_banned_items_match_server_contract(self) -> None:
        self.assertEqual(
            ObjectDetector.BANNED_ITEMS,
            [
                "cheat_sheet",
                "earphone",
                "smartwatch",
                "smartphone",
            ],
        )

    def test_corrupt_frame_returns_none(self) -> None:
        detector = ObjectDetector(
            "unused.pt",
            model=_FakeServerModel(),
            device="cpu",
        )

        self.assertIsNone(detector.process_frame(None, 1.0))
        self.assertIsNone(
            detector.process_frame(np.array([], dtype=np.uint8), 1.0)
        )

    def test_model_failure_returns_none_instead_of_raising(self) -> None:
        detector = ObjectDetector(
            "unused.pt",
            model=_FakeServerModel(fail=True),
            device="cpu",
        )

        result = detector.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            1.0,
        )

        self.assertIsNone(result)

    def test_no_banned_detection_returns_none(self) -> None:
        model = _FakeServerModel()
        model.names = {2: "pen", 3: "earphone"}
        detector = ObjectDetector(
            "unused.pt",
            model=model,
            device="cpu",
        )

        result = detector.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            1.0,
        )

        self.assertIsNone(result)


class ObjectDetectExtractionTests(unittest.TestCase):
    def test_live_cadence_runs_object_model_once_every_four_frames(
        self,
    ) -> None:
        model = _FakeCountingObjectModel()
        module = ObjectDetectModule(
            model=model,
            enable_smartphone_fallback=False,
            detect_every_n_frames=4,
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        results = [
            module.process(frame, "live", frame_id)
            for frame_id in range(1, 9)
        ]

        self.assertEqual(model.calls, 2)
        self.assertIsNone(results[0])
        self.assertTrue(results[3]["inference_ran"])
        self.assertFalse(results[4]["inference_ran"])
        self.assertTrue(results[7]["inference_ran"])

    def test_calculator_button_grid_is_not_a_smartphone(self) -> None:
        calculator = np.full((500, 300, 3), 230, dtype=np.uint8)
        import cv2

        cv2.rectangle(calculator, (50, 20), (250, 480), (30, 30, 30), -1)
        cv2.rectangle(
            calculator,
            (70, 45),
            (230, 120),
            (180, 200, 190),
            -1,
        )
        for row in range(5):
            for column in range(4):
                x = 75 + column * 42
                y = 160 + row * 55
                cv2.rectangle(
                    calculator,
                    (x, y),
                    (x + 28, y + 35),
                    (210, 210, 210),
                    -1,
                )

        phone = np.full((500, 300, 3), 230, dtype=np.uint8)
        cv2.rectangle(phone, (60, 20), (240, 480), (20, 20, 20), -1)
        cv2.rectangle(
            phone,
            (72, 50),
            (228, 435),
            (160, 170, 190),
            -1,
        )

        self.assertTrue(
            _looks_like_calculator(calculator, [0, 0, 300, 500])
        )
        self.assertFalse(_looks_like_calculator(phone, [0, 0, 300, 500]))

    def test_pretrained_remote_overlap_rejects_phone_candidate(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        candidates = [
            {
                "class_name": "smartphone",
                "confidence": 0.62,
                "bbox_xyxy": [100, 100, 220, 300],
                "is_paper_candidate": False,
            },
            {
                "class_name": "remote",
                "confidence": 0.40,
                "bbox_xyxy": [110, 120, 215, 290],
                "is_paper_candidate": False,
                "is_phone_confuser": True,
            },
        ]

        accepted = module._remove_smartphone_confusers(frame, candidates)

        self.assertEqual(accepted, [])

    def test_live_frame_is_downscaled_without_aspect_distortion(self) -> None:
        full_hd = np.zeros((1080, 1920, 3), dtype=np.uint8)
        vga = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_live_frame(full_hd)
        untouched = resize_live_frame(vga)

        self.assertEqual(resized.shape[:2], (540, 960))
        self.assertIs(untouched, vga)

    def test_all_paper_boxes_are_preserved_for_tracking(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        predictions = [
            SimpleNamespace(
                names=_FakeModel.names,
                boxes=[
                    _box(0, 0.90, [10, 20, 110, 160]),
                    _box(0, 0.82, [180, 30, 280, 170]),
                ],
            )
        ]

        detections = module._extract_detections(predictions)

        self.assertEqual(len(detections), 2)
        self.assertTrue(all(item["is_paper_candidate"] for item in detections))

    def test_paper_is_not_sent_to_legacy_direct_alert_counter(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        raw_objects = [
            {
                "class_name": "cheat_sheet",
                "confidence": 0.90,
                "bbox_xyxy": [10, 20, 110, 160],
                "is_paper_candidate": True,
            },
            {
                "class_name": "smartphone",
                "confidence": 0.88,
                "bbox_xyxy": [200, 40, 280, 180],
                "is_paper_candidate": False,
            },
        ]

        detected, boxes = module._best_direct_alert_detections(raw_objects)

        self.assertEqual(detected, {"smartphone": 0.88})
        self.assertEqual(boxes, {"smartphone": [200, 40, 280, 180]})
        self.assertFalse(module.supports_test_paper)

    def test_three_detections_in_five_inferences_confirm_smartphone(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        module._capture_evidence = lambda *args, **kwargs: None
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        smartphone = {"smartphone": 0.85}
        boxes = {"smartphone": [1, 1, 20, 20]}

        results = [
            module._evaluate(
                detected,
                boxes if detected else {},
                frame,
                "rolling_window",
                frame_id,
            )
            for frame_id, detected in enumerate(
                [smartphone, {}, {}, smartphone, smartphone],
                start=1,
            )
        ]

        self.assertTrue(all(item["label"] == "clear" for item in results[:4]))
        self.assertEqual(results[4]["label"], "smartphone_detected")
        self.assertEqual(results[4]["confirmed_classes"], ["smartphone"])

    def test_fast_mode_can_confirm_phone_after_two_detections(self) -> None:
        module = ObjectDetectModule(
            model=_FakeModel(),
            confirm_frames_by_class={"smartphone": 2},
        )
        module._capture_evidence = lambda *args, **kwargs: None
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        smartphone = {"smartphone": 0.70}
        boxes = {"smartphone": [1, 1, 20, 20]}

        first = module._evaluate(
            smartphone,
            boxes,
            frame,
            "fast_confirm",
            1,
        )
        second = module._evaluate(
            smartphone,
            boxes,
            frame,
            "fast_confirm",
            2,
        )

        self.assertEqual(first["label"], "clear")
        self.assertEqual(second["label"], "smartphone_detected")

    def test_coco_cell_phone_is_normalized_and_duplicate_tiles_are_removed(
        self,
    ) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        self.assertEqual(
            module._canonical_class_name("cell phone"),
            "smartphone",
        )
        detections = [
            {
                "confidence": 0.80,
                "bbox_xyxy": [100, 100, 180, 180],
            },
            {
                "confidence": 0.70,
                "bbox_xyxy": [102, 102, 182, 182],
            },
        ]

        kept = module._nms(detections, iou_threshold=0.45)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["confidence"], 0.80)

    def test_coco_book_is_normalized_to_paper_cheat_sheet(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())

        class_name = module._canonical_class_name("book")

        self.assertEqual(class_name, "cheat_sheet")
        self.assertIn(class_name, module._paper_classes)
        self.assertNotIn(class_name, module._flagged)

    def test_nms_does_not_suppress_overlapping_phone_and_book(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        detections = [
            {
                "class_name": "smartphone",
                "confidence": 0.80,
                "bbox_xyxy": [100, 100, 180, 180],
            },
            {
                "class_name": "cheat_sheet",
                "confidence": 0.70,
                "bbox_xyxy": [102, 102, 182, 182],
            },
        ]

        kept = module._nms(detections, iou_threshold=0.45)

        self.assertEqual(len(kept), 2)

    def test_book_shape_guard_rejects_thin_desk_edge(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())

        thin_edge = module._is_plausible_book_bbox(
            [687, 436, 803, 459],
            frame_width=1920,
            frame_height=1080,
        )
        upright_notebook = module._is_plausible_book_bbox(
            [624, 596, 704, 671],
            frame_width=1920,
            frame_height=1080,
        )

        self.assertFalse(thin_edge)
        self.assertTrue(upright_notebook)

    def test_custom_paper_roi_recovers_folded_sheet_and_owner(self) -> None:
        module = ObjectDetectModule(
            model=_FakeCustomRoiPaperModel(),
            enable_smartphone_fallback=False,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        detections = module._detect_custom_paper_rois(
            frame,
            person_rois=[
                {
                    "bbox_xyxy": [300, 300, 700, 1000],
                    "track_id": 7,
                    "person_id": "STUDENT_LEFT",
                }
            ],
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "cheat_sheet")
        self.assertEqual(
            detections[0]["source"],
            "custom_person_roi_paper_model",
        )
        self.assertEqual(detections[0]["bbox_xyxy"], [270, 364, 340, 424])
        self.assertEqual(detections[0]["owner_track_id_hint"], 7)
        self.assertEqual(
            detections[0]["owner_person_id_hint"],
            "STUDENT_LEFT",
        )

    def test_roi_phone_shape_rejects_square_pen_box(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())

        self.assertFalse(
            module._is_plausible_roi_phone_bbox(
                [394, 676, 460, 736],
                frame_width=1920,
                frame_height=1080,
            )
        )
        self.assertTrue(
            module._is_plausible_roi_phone_bbox(
                [409, 684, 480, 710],
                frame_width=1920,
                frame_height=1080,
            )
        )

    def test_person_roi_remaps_phone_and_preserves_owner_hint(self) -> None:
        module = ObjectDetectModule(
            model=_FakeModel(),
            smartphone_model=_FakeFallbackModel(),
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        detections = module._detect_auxiliary_fallback(
            frame,
            person_rois=[
                {
                    "bbox_xyxy": [300, 300, 700, 1000],
                    "track_id": 7,
                    "person_id": "STUDENT_LEFT",
                }
            ],
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "smartphone")
        self.assertEqual(
            detections[0]["source"],
            "coco_person_roi_phone_fallback",
        )
        self.assertEqual(detections[0]["bbox_xyxy"], [290, 344, 320, 399])
        self.assertEqual(detections[0]["owner_track_id_hint"], 7)
        self.assertEqual(
            detections[0]["owner_person_id_hint"],
            "STUDENT_LEFT",
        )

    def test_nms_copies_owner_hint_from_overlapping_roi_box(self) -> None:
        module = ObjectDetectModule(model=_FakeModel())
        detections = [
            {
                "class_name": "smartphone",
                "confidence": 0.80,
                "bbox_xyxy": [100, 100, 180, 180],
            },
            {
                "class_name": "smartphone",
                "confidence": 0.60,
                "bbox_xyxy": [102, 102, 182, 182],
                "owner_track_id_hint": 9,
                "owner_person_id_hint": "STUDENT_RIGHT",
            },
        ]

        kept = module._nms(detections, iou_threshold=0.45)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["owner_track_id_hint"], 9)
        self.assertEqual(
            kept[0]["owner_person_id_hint"],
            "STUDENT_RIGHT",
        )

    def test_interaction_zone_excludes_object_near_feet(self) -> None:
        interaction_zone = [415, 318, 831, 946]

        self.assertTrue(
            _bbox_center_inside(
                [623, 597, 703, 670],
                interaction_zone,
            )
        )
        self.assertFalse(
            _bbox_center_inside(
                [700, 970, 745, 1020],
                interaction_zone,
            )
        )


if __name__ == "__main__":
    unittest.main()
