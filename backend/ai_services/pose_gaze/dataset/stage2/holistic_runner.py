"""
Wrapper cho Dataset Stage 2 dựa trên HolisticLandmarkExtractor (từ holistic_landmarks.py).

Tự động tương thích với cả legacy MediaPipe (`mp.solutions.holistic`)
lẫn MediaPipe Tasks API mới (`holistic_landmarker.task` auto-download).

Tổ chức dữ liệu đầu ra thành HolisticRaw với cả tọa độ crop-normalized và frame-normalized.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.ai_services.pose_gaze.holistic_landmarks import (
    HolisticLandmarkExtractor,
    TrackHolisticResult,
)
from backend.ai_services.pose_gaze.tracking.schemas import BoundingBox, TrackedPerson
from .common import CROP_PADDING, HOLISTIC_MODEL_COMPLEXITY
from .schemas import CropMeta, HolisticRaw

LOGGER = logging.getLogger(__name__)


class DatasetHolisticRunner:
    """
    Chạy MediaPipe Holistic cho từng crop × track_id trong dataset offline.

    Sử dụng HolisticLandmarkExtractor bên dưới để hỗ trợ tự động cả 2 backend:
    - Legacy MediaPipe (`mp.solutions.holistic`) nếu mediapipe < 0.10.30
    - Tasks API (`holistic_landmarker.task`) nếu mediapipe >= 0.10.30 (tự động download model)
    """

    def __init__(
        self,
        *,
        model_complexity: int = HOLISTIC_MODEL_COMPLEXITY,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        crop_padding: float = CROP_PADDING,
    ) -> None:
        self._extractor = HolisticLandmarkExtractor(
            static_image_mode=True,
            model_complexity=model_complexity,
            smooth_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            crop_padding=crop_padding,
        )
        LOGGER.info(
            "DatasetHolisticRunner khởi tạo thành công (backend: %s)",
            self._extractor.backend,
        )

    def process_crop(
        self,
        meta: CropMeta,
        crop_bgr: np.ndarray,
        crop_bbox: BoundingBox,
        full_frame: np.ndarray | None = None,
    ) -> HolisticRaw:
        """
        Chạy MediaPipe Holistic trên một track.

        Parameters
        ----------
        meta        : CropMeta chứa thông tin vị trí track, frame size, ...
        crop_bgr    : ảnh BGR đã cắt (dùng nếu full_frame không được truyền)
        crop_bbox   : BoundingBox pixel tuyệt đối trong frame
        full_frame  : ảnh BGR gốc (nếu có, để trích xuất chính xác nhất)

        Returns
        -------
        HolisticRaw với tọa độ landmark theo crop VÀ frame-normalized.
        """
        # Nếu không có full_frame, tạo ảnh giả vừa đủ chứa crop_bbox
        if full_frame is None:
            h = max(meta.frame_h, int(crop_bbox.y2 + 1))
            w = max(meta.frame_w, int(crop_bbox.x2 + 1))
            frame_to_use = np.zeros((h, w, 3), dtype=np.uint8)
            x1, y1 = int(crop_bbox.x1), int(crop_bbox.y1)
            ch, cw = crop_bgr.shape[:2]
            frame_to_use[y1 : y1 + ch, x1 : x1 + cw] = crop_bgr
        else:
            frame_to_use = full_frame

        # Tạo TrackedPerson giả lập
        tracked_person = TrackedPerson(
            track_id=meta.track_id,
            bbox=BoundingBox(meta.bbox_x1, meta.bbox_y1, meta.bbox_x2, meta.bbox_y2),
            confidence=meta.tracking_confidence,
            age_frames=1,
            missed_frames=0,
            is_present=True,
        )

        res: TrackHolisticResult | None = self._extractor.process_track(
            frame_to_use,
            tracked_person,
            timestamp_ms=int(meta.timestamp_ms),
        )

        raw = HolisticRaw(meta=meta)
        if res is None:
            return raw

        # ── Parse Pose ────────────────────────────────────────────────────────
        if res.pose_landmarks:
            n = len(res.pose_landmarks)
            raw.pose_crop_lm  = np.array([[p.x, p.y, p.z] for p in res.pose_landmarks], dtype=np.float32)
            raw.pose_frame_lm = np.array(
                [[p.frame_x if p.frame_x is not None else p.x,
                  p.frame_y if p.frame_y is not None else p.y, p.z] for p in res.pose_landmarks],
                dtype=np.float32,
            )
            raw.pose_vis      = np.array([p.visibility or 0.0 for p in res.pose_landmarks], dtype=np.float32)
            raw.pose_presence = np.array([p.presence or 0.0 for p in res.pose_landmarks], dtype=np.float32)
            raw.missing_pose  = False

        # ── Parse Face ────────────────────────────────────────────────────────
        if res.selected_face_landmarks:
            n = len(res.selected_face_landmarks)
            raw.face_crop_lm  = np.array([[p.x, p.y, p.z] for p in res.selected_face_landmarks], dtype=np.float32)
            raw.face_frame_lm = np.array(
                [[p.frame_x if p.frame_x is not None else p.x,
                  p.frame_y if p.frame_y is not None else p.y, p.z] for p in res.selected_face_landmarks],
                dtype=np.float32,
            )
            raw.face_vis      = np.array([p.visibility or 0.0 for p in res.selected_face_landmarks], dtype=np.float32)
            raw.face_lm_indices = np.array([p.index for p in res.selected_face_landmarks], dtype=np.int32)
            raw.missing_face  = False

        # ── Parse Left Hand ───────────────────────────────────────────────────
        if res.left_hand_landmarks:
            raw.left_hand_crop_lm  = np.array([[p.x, p.y, p.z] for p in res.left_hand_landmarks], dtype=np.float32)
            raw.left_hand_frame_lm = np.array(
                [[p.frame_x if p.frame_x is not None else p.x,
                  p.frame_y if p.frame_y is not None else p.y, p.z] for p in res.left_hand_landmarks],
                dtype=np.float32,
            )
            raw.missing_left_hand  = False

        # ── Parse Right Hand ──────────────────────────────────────────────────
        if res.right_hand_landmarks:
            raw.right_hand_crop_lm  = np.array([[p.x, p.y, p.z] for p in res.right_hand_landmarks], dtype=np.float32)
            raw.right_hand_frame_lm = np.array(
                [[p.frame_x if p.frame_x is not None else p.x,
                  p.frame_y if p.frame_y is not None else p.y, p.z] for p in res.right_hand_landmarks],
                dtype=np.float32,
            )
            raw.missing_right_hand  = False

        return raw

    def close(self) -> None:
        pass

    def __enter__(self) -> "DatasetHolisticRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
