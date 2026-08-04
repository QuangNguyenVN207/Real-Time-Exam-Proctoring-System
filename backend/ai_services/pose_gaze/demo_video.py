"""
Demo pipeline pose_gaze sử dụng MediaPipe Holistic + YOLO Person Tracker.

Kỹ thuật & Cấu hình chuẩn Paper/Dự án:
  - Crop 1920×900 từ video gốc (loại bỏ phía trên/bàn thừa)
  - Chọn 2 người gần camera nhất (diện tích bbox lớn nhất)
  - Sắp xếp từ trái qua phải để gán P1 (Gauche) / P2 (Droite)
  - Chạy MediaPipe Holistic cho từng người (Face Mesh + Iris + Pose + Hands)
  - Scale cửa sổ hiển thị về 960×450 @ 24 FPS (Đúng tỷ lệ half của 1920×900)

Chạy từ repo root:
    python -m backend.ai_services.pose_gaze.demo_video data/raw_video/WIN_20260726_11_41_55_Pro.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from backend.ai_services.pose_gaze.dataset.stage2.holistic_runner import DatasetHolisticRunner
from backend.ai_services.pose_gaze.tracking.detectors import UltralyticsPersonDetector
from backend.ai_services.pose_gaze.tracking.tracker import IoUPersonTracker
from backend.ai_services.pose_gaze.tracking.schemas import TrackedPerson
from backend.ai_services.pose_gaze.dataset.stage2.crop import crop_with_padding
from backend.ai_services.pose_gaze.dataset.stage2.schemas import CropMeta
from backend.ai_services.pose_gaze.dataset.stage2.head_pose import estimate_head_pose, classify_head_direction
from backend.ai_services.pose_gaze.dataset.stage2.gaze import extract_gaze
from backend.ai_services.pose_gaze.dataset.stage2.quality import face_quality_score
from backend.ai_services.pose_gaze.holistic_landmarks import EYE_CONNECTIONS, HAND_CONNECTIONS, LIP_CONNECTIONS

PERSON_COLORS = [
    (0, 200, 80),    # P1 (Trái) — Xanh lá
    (230, 130, 0),   # P2 (Phải) — Cam
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Đường dẫn video đầu vào")
    p.add_argument("--output", type=Path, default=None, help="Đường dẫn video xuất ra (nếu muốn lưu)")
    p.add_argument("--yolo-model", type=Path, default=Path("weights/yolov8n.pt"))
    p.add_argument("--crop-w", type=int, default=1920, help="Rộng crop từ video gốc")
    p.add_argument("--crop-h", type=int, default=900, help="Cao crop từ video gốc")
    p.add_argument("--display-w", type=int, default=960, help="Rộng hiển thị (1/2 crop)")
    p.add_argument("--display-h", type=int, default=450, help="Cao hiển thị (1/2 crop)")
    p.add_argument("--fps", type=float, default=24.0, help="FPS của video demo")
    p.add_argument("--conf", type=float, default=0.35, help="Ngưỡng conf YOLO")
    p.add_argument("--no-display", action="store_true", help="Tắt hiển thị cửa sổ")
    p.add_argument("--save", action="store_true", help="Lưu video output ra file (mặc định không lưu)")
    return p.parse_args()


def select_front_persons(
    tracked_persons: list[TrackedPerson],
    max_n: int = 2,
    min_confidence: float = 0.5,
    min_area_ratio: float = 0.01,
    frame_width: int = 1920,
    frame_height: int = 900,
) -> list[TrackedPerson]:
    """Chọn max_n người gần camera nhất (diện tích bbox lớn nhất), lọc theo độ rõ + kích thước, sort trái -> phải."""
    present = [p for p in tracked_persons if p.is_present and p.confidence >= min_confidence]
    
    # Lọc theo diện tích tối thiểu (1% frame)
    valid = []
    for p in present:
        area_ratio = (p.bbox.width * p.bbox.height) / max(frame_width * frame_height, 1)
        if area_ratio >= min_area_ratio:
            valid.append(p)
    
    by_area = sorted(valid, key=lambda p: p.bbox.width * p.bbox.height, reverse=True)
    front = by_area[:max_n]
    return sorted(front, key=lambda p: p.bbox.x1)


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {args.input}")

    cap = cv2.VideoCapture(str(args.input))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    crop_w = min(args.crop_w, src_w)
    crop_h = min(args.crop_h, src_h)
    disp_w, disp_h = args.display_w, args.display_h
    sx, sy = disp_w / crop_w, disp_h / crop_h

    # Detector & Tracker
    yolo_weights = args.yolo_model if args.yolo_model.exists() else Path("yolov8n.pt")
    detector = UltralyticsPersonDetector(yolo_weights, confidence_threshold=args.conf)
    tracker = IoUPersonTracker(max_tracks=2, min_iou=0.1, max_missed_frames=15)

    # Video Writer (chỉ bật khi truyền --save)
    writer = None
    if args.save:
        out_path = args.output or args.input.with_name(args.input.stem + "_mediapipe_demo.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (disp_w, disp_h))
        print(f"[INFO] Lưu demo MediaPipe → {out_path} ({disp_w}×{disp_h} @ {args.fps} FPS)")

    WIN = "ExamGuard MediaPipe Demo"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, disp_w, disp_h)

    frame_idx = 0
    frame_interval = 1.0 / args.fps

    with DatasetHolisticRunner() as runner:
        print(f"[INFO] Bắt đầu Demo MediaPipe Holistic (Backend: {runner._extractor.backend}) @ 24 FPS...")

        while True:
            t_frame = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Đã phát hết video.")
                break

            # 1. Crop 1920×900 từ video gốc
            frame_crop = frame[:crop_h, :crop_w]

            # 2. YOLO detect & Track người
            detections = detector.detect(frame_crop)
            tracked = tracker.update(detections)

            # 3. Lọc 2 người gần camera nhất
            front_persons = select_front_persons(
                tracked,
                max_n=2,
                frame_width=frame_crop.shape[1],
                frame_height=frame_crop.shape[0],
            )

            # 4. Canvas để vẽ ở kích thước display (960×450)
            canvas = cv2.resize(frame_crop, (disp_w, disp_h))

            # HUD Bar ở phía trên
            cv2.rectangle(canvas, (0, 0), (disp_w, 36), (255, 255, 255), -1)
            cv2.rectangle(canvas, (0, 0), (disp_w, 36), (13, 71, 107), 1)
            cv2.putText(canvas, f"ExamGuard Demo | Frame: {frame_idx} | 24 FPS | Tracked: {len(front_persons)} Persons",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (13, 71, 107), 1)

            # 5. Chạy MediaPipe cho 2 người được chọn
            for pi, person in enumerate(front_persons):
                color = PERSON_COLORS[pi % len(PERSON_COLORS)]
                
                crop_bgr, crop_bbox = crop_with_padding(frame_crop, person.bbox, padding=0.15)
                
                meta = CropMeta(
                    clip_id="demo", window_id="demo", frame_idx=frame_idx,
                    track_id=person.track_id,
                    bbox_x1=person.bbox.x1, bbox_y1=person.bbox.y1,
                    bbox_x2=person.bbox.x2, bbox_y2=person.bbox.y2,
                    crop_x1=crop_bbox.x1, crop_y1=crop_bbox.y1,
                    crop_x2=crop_bbox.x2, crop_y2=crop_bbox.y2,
                    frame_h=crop_h, frame_w=crop_w, timestamp_ms=frame_idx * (1000 / args.fps),
                    tracking_confidence=person.confidence,
                )

                # Chạy MediaPipe Holistic
                raw = runner.process_crop(meta, crop_bgr, crop_bbox, full_frame=frame_crop)

                # Bbox trên Canvas (scale về 960×450)
                bx1, by1 = int(person.bbox.x1 * sx), int(person.bbox.y1 * sy)
                bx2, by2 = int(person.bbox.x2 * sx), int(person.bbox.y2 * sy)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)

                student_tag = f"P{pi+1} ({'Gauche' if pi==0 else 'Droite'})"
                
                head_txt = ""
                gaze_txt = ""
                if raw.face_crop_lm is not None:
                    crop_w_px = int(crop_bbox.x2 - crop_bbox.x1)
                    crop_h_px = int(crop_bbox.y2 - crop_bbox.y1)
                    q = face_quality_score(raw.face_crop_lm, raw.face_vis, crop_h_px, crop_w_px, raw.face_lm_indices)
                    
                    hp_res = estimate_head_pose(raw.face_crop_lm, crop_w_px, crop_h_px, raw.face_lm_indices)
                    if hp_res is not None:
                        yaw, pitch, roll = hp_res
                        head_dir = classify_head_direction(yaw, pitch)
                        head_txt = f"Head:{head_dir}"

                    gaze_dir, g_valid = extract_gaze(raw.face_crop_lm, q, raw.face_lm_indices)
                    gaze_txt = f"Gaze:{gaze_dir}" if g_valid else "Gaze:low_q"

                    # *_frame_lm là pixel trong frame_crop, sau đó scale lên canvas.
                    if raw.face_frame_lm is not None and raw.face_lm_indices is not None:
                        positions = {int(index): pos for pos, index in enumerate(raw.face_lm_indices)}
                        face_edges = [
                            (positions[start], positions[end])
                            for start, end in (*EYE_CONNECTIONS, *LIP_CONNECTIONS, (10, 1), (1, 152))
                            if start in positions and end in positions
                        ]
                        face_points = raw.face_frame_lm
                        valid = np.isfinite(face_points[:, :2]).all(axis=1)
                        pixels = np.zeros((len(face_points), 2), dtype=np.int32)
                        pixels[valid, 0] = np.clip(face_points[valid, 0] * sx, 0, disp_w - 1).astype(np.int32)
                        pixels[valid, 1] = np.clip(face_points[valid, 1] * sy, 0, disp_h - 1).astype(np.int32)
                        for start, end in face_edges:
                            if valid[start] and valid[end]:
                                cv2.line(canvas, tuple(pixels[start]), tuple(pixels[end]), (0, 255, 255), 1)
                        for point in pixels[valid]:
                            cv2.circle(canvas, tuple(point), 1, (0, 255, 255), -1)

                for points, edges in (
                    (raw.pose_frame_lm, None),
                    (raw.left_hand_frame_lm, HAND_CONNECTIONS),
                    (raw.right_hand_frame_lm, HAND_CONNECTIONS),
                ):
                    if points is None:
                        continue
                    valid = np.isfinite(points[:, :2]).all(axis=1)
                    pixels = np.zeros((len(points), 2), dtype=np.int32)
                    pixels[valid, 0] = np.clip(points[valid, 0] * sx, 0, disp_w - 1).astype(np.int32)
                    pixels[valid, 1] = np.clip(points[valid, 1] * sy, 0, disp_h - 1).astype(np.int32)
                    if edges is not None:
                        for start, end in edges:
                            if start < len(points) and end < len(points) and valid[start] and valid[end]:
                                cv2.line(canvas, tuple(pixels[start]), tuple(pixels[end]), color, 1)
                    for point in pixels[valid]:
                        cv2.circle(canvas, tuple(point), 2, color, -1)

                # Label thông tin phía trên Bbox
                info_str = f"{student_tag} | {head_txt} | {gaze_txt}".strip(" |")
                label_y = max(45, by1 - 8)
                cv2.rectangle(canvas, (bx1, label_y - 18), (bx1 + len(info_str)*8 + 6, label_y + 2), color, -1)
                cv2.putText(canvas, info_str, (bx1 + 3, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            if writer is not None:
                writer.write(canvas)

            if not args.no_display:
                cv2.imshow(WIN, canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Đã ấn 'q' để thoát.")
                    break

            # Khống chế đúng 24 FPS
            elapsed = time.time() - t_frame
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
