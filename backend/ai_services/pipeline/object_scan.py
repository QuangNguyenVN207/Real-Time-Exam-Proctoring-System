"""Chạy YOLO vật cấm, lọc nhãn giấy và bác bỏ ứng viên điện thoại nằm trên ghế.

Sao y app_webrtc.py:255-302.
"""
from __future__ import annotations

from typing import Any, Optional

from backend.core.config import settings

__all__ = ["PAPER_LABELS", "scan_objects"]

PAPER_LABELS = {"paper", "document", "cheatsheet", "sheet", "tai_lieu", "giay", "book"}

#: Model vật cấm chỉ có 5 lớp (smartwatch/earphone/cheat_sheet/smartphone/calculator) và
#: nhãn `smartphone` của nó khớp cả với LƯNG GHẾ NHỰA bo góc có lỗ bầu dục — hình dạng
#: gần giống điện thoại có cụm camera. Đo 2026-10-31 trên weights/best_openvino_model:
#: 9/9 cảnh báo sai là lưng ghế (video_test_demo t=16.2-17.2, cheatsheet t=0.5-10.0) với
#: điểm 0.58-0.81, còn model COCO gốc gọi đúng là `chair` ở 0.87. Vì vậy ứng viên điện
#: thoại nào nằm chồng lên hộp chair/couch của model COCO thì bị bỏ.
PHONE_LABELS = {"smartphone", "cell_phone"}

#: Lưới ô chồng nhau cho `multi_scale`. 3 cột × 2 hàng của khung 1920×1080 -> ô 640×540,
#: letterbox vào imgsz=640 gần như giữ nguyên kích thước thật của vật.
TILE_GRID = (3, 2)
TILE_OVERLAP = 0.20


def _canonical(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _iou(a: list[int], b: list[int]) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, ix) * max(0.0, iy)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms_detections(dets: list[dict[str, Any]], iou_threshold: float = 0.45) -> list[dict[str, Any]]:
    """Gộp kết quả các ô chồng lấn: cùng nhãn và IoU đủ cao thì giữ điểm cao nhất."""
    kept: list[dict[str, Any]] = []
    for det in sorted(dets, key=lambda d: -float(d.get("confidence", 0.0))):
        box = det.get("bbox", [0, 0, 0, 0])
        if all(_canonical(other.get("label", "")) != _canonical(det.get("label", ""))
               or _iou(box, other.get("bbox", [0, 0, 0, 0])) < iou_threshold
               for other in kept):
            kept.append(det)
    return kept


def _confuser_class_ids(confuser_model: Any) -> list[int]:
    """Đổi settings.smartphone_confuser_class_names thành id theo đúng bảng tên của model."""
    wanted = {_canonical(name) for name in settings.smartphone_confuser_class_names}
    names = getattr(confuser_model, "names", {}) or {}
    return sorted(int(cid) for cid, label in names.items() if _canonical(label) in wanted)


def _intersection_over_smaller(a: list[float], b: list[float]) -> float:
    """Giao diện / diện tích hộp NHỎ HƠN — cùng thước đo ObjectDetectModule đang dùng."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, ix) * max(0.0, iy)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _confuser_boxes(confuser_model: Any, frame: Any) -> list[list[float]]:
    ids = _confuser_class_ids(confuser_model)
    if not ids:
        return []
    results = confuser_model(
        frame,
        conf=settings.smartphone_confuser_confidence_threshold,
        classes=ids,
        verbose=False,
    )
    boxes = []
    for result in results:
        for box in getattr(result, "boxes", []) or []:
            boxes.append([float(v) for v in box.xyxy[0]])
    return boxes


def _drop_chair_phones(
    kept: list[dict[str, Any]],
    frame: Any,
    confuser_model: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Bỏ ứng viên smartphone nằm trên ghế. Trả (danh sách còn lại, số bị bỏ)."""
    phones = [det for det in kept if _canonical(det.get("label", "")) in PHONE_LABELS]
    if not phones or confuser_model is None:
        return kept, 0
    chairs = _confuser_boxes(confuser_model, frame)
    if not chairs:
        return kept, 0
    threshold = settings.smartphone_confuser_overlap_threshold
    dropped = {
        id(det)
        for det in phones
        if max((_intersection_over_smaller(det.get("bbox", [0, 0, 0, 0]), chair)
                for chair in chairs), default=0.0) >= threshold
    }
    if not dropped:
        return kept, 0
    return [det for det in kept if id(det) not in dropped], len(dropped)


def _tile_boxes(
    yolo_model: Any,
    frame: Any,
    timestamp: float,
    grid: tuple[int, int],
    overlap: float,
) -> list[dict[str, Any]]:
    """Chạy model vật cấm trên lưới ô chồng nhau, trả detections toạ độ khung hình gốc.

    Lý do: ultralytics luôn letterbox ảnh vào imgsz=640, nên với khung 1920×1080 một chiếc
    điện thoại 78×62 px chỉ còn ~26×21 px và model không thấy gì (đo: 0.00-0.06). Cùng
    trọng số đó chạy trên ô 640×540 cho ra 0.63-0.73.
    """
    cols, rows = grid
    height, width = frame.shape[:2]
    tile_w, tile_h = width / cols, height / rows
    dx, dy = tile_w * overlap, tile_h * overlap
    out: list[dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tile_w - (dx if c else 0)))
            x2 = min(width, int((c + 1) * tile_w + (dx if c < cols - 1 else 0)))
            y1 = max(0, int(r * tile_h - (dy if r else 0)))
            y2 = min(height, int((r + 1) * tile_h + (dy if r < rows - 1 else 0)))
            result = yolo_model.process_frame(frame[y1:y2, x1:x2], timestamp)
            for det in (result or {}).get("detections", []):
                bx, by, bx2, by2 = det["bbox"]
                shifted = dict(det)
                shifted["bbox"] = [bx + x1, by + y1, bx2 + x1, by2 + y1]
                out.append(shifted)
    return _nms_detections(out)


def scan_objects(
    yolo_model: Any,
    frame: Any,
    timestamp: float,
    confuser_model: Any = None,
    multi_scale: bool = False,
) -> Optional[dict[str, Any]]:
    """Trả alert dict `module="object_detect"` khi còn vật cấm, ngược lại None.

    `confuser_model` là model COCO gốc (yolov8n) — cùng cái đang được nạp làm person
    detector. Không truyền thì coi như không có cổng phụ này.
    `multi_scale=True` chạy trên lưới TILE_GRID thay vì một lần full-frame: đắt hơn ~5.5 lần
    (đo 33.8 -> 186.9 ms/khung trên iGPU Arc) nhưng là cách duy nhất hiện tại bắt được vật
    nhỏ; không giả định vị trí nên điện thoại bên hông/dưới bàn vẫn nằm trong điều kiện đo.
    """
    if multi_scale:
        detections = _tile_boxes(yolo_model, frame, timestamp, TILE_GRID, TILE_OVERLAP)
        result = {"detections": detections} if detections else None
    else:
        result = yolo_model.process_frame(frame, timestamp)
    if result is None:
        return None

    detections = result.get("detections", [])
    kept = [det for det in detections if _canonical(det.get("label", "")) not in PAPER_LABELS]
    kept, rejected = _drop_chair_phones(kept, frame, confuser_model)
    if rejected:
        result["confuser_rejected"] = rejected
    if not kept:
        return None

    result["detections"] = kept
    result["status"] = "alert"
    result["module"] = "object_detect"
    return result
