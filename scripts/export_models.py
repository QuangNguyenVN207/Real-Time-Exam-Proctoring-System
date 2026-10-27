"""Export YOLO .pt -> OpenVINO + ONNX để máy chỉ có Intel iGPU dùng được.

    python scripts/export_models.py                       # export các model mặc định trong weights/
    python scripts/export_models.py weights/best.pt --imgsz 640

Kết quả nằm cạnh file .pt:  <tên>_openvino_model/  và  <tên>.onnx
(backend.core.device.resolve_model_path sẽ tự tìm đúng bản theo thiết bị).
"""
import argparse
from pathlib import Path

DEFAULTS = ["weights/best (1).pt", "weights/yolov8n.pt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="*", default=DEFAULTS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--formats", nargs="+", default=["openvino", "onnx"])
    args = ap.parse_args()

    from ultralytics import YOLO
    for w in args.weights:
        if not Path(w).exists():
            print(f"[SKIP] không thấy {w}")
            continue
        for fmt in args.formats:
            print(f"[EXPORT] {w} -> {fmt}")
            YOLO(w).export(format=fmt, imgsz=args.imgsz)


if __name__ == "__main__":
    main()