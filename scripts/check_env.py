"""Chẩn đoán môi trường suy luận (đặc biệt cho WSL2 + NVIDIA).   Chạy:  python scripts/check_env.py
Chạy từ thư mục gốc dự án, trong đúng .venv của dự án."""
import os
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, os.getcwd())


def section(t):
    print(f"\n=== {t} ===")


section("Hệ thống")
is_wsl = "microsoft" in platform.release().lower()
print("Python:", sys.version.split()[0], "| OS:", platform.platform(), "| WSL:", is_wsl)

section("nvidia-smi (driver NVIDIA do Windows cung cấp, KHÔNG cài driver trong WSL)")
if shutil.which("nvidia-smi"):
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
else:
    print("KHÔNG thấy nvidia-smi -> cập nhật driver NVIDIA trên Windows, rồi chạy `wsl --shutdown` và mở lại WSL")

section("PyTorch")
try:
    import torch
    print("torch", torch.__version__, "| build CUDA:", torch.version.cuda, "| cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    elif torch.version.cuda is None:
        print("-> Đây là bản torch CPU-only. Cài lại bản CUDA, ví dụ: "
              "uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
except Exception as e:
    print("Lỗi import torch:", e)

section("ONNX Runtime (FaceNet/insightface)")
try:
    import onnxruntime as ort
    print("onnxruntime", ort.__version__, "| providers:", ort.get_available_providers())
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("-> Thiếu CUDAExecutionProvider: gỡ `onnxruntime`, cài `onnxruntime-gpu` (chỉ giữ 1 trong 2)")
except Exception as e:
    print("Lỗi import onnxruntime:", e)

section("OpenVINO (chỉ cần cho máy Intel iGPU)")
try:
    import openvino as ov
    print("openvino devices:", ov.Core().available_devices)
except Exception as e:
    print("Không có openvino:", e)

section("Thiết bị mà dự án sẽ chọn")
try:
    from backend.core.device import select_device
    print(select_device())
except Exception as e:
    print("Lỗi:", e)

section("Microphone (sounddevice)")
try:
    import sounddevice as sd
    ins = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    print(ins if ins else "KHÔNG có thiết bị input nào")
    if not ins and is_wsl:
        print("-> WSL: cài `sudo apt install libportaudio2 libasound2-plugins pulseaudio-utils`, "
              "kiểm tra `pactl info`; cần Windows 11/WSLg và đã cấp quyền mic cho ứng dụng trên Windows")
except Exception as e:
    print("Lỗi sounddevice:", e)