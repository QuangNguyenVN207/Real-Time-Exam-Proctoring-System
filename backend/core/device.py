"""Chọn thiết bị suy luận dùng chung cho mọi module AI (pose_gaze, paper_count, object_detect...).

Thứ tự ưu tiên khi "auto":
    1. GPU NVIDIA (CUDA)                  -> model .pt (PyTorch), ONNX Runtime CUDAExecutionProvider
    2. GPU Intel tích hợp (OpenVINO)      -> model export (_openvino_model / .onnx), ONNX Runtime OpenVINOExecutionProvider
    3. CPU                                -> model .pt (hoặc .onnx nếu chỉ có .onnx)

Ép thủ công bằng biến môi trường:  PROCTOR_DEVICE = auto | cuda | intel | cpu   (mặc định auto)
Các giá trị `device=` cũ trong code ("0", "cuda", "GPU", "cpu"...) vẫn được hiểu qua resolve_device().
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

CUDA, INTEL, CPU = "cuda", "intel", "cpu"

# Thứ tự định dạng model khi chạy Intel iGPU. Đổi thành ("onnx", "openvino") nếu muốn ưu tiên ONNX.
INTEL_FORMAT_PRIORITY = ("openvino", "onnx")


@dataclass(frozen=True)
class DeviceInfo:
    kind: str                     # "cuda" | "intel" | "cpu"
    name: str                     # tên hiển thị
    torch_device: str             # "0" hoặc "cpu" (kiểu ultralytics predict)
    torch_to: str                 # "cuda:0" hoặc "cpu" (kiểu model.to())
    ort_providers: List[Any] = field(default_factory=list)  # InferenceSession(providers=...)

    @property
    def use_onnx(self) -> bool:
        return self.kind != CUDA


# ---------------------------------------------------------------- phát hiện phần cứng
@lru_cache(maxsize=1)
def _ort_available() -> Tuple[str, ...]:
    try:
        import onnxruntime as ort
        return tuple(ort.get_available_providers())
    except Exception:
        return ()


@lru_cache(maxsize=1)
def _detect_cuda() -> Tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, ""


@lru_cache(maxsize=1)
def _detect_intel_gpu() -> Tuple[bool, str]:
    """OpenVINO thấy thiết bị 'GPU' (Intel iGPU/Arc)."""
    try:
        import openvino as ov
        core = ov.Core()
        for dev in core.available_devices:
            if dev.startswith("GPU"):
                return True, str(core.get_property(dev, "FULL_DEVICE_NAME"))
    except Exception:
        pass
    return False, ""


def _build(kind: str) -> Optional[DeviceInfo]:
    avail = _ort_available()
    if kind == CUDA:
        ok, name = _detect_cuda()
        if not ok:
            return None
        eps = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else []) + ["CPUExecutionProvider"]
        return DeviceInfo(CUDA, name, "0", "cuda:0", eps)
    if kind == INTEL:
        ok, name = _detect_intel_gpu()
        if not ok:
            return None
        if "OpenVINOExecutionProvider" in avail:
            eps: List[Any] = [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"]
        else:  # có iGPU + openvino nhưng thiếu onnxruntime-openvino: ultralytics/openvino vẫn dùng được
            eps = ["CPUExecutionProvider"]
        return DeviceInfo(INTEL, name, "cpu", "cpu", eps)
    return DeviceInfo(CPU, "CPU", "cpu", "cpu", ["CPUExecutionProvider"])


# ---------------------------------------------------------------- chọn thiết bị
_ALIASES = {
    "": None, "auto": None, "gpu": None,            # "GPU" chung chung -> cứ auto (NVIDIA trước, Intel sau)
    "cuda": CUDA, "cuda:0": CUDA, "nvidia": CUDA,
    "intel": INTEL, "intel:gpu": INTEL, "openvino": INTEL,
    "cpu": CPU,
}


def _normalize(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    key = str(requested).strip().lower()
    if key.isdigit():          # "0", "1" -> GPU NVIDIA
        return CUDA
    return _ALIASES.get(key)


@lru_cache(maxsize=None)
def _resolve(pref: Optional[str]) -> DeviceInfo:
    order = [CUDA, INTEL, CPU]
    if pref == CPU:
        order = [CPU]
    elif pref in (CUDA, INTEL):
        order = [pref] + [k for k in order if k != pref]
    for kind in order:
        info = _build(kind)
        if info is not None:
            if pref in (CUDA, INTEL) and kind != pref:
                print(f"[DEVICE] Yêu cầu '{pref}' không khả dụng -> dùng {kind}")
            return info
    return _build(CPU)  # type: ignore[return-value]


def resolve_device(requested: Optional[str] = None) -> DeviceInfo:
    """requested=None/'auto' -> theo PROCTOR_DEVICE (mặc định auto). Giá trị khác -> ưu tiên đúng thiết bị đó."""
    pref = _normalize(requested)
    if pref is None:
        pref = _normalize(os.environ.get("PROCTOR_DEVICE", "auto"))
    return _resolve(pref)


def select_device() -> DeviceInfo:
    return resolve_device(None)


# ---------------------------------------------------------------- chọn file model theo thiết bị
def resolve_model_path(model_path: Any, info: Optional[DeviceInfo] = None) -> str:
    """Từ 1 đường dẫn model (.pt / .onnx / thư mục *_openvino_model) trả về bản phù hợp nhất với thiết bị.

    Quy ước tên (sinh bởi ultralytics export, xem scripts/export_models.py):
        weights/best.pt  ->  weights/best_openvino_model/   weights/best.onnx
    Tên dạng 'best (1).pt' cũng khớp với 'best_openvino_model' / 'best.onnx'.
    """
    info = info or select_device()
    original = Path(str(model_path))
    parent = original.parent
    base = re.sub(r"(_openvino_model|\.pt|\.onnx)$", "", original.name)
    stems = []
    for s in (base, re.sub(r"\s*\(\d+\)$", "", base)):
        if s and s not in stems:
            stems.append(s)

    def find(kind: str) -> Optional[Path]:
        for s in stems:
            cand = {"pt": parent / f"{s}.pt", "onnx": parent / f"{s}.onnx",
                    "openvino": parent / f"{s}_openvino_model"}[kind]
            if cand.exists():
                return cand
        if kind == "pt" and parent.exists():          # best_openvino_model -> 'best (1).pt'
            for s in stems:
                hits = sorted(parent.glob(f"{s}*.pt"))
                if hits:
                    return hits[0]
        return None

    if info.kind == CUDA:
        order = ["pt", "onnx", "openvino"]
    elif info.kind == INTEL:
        order = list(INTEL_FORMAT_PRIORITY) + ["pt"]
    else:
        order = ["pt", "onnx", "openvino"]

    for kind in order:
        found = find(kind)
        if found is not None:
            if found != original:
                print(f"[DEVICE] {info.kind}: dùng model {found} (thay cho {original})")
            if info.kind == INTEL and kind == "pt":
                print(f"[DEVICE] Chưa có bản OpenVINO/ONNX của {original.name} -> chạy .pt trên CPU. "
                      "Chạy: python scripts/export_models.py")
            return str(found)
    return str(original)


def make_ort_session(model_path: str, **kwargs):
    """Tạo onnxruntime.InferenceSession theo thiết bị đã chọn; lỗi provider thì rơi về CPU."""
    import onnxruntime as ort
    info = select_device()
    try:
        return ort.InferenceSession(model_path, providers=info.ort_providers, **kwargs)
    except Exception as e:
        print(f"[DEVICE] Lỗi tạo session với {info.kind} ({e}) -> rơi về CPU")
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], **kwargs)


if __name__ == "__main__":
    print(select_device())