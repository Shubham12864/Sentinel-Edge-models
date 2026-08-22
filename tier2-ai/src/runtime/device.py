"""Automatic device selection: GPU when available, CPU otherwise.

One helper decides the execution plan for every accelerator-backed stage so
the same codebase runs GPU-first on a presentation machine and degrades
cleanly to CPU elsewhere.  Torch (YOLO/Ultralytics) and ONNX Runtime
(InsightFace/ArcFace) are probed independently because their CUDA stacks can
be installed separately.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DevicePlan:
    torch_device: str   # "cuda" | "cpu"  -> passed to Ultralytics
    ctx_id: int         # 0 | -1           -> passed to InsightFace prepare()
    cuda_torch: bool
    cuda_onnx: bool

    @property
    def label(self) -> str:
        if self.cuda_torch and self.cuda_onnx:
            return "gpu"
        if self.cuda_torch or self.cuda_onnx:
            return f"partial-gpu ({'torch' if self.cuda_torch else 'onnxruntime'} only)"
        return "cpu"


def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _onnx_cuda() -> bool:
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def resolve_device(requested: str | int | None = None) -> DevicePlan:
    """Return the best execution plan.

    requested=None  -> auto-detect: use each backend's GPU when present.
    requested='cpu' -> force full CPU regardless of hardware.
    requested='cuda'/'gpu'/'0' -> require at least one CUDA stack, otherwise
    raise; backends without CUDA fall back to CPU individually.
    """
    cuda_torch, cuda_onnx = _torch_cuda(), _onnx_cuda()
    if requested is not None:
        text = str(requested).strip().lower()
        if text in ("cpu", "-1"):
            return DevicePlan("cpu", -1, cuda_torch, cuda_onnx)
        if text in ("cuda", "gpu", "0"):
            if not (cuda_torch or cuda_onnx):
                raise RuntimeError(
                    "CUDA requested but neither torch nor onnxruntime exposes a GPU "
                    "(install torch+cu / onnxruntime-gpu, or pass device='cpu')"
                )
            return DevicePlan("cuda" if cuda_torch else "cpu", 0 if cuda_onnx else -1, cuda_torch, cuda_onnx)
        raise ValueError(f"Unknown device request: {requested!r} (use 'cuda', 'cpu' or None)")
    return DevicePlan("cuda" if cuda_torch else "cpu", 0 if cuda_onnx else -1, cuda_torch, cuda_onnx)
