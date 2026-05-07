from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch


def import_torch_npu_if_available() -> str:
    try:
        import torch_npu  # noqa: F401
        return getattr(torch_npu, "__version__", "imported")
    except Exception as exc:
        return f"not available: {exc}"


def get_triton_version() -> str:
    try:
        import triton
        return getattr(triton, "__version__", "unknown")
    except Exception as exc:
        return f"not available: {exc}"


def resolve_device(device_arg: str = "auto") -> torch.device:
    import_torch_npu_if_available()
    if device_arg != "auto":
        return torch.device(device_arg)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    raise RuntimeError("No NPU/CUDA device is available. Use --dry-run locally or run on the server.")


def synchronize(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class RuntimeInfo:
    device: torch.device
    torch_version: str
    torch_npu_version: str
    triton_version: str


def collect_runtime_info(device_arg: str = "auto") -> RuntimeInfo:
    torch_npu_version = import_torch_npu_if_available()
    device = resolve_device(device_arg)
    return RuntimeInfo(
        device=device,
        torch_version=torch.__version__,
        torch_npu_version=torch_npu_version,
        triton_version=get_triton_version(),
    )


def time_repeated(fn: Callable[[], object], device: torch.device, repeat: int) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize(device)
    end = time.perf_counter()
    return (end - start) * 1000.0 / repeat
