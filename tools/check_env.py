from __future__ import annotations

import platform
import sys

print("python:", sys.version)
print("executable:", sys.executable)
print("platform:", platform.platform())
try:
    import torch
    print("torch:", torch.__version__)
    try:
        import torch_npu
        print("torch_npu:", getattr(torch_npu, "__version__", "imported"))
    except Exception as exc:
        print("torch_npu: not available:", exc)
    print("cuda available:", torch.cuda.is_available())
    if hasattr(torch, "npu"):
        print("npu available:", torch.npu.is_available())
        try:
            print("npu device count:", torch.npu.device_count())
        except Exception as exc:
            print("npu device count failed:", exc)
except Exception as exc:
    print("torch import failed:", exc)
try:
    import triton
    print("triton:", getattr(triton, "__version__", "unknown"))
except Exception as exc:
    print("triton import failed:", exc)
