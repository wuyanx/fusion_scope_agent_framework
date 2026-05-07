from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Failed to import Triton/Triton-Ascend in this environment.") from exc


@triton.jit
def _sigmoid(z):
    return 1.0 / (1.0 + tl.exp(-z))


@triton.jit
def _tanh_approx(z):
    return 2.0 * _sigmoid(2.0 * z) - 1.0


@triton.jit
def chain_group_kernel(
    in_ptr,
    out_ptr,
    coeff_ptr,
    GROUP_START: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    y = tl.load(in_ptr + offsets).to(tl.float32)
    for j in tl.static_range(0, GROUP_SIZE):
        op_idx = GROUP_START + j
        base = op_idx * 4
        a = tl.load(coeff_ptr + base + 0)
        b = tl.load(coeff_ptr + base + 1)
        c = tl.load(coeff_ptr + base + 2)
        d = tl.load(coeff_ptr + base + 3)
        y = _tanh_approx(a * y + b) + 0.125 * _sigmoid(c * y + d) + 0.03125 * y
    tl.store(out_ptr + offsets, y)


class FusionPlan:
    def __init__(self, x: torch.Tensor, coeffs: torch.Tensor, ops: int, fusion_group_size: int, block: int) -> None:
        if ops <= 0 or fusion_group_size <= 0:
            raise ValueError("ops and fusion_group_size must be positive")
        if ops % fusion_group_size != 0:
            raise ValueError("ops must be divisible by fusion_group_size")
        if x.numel() % block != 0:
            raise ValueError("x.numel() must be divisible by block because this kernel uses unmasked load/store")
        if coeffs.numel() != ops * 4:
            raise ValueError("coeffs should contain ops*4 values")
        if not x.is_contiguous() or not coeffs.is_contiguous():
            raise ValueError("x and coeffs must be contiguous")
        self.x = x
        self.coeffs = coeffs
        self.ops = ops
        self.fusion_group_size = fusion_group_size
        self.block = block
        self.n_elements = x.numel()
        self.num_groups = ops // fusion_group_size
        self.y = torch.empty_like(x)
        self.tmp0 = torch.empty_like(x) if self.num_groups > 1 else None
        self.tmp1 = torch.empty_like(x) if self.num_groups > 2 else None

    @property
    def num_kernel_launches(self) -> int:
        return self.num_groups

    def _intermediate_buffer(self, idx: int) -> torch.Tensor:
        if self.tmp1 is None:
            assert self.tmp0 is not None
            return self.tmp0
        return self.tmp0 if idx % 2 == 0 else self.tmp1

    def run(self) -> torch.Tensor:
        grid = (triton.cdiv(self.n_elements, self.block),)
        src = self.x
        for gid in range(self.num_groups):
            is_last = gid == self.num_groups - 1
            dst = self.y if is_last else self._intermediate_buffer(gid)
            chain_group_kernel[grid](
                src, dst, self.coeffs,
                GROUP_START=gid * self.fusion_group_size,
                GROUP_SIZE=self.fusion_group_size,
                BLOCK=self.block,
            )
            src = dst
        return self.y
