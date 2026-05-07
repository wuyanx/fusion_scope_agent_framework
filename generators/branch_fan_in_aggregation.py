from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from fusion_scope_core.scenario import Scenario
from fusion_scope_core.file_utils import write_text
from .common import write_common_project_files


def torch_reference_py() -> str:
    return dedent('''
    from __future__ import annotations

    import torch


    def make_input(n_elements: int, device: torch.device, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        data = torch.randn(n_elements, generator=gen, dtype=torch.float32) * 0.5
        return data.to(device=device)


    def make_coefficients(branches: int, device: torch.device) -> torch.Tensor:
        # Six coefficients per branch: a,b,c,d,e,f.
        vals = []
        for i in range(branches):
            t = float(i + 1)
            vals.extend([
                0.37 + 0.013 * t,
                -0.20 + 0.007 * t,
                0.19 + 0.011 * t,
                0.10 - 0.005 * t,
                0.60 + 0.003 * t,
                -0.15 + 0.002 * t,
            ])
        return torch.tensor(vals, device=device, dtype=torch.float32)


    def _branch_score_value(x: torch.Tensor, coeffs: torch.Tensor, branch_idx: int):
        base = branch_idx * 6
        a, b, c, d, e, f = coeffs[base:base + 6]
        score = torch.tanh(a * x + b) + 0.25 * torch.sigmoid(c * x + d)
        value = e * score + f * x
        return score, value


    def reference_output(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        branches = coeffs.numel() // 6
        scores = []
        values = []
        for i in range(branches):
            s, v = _branch_score_value(x, coeffs, i)
            scores.append(s)
            values.append(v)
        score_tensor = torch.stack(scores, dim=0)
        value_tensor = torch.stack(values, dim=0)
        weights = torch.softmax(score_tensor, dim=0)
        return torch.sum(weights * value_tensor, dim=0)
    ''').lstrip()


def kernels_py() -> str:
    # Unmasked load/store is intentional for current Ascend Triton/CANN 9 beta compatibility.
    return dedent('''
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
    def _branch_score_value(x, coeff_ptr, branch_idx: tl.constexpr):
        base = branch_idx * 6
        a = tl.load(coeff_ptr + base + 0)
        b = tl.load(coeff_ptr + base + 1)
        c = tl.load(coeff_ptr + base + 2)
        d = tl.load(coeff_ptr + base + 3)
        e = tl.load(coeff_ptr + base + 4)
        f = tl.load(coeff_ptr + base + 5)
        score = _tanh_approx(a * x + b) + 0.25 * _sigmoid(c * x + d)
        value = e * score + f * x
        return score, value


    @triton.jit
    def group_stats_kernel(
        x_ptr,
        group_max_ptr,
        group_sum_ptr,
        group_num_ptr,
        coeff_ptr,
        n_elements: tl.constexpr,
        GROUP_ID: tl.constexpr,
        GROUP_START: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets).to(tl.float32)

        group_max = tl.full((BLOCK,), -3.4028234663852886e38, tl.float32)
        for j in tl.static_range(0, GROUP_SIZE):
            score, _ = _branch_score_value(x, coeff_ptr, GROUP_START + j)
            group_max = tl.maximum(group_max, score)

        group_sum = tl.zeros((BLOCK,), tl.float32)
        group_num = tl.zeros((BLOCK,), tl.float32)
        for j in tl.static_range(0, GROUP_SIZE):
            score, value = _branch_score_value(x, coeff_ptr, GROUP_START + j)
            weight = tl.exp(score - group_max)
            group_sum += weight
            group_num += weight * value

        out_offsets = GROUP_ID * n_elements + offsets
        tl.store(group_max_ptr + out_offsets, group_max)
        tl.store(group_sum_ptr + out_offsets, group_sum)
        tl.store(group_num_ptr + out_offsets, group_num)


    @triton.jit
    def final_combine_kernel(
        group_max_ptr,
        group_sum_ptr,
        group_num_ptr,
        y_ptr,
        n_elements: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)

        global_max = tl.full((BLOCK,), -3.4028234663852886e38, tl.float32)
        for g in tl.static_range(0, NUM_GROUPS):
            m_g = tl.load(group_max_ptr + g * n_elements + offsets)
            global_max = tl.maximum(global_max, m_g)

        denominator = tl.zeros((BLOCK,), tl.float32)
        numerator = tl.zeros((BLOCK,), tl.float32)
        for g in tl.static_range(0, NUM_GROUPS):
            m_g = tl.load(group_max_ptr + g * n_elements + offsets)
            s_g = tl.load(group_sum_ptr + g * n_elements + offsets)
            n_g = tl.load(group_num_ptr + g * n_elements + offsets)
            scale = tl.exp(m_g - global_max)
            denominator += scale * s_g
            numerator += scale * n_g

        tl.store(y_ptr + offsets, numerator / denominator)


    @triton.jit
    def full_fusion_kernel(
        x_ptr,
        y_ptr,
        coeff_ptr,
        n_elements: tl.constexpr,
        BRANCHES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets).to(tl.float32)

        global_max = tl.full((BLOCK,), -3.4028234663852886e38, tl.float32)
        for j in tl.static_range(0, BRANCHES):
            score, _ = _branch_score_value(x, coeff_ptr, j)
            global_max = tl.maximum(global_max, score)

        denominator = tl.zeros((BLOCK,), tl.float32)
        numerator = tl.zeros((BLOCK,), tl.float32)
        for j in tl.static_range(0, BRANCHES):
            score, value = _branch_score_value(x, coeff_ptr, j)
            weight = tl.exp(score - global_max)
            denominator += weight
            numerator += weight * value

        tl.store(y_ptr + offsets, numerator / denominator)


    class FusionPlan:
        def __init__(self, x: torch.Tensor, coeffs: torch.Tensor, branches: int, fusion_group_size: int, block: int) -> None:
            if branches <= 0 or fusion_group_size <= 0:
                raise ValueError("branches and fusion_group_size must be positive")
            if branches % fusion_group_size != 0:
                raise ValueError("branches must be divisible by fusion_group_size")
            if x.numel() % block != 0:
                raise ValueError("x.numel() must be divisible by block because this kernel uses unmasked load/store")
            if coeffs.numel() != branches * 6:
                raise ValueError("coeffs should contain branches*6 values")
            if not x.is_contiguous() or not coeffs.is_contiguous():
                raise ValueError("x and coeffs must be contiguous")
            self.x = x
            self.coeffs = coeffs
            self.branches = branches
            self.fusion_group_size = fusion_group_size
            self.block = block
            self.n_elements = x.numel()
            self.num_groups = branches // fusion_group_size
            self.y = torch.empty_like(x)
            if fusion_group_size < branches:
                self.group_max = torch.empty((self.num_groups, self.n_elements), device=x.device, dtype=torch.float32)
                self.group_sum = torch.empty_like(self.group_max)
                self.group_num = torch.empty_like(self.group_max)
            else:
                self.group_max = self.group_sum = self.group_num = None

        @property
        def num_kernel_launches(self) -> int:
            return 1 if self.fusion_group_size == self.branches else self.num_groups + 1

        def run(self) -> torch.Tensor:
            grid = (triton.cdiv(self.n_elements, self.block),)
            if self.fusion_group_size == self.branches:
                full_fusion_kernel[grid](self.x, self.y, self.coeffs, self.n_elements, BRANCHES=self.branches, BLOCK=self.block)
                return self.y
            assert self.group_max is not None and self.group_sum is not None and self.group_num is not None
            for gid in range(self.num_groups):
                group_stats_kernel[grid](
                    self.x, self.group_max, self.group_sum, self.group_num, self.coeffs, self.n_elements,
                    GROUP_ID=gid, GROUP_START=gid * self.fusion_group_size,
                    GROUP_SIZE=self.fusion_group_size, BLOCK=self.block,
                )
            final_combine_kernel[grid](
                self.group_max, self.group_sum, self.group_num, self.y, self.n_elements,
                NUM_GROUPS=self.num_groups, BLOCK=self.block,
            )
            return self.y
    ''').lstrip()


def materialize(scenario: Scenario, out_dir: Path) -> None:
    write_common_project_files(out_dir, scenario, "branch_fan_in_aggregation")
    write_text(out_dir / "fusion_benchmark" / "torch_reference.py", torch_reference_py())
    write_text(out_dir / "fusion_benchmark" / "kernels.py", kernels_py())
