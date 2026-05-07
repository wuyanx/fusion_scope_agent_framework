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
