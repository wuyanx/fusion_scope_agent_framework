from __future__ import annotations

import torch


def make_input(n_elements: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    data = torch.randn(n_elements, generator=gen, dtype=torch.float32) * 0.5
    return data.to(device=device)


def make_coefficients(ops: int, device: torch.device) -> torch.Tensor:
    # Four coefficients per op: a,b,c,d.
    vals = []
    for i in range(ops):
        t = float(i + 1)
        vals.extend([
            0.82 + 0.002 * t,
            -0.11 + 0.001 * t,
            0.23 + 0.003 * t,
            0.07 - 0.001 * t,
        ])
    return torch.tensor(vals, device=device, dtype=torch.float32)


def reference_output(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    ops = coeffs.numel() // 4
    y = x
    for i in range(ops):
        a, b, c, d = coeffs[i * 4:i * 4 + 4]
        y = torch.tanh(a * y + b) + 0.125 * torch.sigmoid(c * y + d) + 0.03125 * y
    return y
