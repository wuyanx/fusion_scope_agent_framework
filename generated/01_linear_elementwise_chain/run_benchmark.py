from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from statistics import median


def parse_groups(text: str) -> list[int]:
    groups = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("groups must not be empty")
    if any(g <= 0 for g in groups):
        raise argparse.ArgumentTypeError("all group sizes must be positive")
    return groups


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fusion scope vs execution time benchmark.")
    parser.add_argument("--N", type=int, default=4_194_304, help="Number of elements in x.")
    parser.add_argument("--ops", type=int, default=32, help="Number of elementwise ops in the chain.")
    parser.add_argument("--groups", type=parse_groups, default=parse_groups("1,2,4,8,16,32"))
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=Path, default=Path("results/linear_elementwise_chain.csv"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-n", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_config(primary: int, groups: list[int], N: int, block: int) -> None:
    if N % block != 0:
        raise ValueError(
            f"N={N} must be divisible by block={block} for current Ascend Triton/CANN compatibility."
        )
    for g in groups:
        if g > primary:
            raise ValueError(f"fusion group size {g} cannot exceed ops={primary}")
        if primary % g != 0:
            raise ValueError(f"ops={primary} must be divisible by group size {g}")


def optional_module_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "imported"))
    except Exception as exc:
        return f"not available: {exc}"


def run_correctness_check(device: torch.device, primary: int, groups: list[int], check_n: int, block: int) -> None:
    from fusion_benchmark.kernels import FusionPlan
    if check_n % block != 0:
        raise ValueError("check_n must be divisible by block")
    print(f"[check] running correctness check with N={check_n}, ops={primary}")
    x = make_input(check_n, device=device, seed=123)
    coeffs = make_coefficients(primary, device=device)
    expected = reference_output(x, coeffs)
    synchronize(device)
    for g in groups:
        plan = FusionPlan(x, coeffs, primary, g, block)
        got = plan.run()
        synchronize(device)
        max_abs_err = torch.max(torch.abs(got - expected)).item()
        max_rel_err = torch.max(torch.abs(got - expected) / torch.clamp(torch.abs(expected), min=1e-6)).item()
        print(
            f"[check] G={g:<4d} launches={plan.num_kernel_launches:<3d} "
            f"max_abs_err={max_abs_err:.6e} max_rel_err={max_rel_err:.6e}"
        )
        torch.testing.assert_close(got, expected, rtol=3e-4, atol=3e-4)
    print("[check] all group sizes passed correctness check")


def benchmark_one(x, coeffs, primary: int, group_size: int, block: int, warmup: int, repeat: int, trials: int):
    from fusion_benchmark.kernels import FusionPlan
    device = x.device
    plan = FusionPlan(x, coeffs, primary, group_size, block)
    for _ in range(warmup):
        plan.run()
    synchronize(device)
    trial_times = [time_repeated(plan.run, device=device, repeat=repeat) for _ in range(trials)]
    return {
        "fusion_group_size": group_size,
        "num_groups": primary // group_size,
        "num_kernel_launches": plan.num_kernel_launches,
        "execution_time_ms": median(trial_times),
        "execution_time_mean_ms": sum(trial_times) / len(trial_times),
        "execution_time_min_ms": min(trial_times),
        "execution_time_max_ms": max(trial_times),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    primary = getattr(args, "ops")
    validate_config(primary, args.groups, args.N, args.block)

    if args.dry_run:
        print("[env] torch:", optional_module_version("torch"))
        print("[env] torch_npu:", optional_module_version("torch_npu"))
        print("[env] triton:", optional_module_version("triton"))
        print("[env] device: not resolved in dry-run mode")
        print(f"[config] kind=linear_elementwise_chain, N={args.N}, ops={primary}, groups={args.groups}, block={args.block}, warmup={args.warmup}, repeat={args.repeat}, trials={args.trials}")
        return

    global torch, collect_runtime_info, synchronize, time_repeated, make_coefficients, make_input, reference_output
    import torch
    from fusion_benchmark.device import collect_runtime_info, synchronize, time_repeated
    from fusion_benchmark.torch_reference import make_coefficients, make_input, reference_output

    print("[env] torch:", torch.__version__)
    runtime_info = collect_runtime_info(args.device)
    device = runtime_info.device
    print("[env] torch_npu:", runtime_info.torch_npu_version)
    print("[env] triton:", runtime_info.triton_version)
    print("[env] device:", device)
    print(f"[config] kind=linear_elementwise_chain, N={args.N}, ops={primary}, groups={args.groups}, block={args.block}, warmup={args.warmup}, repeat={args.repeat}, trials={args.trials}")

    if args.check:
        run_correctness_check(device, primary, args.groups, args.check_n, args.block)

    x = make_input(args.N, device=device, seed=0)
    coeffs = make_coefficients(primary, device=device)
    synchronize(device)

    rows = []
    for g in args.groups:
        print(f"[benchmark] running fusion_group_size={g}")
        row = benchmark_one(x, coeffs, primary, g, args.block, args.warmup, args.repeat, args.trials)
        row.update({
            "scenario_kind": "linear_elementwise_chain",
            "hardware": "Ascend 910B",
            "device": str(device),
            "N": args.N,
            "problem_size": primary,
            "problem_size_name": "ops",
            "block": args.block,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "trials": args.trials,
        })
        rows.append(row)
        print(f"[benchmark] G={g:<4d} launches={row['num_kernel_launches']:<3d} median_ms={row['execution_time_ms']:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_kind", "hardware", "device", "N", "problem_size_name", "problem_size", "block",
        "warmup", "repeat", "trials", "fusion_group_size", "num_groups", "num_kernel_launches",
        "execution_time_ms", "execution_time_mean_ms", "execution_time_min_ms", "execution_time_max_ms",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
