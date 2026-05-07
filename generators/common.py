from __future__ import annotations

from pathlib import Path
import shutil
from textwrap import dedent

from fusion_scope_core.file_utils import write_text
from fusion_scope_core.scenario import Scenario
from fusion_scope_core.scenario_status import write_generated_metadata


def device_py() -> str:
    return dedent('''
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
    ''').lstrip()


def plot_results_py() -> str:
    return dedent('''
    from __future__ import annotations

    import argparse
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt


    def build_arg_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Plot fusion scope benchmark results.")
        p.add_argument("--csv", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, default=Path("results"))
        p.add_argument("--x", default="fusion_group_size")
        p.add_argument("--y", default="execution_time_ms")
        return p


    def main() -> None:
        args = build_arg_parser().parse_args()
        df = pd.read_csv(args.csv).sort_values(args.x)
        args.out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(df[args.x], df[args.y], marker="o")
        plt.xlabel(args.x)
        plt.ylabel("execution time (ms)")
        plt.title("Fusion scope vs execution time")
        plt.grid(True, alpha=0.3)
        plt.savefig(args.out_dir / "fusion_scope_execution_time.png", dpi=200, bbox_inches="tight")
        plt.close()

        baseline = float(df.iloc[0][args.y])
        speedup = baseline / df[args.y]
        plt.figure()
        plt.plot(df[args.x], speedup, marker="o")
        plt.xlabel(args.x)
        plt.ylabel("speedup over smallest fusion scope")
        plt.title("Fusion scope vs speedup")
        plt.grid(True, alpha=0.3)
        plt.savefig(args.out_dir / "fusion_scope_speedup.png", dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[done] wrote plots to {args.out_dir}")


    if __name__ == "__main__":
        main()
    ''').lstrip()


def run_benchmark_py(kind: str) -> str:
    extra_arg = "--B" if kind == "branch_fan_in_aggregation" else "--ops"
    extra_name = "B" if kind == "branch_fan_in_aggregation" else "ops"
    extra_help = "Number of branches." if kind == "branch_fan_in_aggregation" else "Number of elementwise ops in the chain."
    output_default = "results/branch_fan_in_aggregation.csv" if kind == "branch_fan_in_aggregation" else "results/linear_elementwise_chain.csv"
    return dedent(f'''
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
        parser.add_argument("{extra_arg}", type=int, default=32, help="{extra_help}")
        parser.add_argument("--groups", type=parse_groups, default=parse_groups("1,2,4,8,16,32"))
        parser.add_argument("--block", type=int, default=256)
        parser.add_argument("--warmup", type=int, default=20)
        parser.add_argument("--repeat", type=int, default=100)
        parser.add_argument("--trials", type=int, default=5)
        parser.add_argument("--device", type=str, default="auto")
        parser.add_argument("--out", type=Path, default=Path("{output_default}"))
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--check-n", type=int, default=4096)
        parser.add_argument("--dry-run", action="store_true")
        return parser


    def validate_config(primary: int, groups: list[int], N: int, block: int) -> None:
        if N % block != 0:
            raise ValueError(
                f"N={{N}} must be divisible by block={{block}} for current Ascend Triton/CANN compatibility."
            )
        for g in groups:
            if g > primary:
                raise ValueError(f"fusion group size {{g}} cannot exceed {extra_name}={{primary}}")
            if primary % g != 0:
                raise ValueError(f"{extra_name}={{primary}} must be divisible by group size {{g}}")


    def optional_module_version(module_name: str) -> str:
        try:
            module = importlib.import_module(module_name)
            return str(getattr(module, "__version__", "imported"))
        except Exception as exc:
            return f"not available: {{exc}}"


    def run_correctness_check(device: torch.device, primary: int, groups: list[int], check_n: int, block: int) -> None:
        from fusion_benchmark.kernels import FusionPlan
        if check_n % block != 0:
            raise ValueError("check_n must be divisible by block")
        print(f"[check] running correctness check with N={{check_n}}, {extra_name}={{primary}}")
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
                f"[check] G={{g:<4d}} launches={{plan.num_kernel_launches:<3d}} "
                f"max_abs_err={{max_abs_err:.6e}} max_rel_err={{max_rel_err:.6e}}"
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
        return {{
            "fusion_group_size": group_size,
            "num_groups": primary // group_size,
            "num_kernel_launches": plan.num_kernel_launches,
            "execution_time_ms": median(trial_times),
            "execution_time_mean_ms": sum(trial_times) / len(trial_times),
            "execution_time_min_ms": min(trial_times),
            "execution_time_max_ms": max(trial_times),
        }}


    def main() -> None:
        args = build_arg_parser().parse_args()
        primary = getattr(args, "{extra_name}")
        validate_config(primary, args.groups, args.N, args.block)

        if args.dry_run:
            print("[env] torch:", optional_module_version("torch"))
            print("[env] torch_npu:", optional_module_version("torch_npu"))
            print("[env] triton:", optional_module_version("triton"))
            print("[env] device: not resolved in dry-run mode")
            print(f"[config] kind={kind}, N={{args.N}}, {extra_name}={{primary}}, groups={{args.groups}}, block={{args.block}}, warmup={{args.warmup}}, repeat={{args.repeat}}, trials={{args.trials}}")
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
        print(f"[config] kind={kind}, N={{args.N}}, {extra_name}={{primary}}, groups={{args.groups}}, block={{args.block}}, warmup={{args.warmup}}, repeat={{args.repeat}}, trials={{args.trials}}")

        if args.check:
            run_correctness_check(device, primary, args.groups, args.check_n, args.block)

        x = make_input(args.N, device=device, seed=0)
        coeffs = make_coefficients(primary, device=device)
        synchronize(device)

        rows = []
        for g in args.groups:
            print(f"[benchmark] running fusion_group_size={{g}}")
            row = benchmark_one(x, coeffs, primary, g, args.block, args.warmup, args.repeat, args.trials)
            row.update({{
                "scenario_kind": "{kind}",
                "hardware": "Ascend 910B",
                "device": str(device),
                "N": args.N,
                "problem_size": primary,
                "problem_size_name": "{extra_name}",
                "block": args.block,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "trials": args.trials,
            }})
            rows.append(row)
            print(f"[benchmark] G={{g:<4d}} launches={{row['num_kernel_launches']:<3d}} median_ms={{row['execution_time_ms']:.6f}}")

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
        print(f"[done] wrote {{args.out}}")


    if __name__ == "__main__":
        main()
    ''').lstrip()


def write_common_project_files(out_dir: Path, scenario: Scenario, kind: str) -> None:
    write_text(out_dir / "fusion_benchmark" / "device.py", device_py())
    write_text(out_dir / "fusion_benchmark" / "__init__.py", "\"\"\"Generated benchmark package.\"\"\"\n")
    write_text(out_dir / "run_benchmark.py", run_benchmark_py(kind))
    write_text(out_dir / "plot_results.py", plot_results_py())
    write_text(out_dir / "scenario.md", scenario.path.read_text(encoding="utf-8"))
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    readme = dedent(f'''
    # Generated benchmark: {scenario.name}

    This directory was materialized from `{scenario.path.name}`.

    ## Dry run

    ```bash
    python run_benchmark.py --dry-run
    ```

    ## Correctness check and benchmark

    Use `tools/run_scenario.py` from the framework root, or run this generated project directly.
    Agent-managed runs write CSV and plots under `results/<timestamp>/`.
    The direct command is recorded in the parent framework output.
    ''').lstrip()
    write_text(out_dir / "README.md", readme)
    write_text(out_dir / "README_GENERATED.md", readme)
    write_generated_metadata(out_dir, scenario)
