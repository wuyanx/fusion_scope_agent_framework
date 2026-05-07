from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_scope_core.scenario import comma_join, load_scenario
from fusion_scope_core.file_utils import reset_dir
from fusion_scope_core.scenario_status import classify_scenario
from generators import REGISTRY


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Materialize and optionally run one fusion-scope scenario.")
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("generated"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Materialize and run generated benchmark in --dry-run mode only.")
    p.add_argument("--check", action="store_true", help="Run correctness check before benchmark.")
    p.add_argument("--run", action="store_true", help="Run full benchmark.")
    p.add_argument("--plot", action="store_true", help="Plot output CSV after full benchmark.")
    p.add_argument("--python", default=sys.executable, help="Python executable for generated benchmark.")
    p.add_argument("--device", default=None, help="Device override, e.g. npu:0.")
    p.add_argument("--run-id", default=None, help="Result subdirectory name. Defaults to a timestamp when --run is used.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args appended to generated run_benchmark.py")
    return p


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def output_csv_for_run(scenario, run_id: str | None) -> Path:
    cfg = scenario.config
    configured = Path(cfg.get("output_csv", "results/result.csv"))
    if not run_id:
        return configured
    result_root = configured.parent if configured.parent != Path("") else Path("results")
    return result_root / run_id / configured.name


def build_generated_command(
    scenario,
    generated_dir: Path,
    dry_run: bool,
    check: bool,
    run: bool,
    device: str | None,
    extra: list[str] | None,
    output_csv: Path | None = None,
) -> list[str]:
    cfg = scenario.config
    out = output_csv or output_csv_for_run(scenario, None)
    cmd = [
        str(generated_dir / "run_benchmark.py"),
        "--N", str(cfg["default_N"]),
        "--groups", comma_join(cfg["fusion_scopes"]),
        "--block", str(cfg["block"]),
        "--warmup", str(cfg["warmup"]),
        "--repeat", str(cfg["repeat"]),
        "--trials", str(cfg["trials"]),
        "--out", str(out),
    ]
    if scenario.benchmark_kind == "branch_fan_in_aggregation":
        cmd.extend(["--B", str(cfg.get("default_B", cfg.get("problem_size", 32)))])
    elif scenario.benchmark_kind == "linear_elementwise_chain":
        cmd.extend(["--ops", str(cfg.get("default_ops", cfg.get("problem_size", 32)))])
    else:
        raise ValueError(f"Unsupported benchmark_kind={scenario.benchmark_kind}")
    if dry_run:
        cmd.append("--dry-run")
    if check:
        cmd.extend(["--check", "--check-n", str(cfg.get("check_n", 4096))])
    if device:
        cmd.extend(["--device", device])
    if extra:
        cmd.extend(extra)
    if not dry_run and not check and not run:
        # For safety, if no execution flag is provided, only dry-run.
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario = load_scenario(args.scenario)
    generated_dir = (args.out_root / scenario.scenario_id).resolve()
    run_id = args.run_id or (default_run_id() if args.run else None)
    output_csv = output_csv_for_run(scenario, run_id)
    if run_id:
        print(f"[run] run_id={run_id}", flush=True)
        print(f"[run] output_csv={output_csv}", flush=True)
    status = classify_scenario(scenario, generated_dir, REGISTRY.keys())
    print(f"[scenario] state={status.state}", flush=True)
    print(f"[scenario] action={status.message}", flush=True)
    if scenario.benchmark_kind not in REGISTRY:
        raise ValueError(f"Unknown benchmark_kind={scenario.benchmark_kind}. Available: {sorted(REGISTRY)}")

    reset_dir(generated_dir, overwrite=args.overwrite)
    REGISTRY[scenario.benchmark_kind](scenario, generated_dir)
    print(f"[done] materialized scenario {scenario.scenario_id} -> {generated_dir}", flush=True)

    should_execute = args.dry_run or args.check or args.run
    if should_execute:
        cmd = [args.python] + build_generated_command(
            scenario=scenario,
            generated_dir=generated_dir,
            dry_run=args.dry_run and not args.run,
            check=args.check,
            run=args.run,
            device=args.device,
            extra=args.extra,
            output_csv=output_csv,
        )
        print("[exec]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=generated_dir, check=True)

    if args.plot:
        out_dir = output_csv.parent if output_csv.parent != Path("") else Path("results")
        cmd = [args.python, str(generated_dir / "plot_results.py"), "--csv", str(output_csv), "--out-dir", str(out_dir)]
        print("[exec]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=generated_dir, check=True)


if __name__ == "__main__":
    main()
