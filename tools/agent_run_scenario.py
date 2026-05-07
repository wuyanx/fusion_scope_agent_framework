from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_scope_core.file_utils import reset_dir
from fusion_scope_core.scenario import load_scenario
from fusion_scope_core.scenario_status import classify_scenario
from generators import REGISTRY
from tools.run_scenario import build_generated_command, default_run_id, output_csv_for_run


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Agent-facing scenario runner. It classifies whether generated scripts can be reused, "
            "must be regenerated, or need a new generator."
        )
    )
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("generated"))
    p.add_argument("--status-only", action="store_true")
    p.add_argument("--force-materialize", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default=None)
    p.add_argument("--run-id", default=None, help="Result subdirectory name. Defaults to a timestamp when --run is used.")
    p.add_argument("--extra", nargs=argparse.REMAINDER)
    return p


def print_status(status) -> None:
    print(f"[scenario] id={status.scenario_id}", flush=True)
    print(f"[scenario] benchmark_kind={status.benchmark_kind}", flush=True)
    print(f"[scenario] state={status.state}", flush=True)
    print(f"[scenario] generated_dir={status.generated_dir}", flush=True)
    print(f"[scenario] generator_available={status.generator_available}", flush=True)
    print(f"[scenario] needs_generator={status.needs_generator}", flush=True)
    print(f"[scenario] needs_materialize={status.needs_materialize}", flush=True)
    if status.recorded_sha256:
        print(f"[scenario] recorded_sha256={status.recorded_sha256}", flush=True)
    print(f"[scenario] current_sha256={status.scenario_sha256}", flush=True)
    if status.missing_generated_paths:
        print("[scenario] missing_generated_paths=" + ",".join(status.missing_generated_paths), flush=True)
    print(f"[scenario] action={status.message}", flush=True)


def run_generated(args, scenario, generated_dir: Path, output_csv: Path) -> None:
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
    print_status(status)

    if args.status_only:
        return

    if status.needs_generator:
        generator_path = Path("generators") / f"{scenario.benchmark_kind}.py"
        raise SystemExit(
            "New benchmark_kind detected. The agent must create "
            f"{generator_path}, register it in generators/__init__.py, and rerun this command."
        )

    if status.needs_materialize or args.force_materialize:
        reset_dir(generated_dir, overwrite=True)
        REGISTRY[scenario.benchmark_kind](scenario, generated_dir)
        print(f"[done] materialized scenario {scenario.scenario_id} -> {generated_dir}", flush=True)
    else:
        print(f"[done] reusing generated benchmark -> {generated_dir}", flush=True)

    run_generated(args, scenario, generated_dir, output_csv)


if __name__ == "__main__":
    main()
