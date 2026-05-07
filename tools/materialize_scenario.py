from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as `python tools/materialize_scenario.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion_scope_core.file_utils import reset_dir
from fusion_scope_core.scenario import load_scenario
from generators import REGISTRY


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Materialize a benchmark project from a scenario Markdown file.")
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("generated"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    scenario = load_scenario(args.scenario)
    if scenario.benchmark_kind not in REGISTRY:
        raise ValueError(f"Unknown benchmark_kind={scenario.benchmark_kind}. Available: {sorted(REGISTRY)}")
    out_dir = args.out_dir or (args.out_root / scenario.scenario_id)
    reset_dir(out_dir, overwrite=args.overwrite)
    REGISTRY[scenario.benchmark_kind](scenario, out_dir)
    print(f"[done] materialized scenario {scenario.scenario_id} -> {out_dir}")


if __name__ == "__main__":
    main()
