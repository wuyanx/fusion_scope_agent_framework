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
