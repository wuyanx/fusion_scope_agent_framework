from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description="Generic plotter for materialized benchmark CSV files.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args()
    df = pd.read_csv(args.csv).sort_values("fusion_group_size")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["fusion_group_size"], df["execution_time_ms"], marker="o")
    plt.xlabel("fusion_group_size")
    plt.ylabel("execution time (ms)")
    plt.grid(True, alpha=0.3)
    plt.title("Fusion scope vs execution time")
    plt.savefig(args.out_dir / "fusion_scope_execution_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    baseline = float(df.iloc[0]["execution_time_ms"])
    plt.figure()
    plt.plot(df["fusion_group_size"], baseline / df["execution_time_ms"], marker="o")
    plt.xlabel("fusion_group_size")
    plt.ylabel("speedup over smallest scope")
    plt.grid(True, alpha=0.3)
    plt.title("Fusion scope vs speedup")
    plt.savefig(args.out_dir / "fusion_scope_speedup.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[done] wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
