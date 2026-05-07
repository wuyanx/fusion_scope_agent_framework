from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("[exec]", " ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sync framework to remote Ascend server and run one scenario.")
    p.add_argument("--host", default="910B3", help="SSH host alias. Prefer using ~/.ssh/config.")
    p.add_argument("--remote-root", default="/home/wyx/fusion_scope_agent_framework")
    p.add_argument("--python", default="/root/miniconda3/envs/tlx/bin/python")
    p.add_argument("--scenario", required=True, help="Scenario path relative to project root, e.g. scenarios/01_linear_elementwise_chain.md")
    p.add_argument("--check", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--run-id", default=None, help="Result subdirectory name on the remote run.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    local_root = Path(__file__).resolve().parents[1]
    remote = f"{args.host}:{args.remote_root.rstrip('/')}/"
    run([
        "rsync", "-az", "--delete",
        "--exclude", "generated/", "--exclude", "results/", "--exclude", "__pycache__/",
        str(local_root) + "/", remote,
    ], dry_run=args.dry_run)

    flags = ["--scenario", shlex.quote(args.scenario), "--python", shlex.quote(args.python), "--device", shlex.quote(args.device)]
    if args.overwrite:
        flags.append("--overwrite")
    if args.check:
        flags.append("--check")
    if args.run:
        flags.append("--run")
    if args.plot:
        flags.append("--plot")
    if args.run_id:
        flags.extend(["--run-id", shlex.quote(args.run_id)])
    if not (args.check or args.run):
        flags.append("--dry-run")

    remote_cmd = f"cd {shlex.quote(args.remote_root)} && {shlex.quote(args.python)} tools/run_scenario.py " + " ".join(flags)
    run(["ssh", args.host, remote_cmd], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
