from __future__ import annotations

import argparse
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Small SSH helper for the 910B3 server.")
    parser.add_argument("--host", default="910B3", help="SSH host alias from ~/.ssh/config")
    parser.add_argument("--cmd", required=True, help="Command to run on the remote host")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cmd = ["ssh", args.host, args.cmd]
    print("[exec]", " ".join(shlex.quote(x) for x in cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
