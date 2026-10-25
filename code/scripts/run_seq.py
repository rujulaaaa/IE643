"""Run several repository scripts one after another on the same GPU (one Kaggle job per GPU).

Each --cmd is "name=script.py args..."; it gets --out <out>/<name> appended. A failing
command is reported, leaves <out>/<name>.FAILED, and the next one still runs; the exit code is
non-zero if any failed.

    python scripts/run_seq.py --out /kaggle/working/out/job0 \
        --cmd "dev=scripts/edit_sweep.py --split dev --arms coarse --shard 0/2" \
        --cmd "lora=scripts/train_motion_lora.py --variant full"
"""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    codes = []
    for c in args.cmd:
        name, line = c.split("=", 1)
        argv = [sys.executable, "-u", *shlex.split(line), "--out", str(args.out / name)]
        print(f"=== {name}: {' '.join(argv)}", flush=True)
        t0 = time.perf_counter()
        codes.append(subprocess.run(argv).returncode)
        if codes[-1]:   # marker for jobs waiting on this one's outputs (edit_sweep --lora-wait)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.FAILED").write_text(str(codes[-1]))
        print(f"=== {name}: exit {codes[-1]} after {time.perf_counter() - t0:.0f} s", flush=True)
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
