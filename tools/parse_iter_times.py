"""Parse `iter N: loss ..., time XX.XXms, lr ...` lines from train.py
stdout and report median iter time per log file.

Usage::

    python tools/parse_iter_times.py logs/nanogpt_*.out

If invoked with no args, scans logs/ for everything.
"""

import argparse
import glob
import os
import re
import statistics
from collections import defaultdict


ITER_RE = re.compile(r"^iter (\d+): loss [\d.]+, time ([\d.]+)ms, lr [\d.eE+-]+")
OUT_RE = re.compile(r'(?:Overriding: out_dir =|^out_dir = "?)([^"\s]+)')
LR_RE = re.compile(r'(?:Overriding: learning_rate =|^learning_rate =)\s*([\d.eE+-]+)')


def extract(path):
    out_dir = None
    lr = None
    iter_times = []
    with open(path, errors="replace") as f:
        for line in f:
            m = OUT_RE.search(line)
            if m:
                out_dir = m.group(1).strip('"')
                continue
            m = LR_RE.search(line)
            if m:
                lr = m.group(1)
                continue
            m = ITER_RE.match(line)
            if m:
                iter_times.append(float(m.group(2)))
    return out_dir, lr, iter_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="log files; default = logs/nanogpt_*.out")
    args = ap.parse_args()

    paths = args.logs or sorted(glob.glob("logs/nanogpt_*.out"))
    paths = [p for p in paths if os.path.getsize(p) > 100]

    rows = []
    for p in paths:
        out_dir, lr, times = extract(p)
        if not times:
            continue
        # Skip first 100 iters (warmup, cute_prism kernel autotune)
        steady = times[100:] if len(times) > 100 else times
        if not steady:
            steady = times
        rows.append({
            "jobid": os.path.basename(p).split("_")[1].split(".")[0],
            "out_dir": out_dir or "?",
            "lr": lr or "?",
            "n_iters": len(times),
            "median_ms": statistics.median(steady),
            "p90_ms": sorted(steady)[int(len(steady) * 0.9)] if steady else 0,
        })

    # Group by config family for easier reading
    rows.sort(key=lambda r: (r["out_dir"] or "", r["lr"] or ""))

    header = (
        f"{'job':>8s} | {'out_dir':40s} | {'lr':>8s} | "
        f"{'iters':>6s} | {'median ms/iter':>14s} | {'p90 ms':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['jobid']:>8s} | {r['out_dir'][:40]:40s} | {r['lr']:>8s} | "
            f"{r['n_iters']:>6d} | {r['median_ms']:>12.1f}   | {r['p90_ms']:>6.1f}"
        )


if __name__ == "__main__":
    main()
