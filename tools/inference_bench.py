"""Inference-only latency benchmark for vanilla vs prism MLPs.

Measures forward-pass latency for a given checkpoint at varying batch
sizes and sequence lengths. No autoregressive generation — pure forward
pass through the transformer (the speed-critical kernel-bound part).

Reports:
  - median, p10, p90 latency per (config, batch_size, seq_len)
  - tokens/second throughput

When run on Clariden: prism uses cublas backend (the training-supported
one), vanilla uses standard nn.Linear. The wallclock gap measured here
is the cost prism pays for grouped factorization with the current
training kernel.

When run on Ault (with --backend cute), prism uses your optimized cute
kernel; this is the inference-side comparison that may flip the story.
Note: cute backend rejection of training-mode features may still apply;
this script runs forward-only with torch.no_grad() so all features are
inference-safe.

Usage::

    # Single checkpoint, sweep batch sizes
    python tools/inference_bench.py \\
        --ckpt /capstor/scratch/.../out-fineweb-vanilla/ckpt.pt \\
        --batch_sizes 1,4,16,32 \\
        --warmup 20 --iters 200

    # Compare two checkpoints
    python tools/inference_bench.py \\
        --ckpt vanilla:/path/A \\
        --ckpt prism:/path/B \\
        --batch_sizes 1,8,32

    # On Ault, force cute backend on a prism checkpoint
    python tools/inference_bench.py \\
        --ckpt prism_cute:/path/B --force_backend cute
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig


def load_ckpt(path: str, force_backend: str = None, device: str = "cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = dict(ck["model_args"])
    if force_backend and "prism_backend" in args:
        args["prism_backend"] = force_backend
    cfg = GPTConfig(**args)
    model = GPT(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def bench_one(model, batch_size: int, seq_len: int, warmup: int, iters: int,
              device: str = "cuda") -> dict:
    """Return latency stats (in ms) for batch_size × seq_len forward."""
    x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x, None)
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(iters):
            start.record()
            model(x, None)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

    times_sorted = sorted(times_ms)
    median = statistics.median(times_ms)
    p10 = times_sorted[int(0.10 * iters)]
    p90 = times_sorted[int(0.90 * iters)]
    tokens_per_iter = batch_size * seq_len
    tokens_per_sec_median = tokens_per_iter / (median / 1000.0)
    return dict(
        batch_size=batch_size, seq_len=seq_len,
        n_iters=iters,
        ms_median=round(median, 3),
        ms_p10=round(p10, 3),
        ms_p90=round(p90, 3),
        tokens_per_sec=int(tokens_per_sec_median),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                    help="LABEL:/path/to/ckpt.pt — may be repeated")
    ap.add_argument("--batch_sizes", default="1,4,16,32",
                    help="comma-separated list")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="defaults to model's block_size")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--force_backend", default=None,
                    choices=[None, "cublas", "cute", "pytorch"],
                    help="override the checkpoint's prism_backend (e.g. test cute on Ault)")
    args = ap.parse_args()

    if not args.ckpt:
        ap.error("Provide at least one --ckpt LABEL:/path/to/ckpt.pt")

    batches = [int(b) for b in args.batch_sizes.split(",")]
    rows = []

    for spec in args.ckpt:
        if ":" not in spec:
            ap.error(f"--ckpt arg {spec!r} must be LABEL:PATH")
        label, path = spec.split(":", 1)
        print(f"=== {label}: {path} ===")
        if args.force_backend:
            print(f"    forcing backend={args.force_backend}")
        model, cfg = load_ckpt(path, args.force_backend)
        seq_len = args.seq_len or cfg.block_size
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    n_params={n_params/1e6:.2f}M  n_embd={cfg.n_embd}  "
              f"n_layer={cfg.n_layer}  mlp_type={cfg.mlp_type}  "
              f"backend={getattr(cfg, 'prism_backend', 'n/a')}")
        for bs in batches:
            try:
                r = bench_one(model, bs, seq_len, args.warmup, args.iters)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"    bs={bs}: OOM, skipping")
                    continue
                raise
            r["label"] = label
            r["n_params_M"] = round(n_params / 1e6, 2)
            r["mlp_type"] = cfg.mlp_type
            r["backend"] = getattr(cfg, "prism_backend", "n/a")
            rows.append(r)
            print(f"    bs={bs:3d}  seq={seq_len}  median={r['ms_median']:>8.3f} ms  "
                  f"p10={r['ms_p10']:>8.3f}  p90={r['ms_p90']:>8.3f}  "
                  f"throughput={r['tokens_per_sec']:>10d} tok/s")
        del model
        torch.cuda.empty_cache()

    # Print comparison table
    print()
    print("=== summary ===")
    print(f"{'label':>14s} | {'mlp':>8s} | {'backend':>8s} | {'bs':>3s} | {'seq':>5s} | "
          f"{'ms_med':>8s} | {'tok/s':>10s} | {'vs first':>9s}")
    print("-" * 100)
    if rows:
        ref_by_bs = {r["batch_size"]: r for r in rows
                     if r["label"] == rows[0]["label"]}
        for r in rows:
            ref = ref_by_bs.get(r["batch_size"])
            speedup = ref["ms_median"] / r["ms_median"] if ref else 1.0
            print(f"{r['label']:>14s} | {r['mlp_type']:>8s} | {r['backend']:>8s} | "
                  f"{r['batch_size']:>3d} | {r['seq_len']:>5d} | "
                  f"{r['ms_median']:>8.3f} | {r['tokens_per_sec']:>10d} | "
                  f"{speedup:>9.2f}×")


if __name__ == "__main__":
    main()
