"""Per-config FLOPs counter for the param-matched ablation.

Uses torch.utils.flop_counter.FlopCounterMode on a single forward pass at
the config's batch_size × block_size. Reports total FLOPs plus the
per-component breakdown so the MLP delta vs vanilla is visible.

The cute_prism::forward custom op is opaque to the dispatcher, so its
FLOPs do NOT show up automatically. We add them via an explicit fallback:
PrismLinear's effective FLOPs = (in × out × n_groups_or_seq_terms);
specifically:

  up.weight @ x      : B*T*in*out                  (this IS captured)
  reconn   * x       : B*T*n_groups*r*in           (NOT captured — added)
  silu_gate elementwise: skipped (negligible)

We use B=1, T=block_size for fair per-token reporting, then multiply by
the effective batch tokens (bs * accum * seq) for per-iter numbers.
"""

import os
import sys

# Allow `python tools/count_flops.py` from the repo root by ensuring the
# parent dir (where model.py lives) is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.flop_counter import FlopCounterMode

from model import GPT, GPTConfig


def base_cfg(scale: str):
    if scale == "tinystories":
        return dict(
            n_layer=8, n_head=8, n_embd=512, block_size=512,
            vocab_size=50304, bias=False, dropout=0.0,
            group_size=64, reconn_sz=8,
        )
    if scale == "gpt2small":
        return dict(
            n_layer=12, n_head=12, n_embd=768, block_size=1024,
            vocab_size=50304, bias=False, dropout=0.0,
            group_size=64, reconn_sz=8,
        )
    raise ValueError(scale)


def prism_extra_flops_per_token(cfg: GPTConfig) -> int:
    """FLOPs that the FlopCounter misses because cute_prism::forward is an
    opaque custom op. We add back both matmuls inside the kernel:

      1. preact = x @ B.T     : 2 * out * in  = 2 * d_ff * d  per token
      2. ar     = x @ R.T     : 2 * (n_groups*r) * in  per token

    Plus the silu_gate elementwise (negligible, skipped).
    """
    if cfg.mlp_type != "prism":
        return 0
    d = cfg.n_embd
    d_ff = cfg.mlp_expansion * d
    n_groups = d_ff // cfg.group_size
    r = cfg.reconn_sz
    b_matmul   = 2 * d_ff * d                # the B (up.weight) matmul
    r_matmul   = 2 * n_groups * r * d        # the reconnection matmul
    per_layer = b_matmul + r_matmul
    return per_layer * cfg.n_layer


def main():
    configs = [
        # (label, scale, overrides)
        ("vanilla 4× tinystories", "tinystories", dict(mlp_type="vanilla", mlp_expansion=4)),
        ("prism   4× tinystories", "tinystories", dict(mlp_type="prism",   mlp_expansion=4)),
        ("prism   3× tinystories", "tinystories", dict(mlp_type="prism",   mlp_expansion=3)),
        ("prism   2× tinystories", "tinystories", dict(mlp_type="prism",   mlp_expansion=2)),
        ("vanilla 4× gpt2small",   "gpt2small",   dict(mlp_type="vanilla", mlp_expansion=4)),
        ("prism   4× gpt2small",   "gpt2small",   dict(mlp_type="prism",   mlp_expansion=4)),
        ("prism   3× gpt2small",   "gpt2small",   dict(mlp_type="prism",   mlp_expansion=3)),
        ("prism   2× gpt2small",   "gpt2small",   dict(mlp_type="prism",   mlp_expansion=2)),
    ]

    rows = []
    ref_per_scale = {}
    for label, scale, overrides in configs:
        kw = dict(base_cfg(scale), **overrides)
        cfg = GPTConfig(**kw)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = GPT(cfg).to(device=device, dtype=dtype).eval()
        idx = torch.zeros((1, cfg.block_size), dtype=torch.long, device=device)
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(idx, None)
        flops_captured = fc.get_total_flops()
        flops_extra = prism_extra_flops_per_token(cfg) * cfg.block_size
        flops_total = flops_captured + flops_extra
        params = sum(p.numel() for p in model.parameters())
        rows.append((label, scale, params, flops_captured, flops_extra, flops_total))
        if scale not in ref_per_scale and overrides["mlp_type"] == "vanilla":
            ref_per_scale[scale] = flops_total
        del model, idx
        if device == "cuda":
            torch.cuda.empty_cache()

    header = (
        f"{'config':30s} | {'params':>10s} | {'fwd FLOPs (cap)':>16s} | "
        f"{'+prism kernel':>14s} | {'total':>14s} | {'vs vanilla':>10s}"
    )
    print(header)
    print("-" * len(header))
    for label, scale, params, cap, extra, total in rows:
        ref = ref_per_scale[scale]
        delta = (total - ref) / ref * 100.0
        print(
            f"{label:30s} | {params/1e6:>8.2f}M | {cap/1e9:>14.3f}G | "
            f"{extra/1e9:>12.3f}G | {total/1e9:>12.3f}G | {delta:>+8.2f}%"
        )

    print()
    print("Notes:")
    print("- 'fwd FLOPs (cap)' = what FlopCounterMode tracks (most matmuls + softmax).")
    print("- '+prism kernel' = the B@x and R@x matmuls inside cute_prism::forward")
    print("  that the dispatcher does NOT see; added analytically.")
    print("- Numbers are PER ONE forward pass over (B=1, T=block_size).")
    print("- Multiply by gradient_accumulation_steps * batch_size for per-iter FLOPs,")
    print("  and by ~3 for forward+backward (≈ 2x backward + 1x forward).")


if __name__ == "__main__":
    main()
