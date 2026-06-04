"""SVD-based effective-rank analysis of MLP weight matrices.

For each PrismLinear / VanillaMLP in a checkpoint, compute:
  - singular value spectrum (top-k + cumulative)
  - effective rank: r_eff = exp(H(p)), where p_i = sigma_i / sum(sigma_j)
    (entropy-based; ranges 1..min(M,N); higher = more rank used)
  - 50% / 90% / 99% energy rank: smallest k where cumsum(p) ≥ threshold

For PrismLinear, three views are computed:
  - B alone (the up.weight) — what the dense projection's spectrum looks like
  - R structural bound — block-diagonal R has rank ≤ n_groups * reconn_sz
  - effective combined: B + Cayley(R) full materialization (only meaningful
    when prism is in "gated" mode does this approximation hold; we mainly
    report B's spectrum since that's the trainable matrix)

Vanilla c_fc + c_proj are SVDed directly.

Usage::

    python tools/effective_rank.py \\
        --vanilla_ckpt /capstor/scratch/.../out-tinystories-vanilla/ckpt.pt \\
        --prism_ckpt   /capstor/scratch/.../out-tinystories-prism-3x-lr1.5e-3-r0.5/ckpt.pt \\
        --output_dir   /users/lshuhao/nanogpt/tools/out_effective_rank/

Outputs in --output_dir:
  spectrum_<run>.csv        # singular values per layer
  effective_rank_summary.csv # one row per (run, layer, matrix) with summary stats
  spectrum_comparison.png   # log-scaled spectrum plot (vanilla vs prism)
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["model_args"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["iter_num"]


def entropy_effective_rank(s: torch.Tensor) -> float:
    """r_eff = exp(-sum p_i log p_i) where p_i = s_i / sum(s)."""
    s = s.float()
    s = s[s > 0]
    p = s / s.sum()
    h = -(p * p.log()).sum().item()
    return math.exp(h)


def energy_rank(s: torch.Tensor, fractions=(0.5, 0.9, 0.99)) -> dict:
    """Smallest k s.t. cumsum(s) / sum(s) >= fraction."""
    s = s.float()
    total = s.sum()
    cum = torch.cumsum(s, 0) / total
    out = {}
    for frac in fractions:
        idxs = torch.nonzero(cum >= frac, as_tuple=True)[0]
        out[f"rank_{int(100*frac)}pct"] = int(idxs[0].item()) + 1 if idxs.numel() else len(s)
    return out


def collect_mlp_matrices(model, label: str):
    """Yield (layer_idx, matrix_name, weight_tensor) for every MLP weight."""
    for li, block in enumerate(model.transformer.h):
        mlp = block.mlp
        if mlp.__class__.__name__ == "VanillaMLP":
            yield li, "c_fc",   mlp.c_fc.weight.detach()
            yield li, "c_proj", mlp.c_proj.weight.detach()
        elif mlp.__class__.__name__ == "PrismMLP":
            # PrismLinear stores B at .weight
            yield li, "up.B",   mlp.up.weight.detach()
            yield li, "up.R",   mlp.up.reconn.detach()
            yield li, "down",   mlp.down.weight.detach()
        else:
            print(f"[warn] {label} block {li}: unknown MLP class "
                  f"{mlp.__class__.__name__}")


def analyse(model, run_label: str, out_dir: str):
    rows = []
    spectra_rows = []
    for li, mname, W in collect_mlp_matrices(model, run_label):
        # SVD on a 2D matrix
        if W.dim() == 1:
            continue
        W = W.float()
        if W.dim() > 2:  # e.g. reconn shape (n_groups*r, in_features)
            W = W.reshape(W.shape[0], -1)
        # full_matrices=False keeps min(M,N) singular values
        s = torch.linalg.svdvals(W)
        r_eff = entropy_effective_rank(s)
        e_ranks = energy_rank(s)
        rows.append({
            "run": run_label,
            "layer": li,
            "matrix": mname,
            "shape": tuple(W.shape),
            "min_dim": min(W.shape),
            "effective_rank": round(r_eff, 2),
            "effective_rank_pct": round(100 * r_eff / min(W.shape), 1),
            **e_ranks,
            "sigma_max": round(s[0].item(), 4),
            "sigma_min": round(s[-1].item(), 6),
            "condition_number": round((s[0] / s[-1].clamp_min(1e-12)).item(), 2),
        })
        for k, sv in enumerate(s.tolist()):
            spectra_rows.append({"run": run_label, "layer": li, "matrix": mname,
                                 "k": k + 1, "sigma": sv})
    return rows, spectra_rows


def maybe_plot(spectra_rows, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed; skipping plot")
        return
    # One panel per (layer, matrix), grouped by run
    by_key = {}
    for r in spectra_rows:
        by_key.setdefault((r["layer"], r["matrix"]), {}).setdefault(r["run"], []).append(r["sigma"])
    n = len(by_key)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, ((li, mname), runs) in zip(axes.flat, sorted(by_key.items())):
        for run_label, svs in runs.items():
            ax.semilogy(range(1, len(svs) + 1), svs, label=run_label, lw=1.2)
        ax.set_title(f"L{li} {mname}", fontsize=9)
        ax.set_xlabel("k")
        ax.set_ylabel("σ_k")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(by_key):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  spectrum plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", nargs=2, metavar=("LABEL", "PATH"),
                    help="May be repeated: --ckpt vanilla /path/to/ckpt.pt --ckpt prism /other/ckpt.pt")
    ap.add_argument("--vanilla_ckpt")
    ap.add_argument("--prism_ckpt")
    ap.add_argument("--output_dir", default="tools/out_effective_rank")
    args = ap.parse_args()

    # Backward-compatible: positional flags
    runs = []
    if args.vanilla_ckpt: runs.append(("vanilla", args.vanilla_ckpt))
    if args.prism_ckpt:   runs.append(("prism",   args.prism_ckpt))
    for label, path in args.ckpt or []:
        runs.append((label, path))
    if not runs:
        ap.error("Provide at least one --ckpt LABEL PATH (or --vanilla_ckpt/--prism_ckpt)")

    os.makedirs(args.output_dir, exist_ok=True)

    all_summary = []
    all_spectra = []
    for label, path in runs:
        print(f"=== {label}: {path} ===")
        model, cfg, it = load_model(path)
        print(f"    iter={it}  mlp_type={cfg.mlp_type}  n_embd={cfg.n_embd}  n_layer={cfg.n_layer}")
        summary, spectra = analyse(model, label, args.output_dir)
        all_summary.extend(summary)
        all_spectra.extend(spectra)
        del model

    # Write CSVs
    sp_path = os.path.join(args.output_dir, "effective_rank_summary.csv")
    if all_summary:
        with open(sp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            w.writeheader()
            w.writerows(all_summary)
        print(f"summary -> {sp_path}")

    sp_path = os.path.join(args.output_dir, "spectra.csv")
    if all_spectra:
        with open(sp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_spectra[0].keys()))
            w.writeheader()
            w.writerows(all_spectra)
        print(f"spectra -> {sp_path}")

    # Print summary table
    print()
    print(f"{'run':>10s} | {'L':>2s} | {'matrix':>8s} | {'shape':>14s} | "
          f"{'r_eff':>8s} | {'r_eff%':>7s} | {'r50%':>5s} | {'r90%':>5s} | {'r99%':>5s} | "
          f"{'cond':>8s}")
    print("-" * 110)
    for r in all_summary:
        print(f"{r['run']:>10s} | {r['layer']:>2d} | {r['matrix']:>8s} | "
              f"{str(r['shape']):>14s} | "
              f"{r['effective_rank']:>8.1f} | {r['effective_rank_pct']:>6.1f}% | "
              f"{r['rank_50pct']:>5d} | {r['rank_90pct']:>5d} | {r['rank_99pct']:>5d} | "
              f"{r['condition_number']:>8.1f}")

    maybe_plot(all_spectra, os.path.join(args.output_dir, "spectrum_comparison.png"))


if __name__ == "__main__":
    main()
