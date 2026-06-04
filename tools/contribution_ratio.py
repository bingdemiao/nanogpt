"""Measure how much PrismLinear's reconnection R actually shapes the output.

Forward (silu_gate mode):
  preact = A @ B.T                                # shape (T, d_ff)
  ar     = A @ R_combined                         # shape (T, d_ff_or_groups)
  gate   = SiLU(ar + ib)                          # per-group, broadcast to d_ff
  H      = preact * gate                          # element-wise

We log:
  - ||gate||_2, mean/median/var of gate values per layer
  - "gate utilization": fraction of gates with |gate| in [0.1, 0.9] (the
    informative middle band; outside this band SiLU is saturating)
  - "suppression": mean(|H|) / mean(|preact|) — if << 1, R is mostly
    attenuating; if ≈ 1, gating is near-identity (R does nothing useful)
  - histogram of gate values per layer

Usage::

    python tools/contribution_ratio.py \\
        --ckpt /path/to/ckpt.pt \\
        --dataset tinystories \\
        --n_batches 16 \\
        --output_dir tools/out_contribution_ratio/

The script registers a forward hook on every PrismLinear and runs a few
validation batches. Statistics are aggregated across batches.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

try:
    import cute_prism
except ImportError:
    cute_prism = None


def load_model(ckpt_path: str, device: str = "cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["model_args"])
    model = GPT(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["iter_num"]


def load_val_data(dataset: str):
    """Return (val_data_array_uint16, meta_dict_or_None)."""
    data_dir = os.path.join("data", dataset)
    val_bin = os.path.join(data_dir, "val.bin")
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta = None
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    return val, meta


def get_batch(val, batch_size, block_size, device):
    ix = torch.randint(len(val) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(val[i:i + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True)


def compute_ratios(model, sample_X, layer_outputs):
    """Compute per-layer contribution stats by re-running the prism forward
    with B and R separated. Uses the saved captured inputs."""
    target_cls = cute_prism.PrismLinear if cute_prism else None
    prism_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, target_cls)]

    stats = []
    for idx, (name, layer) in enumerate(prism_layers):
        A = layer_outputs.get(name)
        if A is None:
            continue
        A = A.reshape(-1, layer.in_features).float()

        # preact = A @ B.T
        B = layer.weight.detach().float()
        preact = A @ B.T   # (T, d_ff)

        # ar = A @ R for each group, get gating values
        # R has shape (n_groups * r, in_features) — block-diagonal applied per group
        # The gate uses (per-group) summation; replicate the cute_prism path:
        n_groups = layer.out_features // layer.group_size
        r = layer.reconn_sz
        n_blocks = layer.in_features // r
        R = layer.reconn.detach().float()
        R_blocks = R.reshape(n_groups, r, n_blocks, r).permute(0, 2, 1, 3).contiguous()
        A_blocks = A.reshape(-1, n_blocks, r)
        # AR shape: (T, n_groups, r)  via per-block matmul + sum over blocks
        AR = torch.einsum("tbi,gbij->tgj", A_blocks, R_blocks)
        # The "gate value" per group is broadcast across group_size output channels.
        # Use the squared L2 of AR-per-group as the scalar input to silu (matches
        # cute_prism's silu_gate semantics modulo sign; per-group is one number).
        gate_input = AR.reshape(AR.shape[0], n_groups, r).mean(dim=-1)   # (T, n_groups)
        gate = torch.sigmoid(gate_input) * gate_input   # silu = x * sigmoid(x)

        # Statistics
        gate_flat = gate.flatten()
        in_band = ((gate_flat > 0.1) & (gate_flat < 0.9)).float().mean().item()
        gate_mean = gate_flat.mean().item()
        gate_var = gate_flat.var().item()
        gate_abs = gate_flat.abs().mean().item()

        # H ≈ preact gated. preact shape (T, d_ff); group g maps to channels
        # [g*group_size : (g+1)*group_size]. The gate is per-group.
        gate_full = gate.repeat_interleave(layer.group_size, dim=-1)  # (T, d_ff)
        H = preact * gate_full
        suppression = (H.abs().mean() / preact.abs().mean().clamp_min(1e-8)).item()

        # Histogram of gate values
        hist, edges = np.histogram(gate_flat.cpu().numpy(), bins=20, range=(-0.5, 1.5))

        stats.append({
            "layer": idx,
            "name": name,
            "n_groups": n_groups,
            "gate_mean": round(gate_mean, 4),
            "gate_var": round(gate_var, 4),
            "gate_abs_mean": round(gate_abs, 4),
            "gate_in_band_frac": round(in_band, 4),   # frac of gates in [0.1, 0.9]
            "suppression": round(suppression, 4),     # |H|/|preact|
            "hist": hist.tolist(),
            "hist_edges": edges.tolist(),
        })
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="tinystories",
                    help="data/<dataset>/val.bin must exist")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="tools/out_contribution_ratio")
    args = ap.parse_args()

    if cute_prism is None:
        print("[fatal] cute_prism not importable in this env"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg, it = load_model(args.ckpt, args.device)
    print(f"loaded {args.ckpt} (iter {it}, mlp_type={cfg.mlp_type})")
    if cfg.mlp_type != "prism":
        print("[fatal] this script only meaningful for prism models"); sys.exit(1)

    val, _ = load_val_data(args.dataset)

    # Register pre-hooks to capture inputs to each PrismLinear
    captured = {}
    hooks = []
    target_cls = cute_prism.PrismLinear
    for n, m in model.named_modules():
        if isinstance(m, target_cls):
            def _hook(_mod, inputs, _n=n):
                captured[_n] = inputs[0].detach()
            hooks.append(m.register_forward_pre_hook(_hook))

    # Aggregate stats across batches
    layer_stats = []
    for b in range(args.n_batches):
        x = get_batch(val, args.batch_size, cfg.block_size, args.device)
        with torch.no_grad():
            model(x, None)
        stats = compute_ratios(model, x, captured)
        layer_stats.append(stats)
        captured.clear()
        print(f"  batch {b+1}/{args.n_batches} done")

    for h in hooks:
        h.remove()

    # Average across batches
    if not layer_stats:
        print("[fatal] no prism layers found"); sys.exit(1)
    n_layers = len(layer_stats[0])
    avg_rows = []
    for li in range(n_layers):
        per_layer = [batch[li] for batch in layer_stats]
        avg = {
            "layer": li,
            "n_groups": per_layer[0]["n_groups"],
            "gate_mean":         round(np.mean([s["gate_mean"]         for s in per_layer]), 4),
            "gate_var":          round(np.mean([s["gate_var"]          for s in per_layer]), 4),
            "gate_abs_mean":     round(np.mean([s["gate_abs_mean"]     for s in per_layer]), 4),
            "gate_in_band_frac": round(np.mean([s["gate_in_band_frac"] for s in per_layer]), 4),
            "suppression":       round(np.mean([s["suppression"]       for s in per_layer]), 4),
        }
        avg_rows.append(avg)

    # Print
    print()
    print(f"{'L':>2s} | {'n_groups':>8s} | {'gate_mean':>10s} | "
          f"{'gate_var':>9s} | {'|gate|':>7s} | {'in_band':>8s} | {'|H|/|preact|':>13s}")
    print("-" * 80)
    for r in avg_rows:
        print(f"{r['layer']:>2d} | {r['n_groups']:>8d} | "
              f"{r['gate_mean']:>10.4f} | {r['gate_var']:>9.4f} | "
              f"{r['gate_abs_mean']:>7.4f} | {r['gate_in_band_frac']:>7.1%} | "
              f"{r['suppression']:>13.4f}")
    print()
    print("Interpretation:")
    print("  gate_in_band_frac < 0.4 → many gates saturated (extremes), R is masking not gating")
    print("  suppression ≈ 1.0       → R has no effect")
    print("  suppression << 1.0      → R is attenuating but probably not informative")
    print("  good target: in_band > 0.6 and suppression ∈ [0.4, 0.7]")

    # Save
    csv_path = os.path.join(args.output_dir, "contribution_ratio.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
        w.writeheader()
        w.writerows(avg_rows)
    print(f"\nwrote {csv_path}")

    # Optional histogram plot — last batch only, per layer
    try:
        import matplotlib.pyplot as plt
        last = layer_stats[-1]
        n = len(last)
        cols = min(4, n); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for ax, s in zip(axes.flat, last):
            edges = s["hist_edges"]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
            ax.bar(centers, s["hist"], width=(edges[1] - edges[0]) * 0.9)
            ax.axvspan(0.1, 0.9, alpha=0.15, color="green", label="informative band")
            ax.set_title(f"L{s['layer']} gate dist")
            ax.set_xlabel("gate value")
            ax.set_ylabel("count")
            ax.legend(fontsize=7)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.tight_layout()
        plot_path = os.path.join(args.output_dir, "gate_histograms.png")
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"wrote {plot_path}")
    except ImportError:
        print("[skip] matplotlib not installed; skipping histogram plot")


if __name__ == "__main__":
    main()
