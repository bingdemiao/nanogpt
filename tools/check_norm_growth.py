"""Diagnostic: where does the large AR variance come from?

We measured `ar_std ≈ 4` empirically on trained prism checkpoints, while
the design-time variance is `~1` (block-diagonal R with std 1/sqrt(r),
unit-variance A → AR.std ≈ 1). This script attributes the 4× excess.

For a given checkpoint, dumps per-layer:
  - LayerNorm γ stats (mean / max / min / std) — should be ~1 at init
  - A.std at the PrismLinear input — should be ~1 if LN works as designed
  - R.weight.std — should be ~1/sqrt(reconn_sz) ≈ 0.354 at init for
    r_init_scale=1.0, half that for r_init_scale=0.5
  - R growth ratio: current std / design init std
  - AR.std (recomputed live) — should match wandb's ar_std_mean panel
  - Decomposition: AR.std² ≈ block_size · A.var · R.var

Run::

    python tools/check_norm_growth.py \\
        --ckpt /capstor/scratch/.../out-tinystories-prism-3x-lr1.5e-3-r0.5/ckpt.pt \\
        --dataset tinystories --n_batches 4

Bottom-line table tells you whether the AR explosion is due to:
  (a) LN γ growing (A.std went up) — fix: constrain LN γ or normalize AR
  (b) R growing (R.std went up) — fix: r_wd or r_lr_mult tweak
  (c) both
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from model import GPT, GPTConfig

try:
    import cute_prism
except ImportError:
    cute_prism = None


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["model_args"])
    model = GPT(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["iter_num"]


def load_val_data(dataset):
    val_bin = os.path.join("data", dataset, "val.bin")
    return np.memmap(val_bin, dtype=np.uint16, mode="r")


def get_batch(val, bs, block_size, device):
    ix = torch.randint(len(val) - block_size, (bs,))
    x = torch.stack([torch.from_numpy(val[i:i + block_size].astype(np.int64))
                     for i in ix])
    return x.to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="tinystories")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if cute_prism is None:
        print("[fatal] cute_prism not importable"); sys.exit(1)

    model, cfg, it = load_model(args.ckpt, args.device)
    print(f"loaded {args.ckpt} (iter {it})")
    print(f"  mlp_type={cfg.mlp_type}, n_embd={cfg.n_embd}, n_layer={cfg.n_layer}, "
          f"reconn_sz={cfg.reconn_sz}, group_size={cfg.group_size}, "
          f"r_init_scale={cfg.r_init_scale}")
    print()

    if cfg.mlp_type != "prism":
        print("[fatal] only meaningful for prism models"); sys.exit(1)

    # ----------------- 1. LayerNorm γ stats -----------------
    print("=== LayerNorm γ statistics (init=1.0 for all) ===")
    print(f"{'name':>16s} | {'mean':>7s} | {'std':>7s} | {'min':>7s} | {'max':>7s}")
    print("-" * 60)
    for li, block in enumerate(model.transformer.h):
        for ln_name in ("ln_1", "ln_2"):
            ln = getattr(block, ln_name)
            g = ln.weight.detach().float()
            print(f"  L{li}.{ln_name:>5s} | {g.mean().item():>7.4f} | "
                  f"{g.std().item():>7.4f} | {g.min().item():>7.4f} | "
                  f"{g.max().item():>7.4f}")
    g = model.transformer.ln_f.weight.detach().float()
    print(f"  {'ln_f':>16s} | {g.mean().item():>7.4f} | "
          f"{g.std().item():>7.4f} | {g.min().item():>7.4f} | "
          f"{g.max().item():>7.4f}")
    print()

    # ----------------- 2. R weight statistics -----------------
    print("=== R (reconn) weight statistics ===")
    r = cfg.reconn_sz
    design_init_std = cfg.r_init_scale / math.sqrt(r)
    print(f"design init std = r_init_scale / sqrt(r) = "
          f"{cfg.r_init_scale} / sqrt({r}) = {design_init_std:.4f}")
    print()
    print(f"{'layer':>6s} | {'R.std':>7s} | {'R.std/init':>10s} | "
          f"{'R.abs.mean':>10s} | {'frobenius':>10s}")
    print("-" * 60)
    r_stds = []
    for li, block in enumerate(model.transformer.h):
        R = block.mlp.up.reconn.detach().float()
        r_std = R.std().item()
        r_stds.append(r_std)
        ratio = r_std / design_init_std
        frob = R.norm().item()
        print(f"  L{li:>3d} | {r_std:>7.4f} | {ratio:>9.2f}× | "
              f"{R.abs().mean().item():>10.4f} | {frob:>10.2f}")
    print(f"  mean R.std = {np.mean(r_stds):.4f}, "
          f"growth factor vs design = {np.mean(r_stds)/design_init_std:.2f}×")
    print()

    # ----------------- 3. B weight statistics -----------------
    print("=== B (up.weight) statistics ===")
    # Design init for B in gated mode: std ≈ alpha/sqrt(K) = 1.66/sqrt(d)
    # but cute_prism's exact init is more nuanced; we just report the
    # current std and let you compare to typical nn.Linear (1/sqrt(d_in))
    print(f"reference: nn.Linear init std = 1/sqrt(in_features) = "
          f"{1/math.sqrt(cfg.n_embd):.4f}")
    print()
    print(f"{'layer':>6s} | {'B.std':>7s} | {'B.abs.mean':>10s} | {'frobenius':>10s}")
    print("-" * 50)
    for li, block in enumerate(model.transformer.h):
        B = block.mlp.up.weight.detach().float()
        print(f"  L{li:>3d} | {B.std().item():>7.4f} | "
              f"{B.abs().mean().item():>10.4f} | {B.norm().item():>10.2f}")
    print()

    # ----------------- 4. Live A.std and AR.std at PrismLinear inputs -----------------
    print("=== Live forward pass: A.std and AR.std at prism inputs ===")
    val = load_val_data(args.dataset)
    captured = {}
    hooks = []
    target_cls = cute_prism.PrismLinear
    for n, m in model.named_modules():
        if isinstance(m, target_cls):
            def _hook(_mod, inputs, _n=n):
                captured.setdefault(_n, []).append(inputs[0].detach().float().cpu())
            hooks.append(m.register_forward_pre_hook(_hook))

    for _ in range(args.n_batches):
        x = get_batch(val, args.batch_size, cfg.block_size, args.device)
        with torch.no_grad():
            model(x, None)
    for h in hooks:
        h.remove()

    prism_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, target_cls)]

    print(f"{'layer':>6s} | {'A.std':>7s} | {'A.mean':>8s} | {'AR.std':>7s} | "
          f"{'pred AR.std':>11s} | {'AR.mean':>8s}")
    print("-" * 70)
    a_stds = []
    ar_stds = []
    for li, (n, m) in enumerate(prism_layers):
        if n not in captured:
            continue
        A_all = torch.cat(captured[n], dim=0).reshape(-1, m.in_features)
        a_std = A_all.std().item()
        a_mean = A_all.mean().item()
        a_stds.append(a_std)

        # Recompute AR
        nb = m.in_features // m.reconn_sz
        ng = m.out_features // m.group_size
        rr = m.reconn_sz
        R = m.reconn.detach().float().cpu()
        R_blocks = R.reshape(ng, rr, nb, rr).permute(0, 2, 1, 3).contiguous()
        A_blocks = A_all.reshape(-1, nb, rr)
        AR = torch.einsum("tbi,gbij->tgj", A_blocks, R_blocks)
        ar_std = AR.std().item()
        ar_mean = AR.mean().item()
        ar_stds.append(ar_std)

        # Predicted: var(AR) = r * var(A) * var(R)
        r_std = r_stds[li]
        pred_ar_std = math.sqrt(rr) * a_std * r_std

        print(f"  L{li:>3d} | {a_std:>7.4f} | {a_mean:>8.4f} | {ar_std:>7.4f} | "
              f"{pred_ar_std:>11.4f} | {ar_mean:>8.4f}")
    print()

    # ----------------- 5. Attribution summary -----------------
    print("=" * 75)
    print("ATTRIBUTION OF AR.std EXCESS (target ≈ 1.0, observed ≈ 4.0)")
    print("=" * 75)
    a_mean_std = float(np.mean(a_stds))
    r_mean_std = float(np.mean(r_stds))
    ar_mean_std = float(np.mean(ar_stds))
    design_a_std = 1.0
    design_r_std = design_init_std
    design_ar_std = math.sqrt(r) * design_a_std * design_r_std

    print(f"  A.std        : design 1.000  →  measured {a_mean_std:.3f}   "
          f"({a_mean_std/design_a_std:.2f}× excess)")
    print(f"  R.std        : design {design_r_std:.3f}  →  measured {r_mean_std:.3f}   "
          f"({r_mean_std/design_r_std:.2f}× excess)")
    print(f"  AR.std       : design {design_ar_std:.3f}  →  measured {ar_mean_std:.3f}   "
          f"({ar_mean_std/design_ar_std:.2f}× excess)")
    print()
    print("Decomposition of AR.std excess:")
    a_contrib = a_mean_std / design_a_std
    r_contrib = r_mean_std / design_r_std
    print(f"  from A: {a_contrib:.2f}×")
    print(f"  from R: {r_contrib:.2f}×")
    print(f"  product (predicted total): {a_contrib * r_contrib:.2f}×")
    print(f"  observed total:           {ar_mean_std / design_ar_std:.2f}×")
    print()
    if a_contrib > 1.5 and r_contrib > 1.5:
        cause = "BOTH A and R have grown (LN γ drift + R growth). " \
                "Two-pronged fix needed: constrain LN γ + lower r_wd."
    elif a_contrib > 1.5:
        cause = "Primarily A (LN γ drift). Fix: constrain LN γ, or normalize " \
                "AR before SiLU (add ar_norm), or learned ar_scale."
    elif r_contrib > 1.5:
        cause = "Primarily R growth. Fix: smaller r_init_scale, lower " \
                "r_lr_mult, or higher r_wd_factor."
    else:
        cause = "Neither A nor R grew much — variance excess source unclear; " \
                "check correlation effects."
    print(f"Diagnosis: {cause}")


if __name__ == "__main__":
    main()
