"""Reproduce + localize the NaN seen in the ibT-d* sweep runs.

Stage 1: isolated PrismLinear (already known to be clean — kept as sanity).
Stage 2: full GPT model with the exact sweep config (mlp_expansion=3,
         group_size=32, reconn_sz=16, input_shuffle=True, internal_bias=True,
         r_init_scale=0.5), autocast(bf16), train() mode, one fwd+bwd.

If stage 2 NaNs, drill down per-layer to find where.
"""
import os
import sys
import torch
import torch.nn as nn
import cute_prism

sys.path.insert(0, "/workspace")
import model as gpt_model
from model import GPTConfig, GPT  # noqa: E402


def has_bad(t):
    if t is None:
        return False
    return bool(torch.isnan(t).any().item() or torch.isinf(t).any().item())


def summ(t):
    if t is None:
        return "None"
    n = torch.isnan(t).any().item()
    i = torch.isinf(t).any().item()
    return f"shape={tuple(t.shape)} amax={float(t.abs().max()):.4g} nan={n} inf={i}"


def stage1_isolated_prismlinear(internal_bias: bool):
    print(f"\n--- stage1 PrismLinear internal_bias={internal_bias} ---")
    torch.manual_seed(0)
    layer = cute_prism.PrismLinear(
        in_features=512, out_features=1536,
        group_size=32, reconn_sz=16,
        bias=False, activation="silu_gate", backend="cublas",
        internal_bias=internal_bias, dropout=0.0, r_init_scale=0.5,
        input_shuffle=True, shuffle_blk_k=16,
    ).cuda().bfloat16()
    layer.train()
    x = torch.randn(4, 128, 512, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    y = layer(x)
    print(f"  y: {summ(y)}")
    y.float().pow(2).mean().backward()
    print(f"  dx: {summ(x.grad)}")
    if layer._internal_bias is not None:
        print(f"  d_ib: {summ(layer._internal_bias.grad)}")


def build_gpt(internal_bias: bool) -> GPT:
    cfg = GPTConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=False,
        # prism options must match config/train_tinystories_prism_3x.py + sweep
        mlp_type="prism",
        prism_backend="cublas",
        prism_activation="silu_gate",
        mlp_expansion=3,
        group_size=32,
        reconn_sz=16,
        r_init_scale=0.5,
        prism_input_shuffle=True,
        prism_internal_bias=internal_bias,
        prism_internal_dropout=0.0,
    )
    return GPT(cfg)


def stage2_full_gpt(internal_bias: bool):
    print(f"\n--- stage2 full GPT internal_bias={internal_bias} ---")
    torch.manual_seed(0)
    m = build_gpt(internal_bias).cuda()
    # train.py:259 casts the model to bf16 (PrismLinear doesn't autocast safely).
    m = m.to(torch.bfloat16)
    m.train()
    # Probe every PrismLinear._internal_bias for bad init.
    bad_init = []
    for n, p in m.named_parameters():
        if "_internal_bias" in n:
            print(f"  init {n}: {summ(p)}")
            if has_bad(p):
                bad_init.append(n)
    if bad_init:
        print(f"  !! bad init params: {bad_init}")

    # Install forward hooks on every PrismMLP submodule to see where NaN first appears.
    bad_layers = []
    handles = []
    for n, mod in m.named_modules():
        if mod.__class__.__name__ in ("PrismMLP", "CausalSelfAttention", "Block",
                                       "LayerNorm", "GroupNormLast"):
            def make_hook(name):
                def hook(_, inp, out):
                    if isinstance(out, tuple):
                        for j, o in enumerate(out):
                            if has_bad(o):
                                bad_layers.append((name, f"out[{j}]", summ(o)))
                    else:
                        if has_bad(out):
                            bad_layers.append((name, "out", summ(out)))
                return hook
            handles.append(mod.register_forward_hook(make_hook(n)))

    torch.manual_seed(1)
    B, T = 4, 128
    X = torch.randint(0, 50304, (B, T), device="cuda")
    Y = torch.randint(0, 50304, (B, T), device="cuda")

    # Match train.py: model is in bf16, so no autocast.
    logits, loss = m(X, Y)
    print(f"  logits: {summ(logits)}")
    print(f"  loss: {float(loss):.4f}  nan={bool(torch.isnan(loss).any())}")
    for h in handles:
        h.remove()
    if bad_layers:
        print("  !! first bad submodule outputs:")
        for (n, slot, s) in bad_layers[:8]:
            print(f"     {n} {slot} {s}")
    else:
        print("  ok all submodule outputs finite")

    # Run backward and check param grads.
    if not torch.isnan(loss).any():
        loss.backward()
        bad_grad = []
        for n, p in m.named_parameters():
            if p.grad is not None and has_bad(p.grad):
                bad_grad.append((n, summ(p.grad)))
        if bad_grad:
            print("  !! bad grads:")
            for (n, s) in bad_grad[:10]:
                print(f"     {n}: {s}")
        else:
            print("  ok all grads finite")


def main():
    print(f"torch={torch.__version__}  cute_prism={cute_prism.__file__}")
    stage1_isolated_prismlinear(internal_bias=False)
    stage1_isolated_prismlinear(internal_bias=True)
    stage2_full_gpt(internal_bias=False)
    stage2_full_gpt(internal_bias=True)


if __name__ == "__main__":
    main()
