# nanogpt-prism

Karpathy's nanoGPT with the MLP up-projection swapped for `cute_prism.PrismLinear`.

```
LayerNorm(d) → PrismLinear(d, 4d, silu_gate, cublas, bf16) → GroupNorm(n_groups) → Linear(4d, d) → Dropout
```

Layout:

| file | purpose |
|---|---|
| `model.py` | GPT with `VanillaMLP` (baseline) and `PrismMLP` (default) |
| `train.py` | DDP-aware train loop, bf16 throughout, AdamW with the §4 param groups |
| `configurator.py` | `python train.py config/foo.py --batch_size=8` overrides |
| `sample.py` | Generate from a checkpoint |
| `config/train_*.py` | Per-dataset training configs |
| `data/*/prepare.py` | Tokenization scripts (writes `train.bin`, `val.bin`, `meta.pkl`) |
| `submit.sh` | Slurm wrapper (CSCS Clariden, container env `nanogpt` → `~/.edf/nanogpt.toml`) |
| `sweeps/*.sh` | Multi-job launchers — hyperparam sweeps, scale-up ablations, one-shot data prep |
| `tools/count_flops.py` | Per-config forward FLOPs (with the cute_prism custom-op matmuls added back analytically) |
| `tools/parse_iter_times.py` | Median ms/iter from any past training log (skips first 100 iters as warmup) |

## Quick start (single GPU smoke test)

The `nanogpt` env reuses `oft.sqfs` but mounts `/users/lshuhao/nanogpt` as
`/workspace`. For login-node smoke tests use enroot directly:

```bash
ENROOT_SLURM_HOOK=off enroot start --rw \
    -m /users/lshuhao/nanogpt:/workspace \
    -m /users/lshuhao/cute_mma:/users/lshuhao/cute_mma \
    -m /users/lshuhao/.cache:/users/lshuhao/.cache \
    -e CUTE_PRISM_CACHE_DIR=/tmp/cute_prism_cache \
    -e NCCL_SOCKET_IFNAME=lo \
    /iopsstor/scratch/cscs/lshuhao/containers/oft.sqfs \
    bash

# inside the container:
cd /workspace
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py        # ~5 min / GH200
python sample.py --out_dir=out-shakespeare-char-prism --start=ROMEO:
```

Verified smoke-test result (50 iters, GH200): train loss 4.25 → 2.56,
≈77 ms/iter, kernel auto-compiles on first call.

## Multi-GPU / Slurm

```bash
# inside container, single node:
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb.py

# Slurm:
sbatch --time=4:00:00 \
    --export=ALL,CONFIG=config/train_fineweb.py \
    submit.sh
```

## Datasets

```bash
python data/shakespeare_char/prepare.py   # ~1MB,    ~1M tokens
python data/tinystories/prepare.py        # ~1.5GB,  ~500M tokens
python data/fineweb/prepare.py            # ~20GB,   ~10B tokens (FineWeb-Edu sample-10BT)
python data/openwebtext/prepare.py        # ~17GB,   ~9B tokens
```

`fineweb`, `openwebtext`, and `tinystories` use GPT-2 BPE (50257 vocab) via
`tiktoken`. `shakespeare_char` uses a 65-symbol char vocab and writes its
`stoi`/`itos` tables to `meta.pkl`.

### Difficulty ordering (which dataset for which scale)

| # | Dataset | Tokens | Vocab | Type | Status | Suitable model size |
|---|---|---|---|---|---|---|
| 1 (trivial) | **shakespeare_char** | ~1 M | 65 char | Single Shakespeare text | ✓ prepared | <1 M params |
| 2 (easy) | **tinystories** | ~470 M | 50 k BPE | Children's stories, constrained vocab | ✓ prepared | 1 M – 30 M params |
| 3 (hard) | **fineweb-edu (10BT)** | ~10 B | 50 k BPE | Filtered high-quality web text | not prepared (run `sbatch sweeps/prep_fineweb.sh`) | 50 M – 1 B+ params |
| 4 (hardest) | **openwebtext** | ~9 B | 50 k BPE | Raw scraped web text | not prepared | 50 M – 1 B+ params |

TinyStories saturates around perplexity 3.5 for 50 M+ param models — the
constrained vocabulary/style means architecture differences get squashed
once both models hit the dataset's complexity ceiling. **FineWeb-Edu is
the recommended next challenge** for 50 M–125 M models: 10 B tokens
(plenty of headroom), modern curated web text, and the standard benchmark
for current nanoGPT-scale work. OpenWebText is older and noisier;
generally skip it unless you specifically want cross-dataset robustness.

## Baseline vs Prism

Set `--mlp_type=vanilla` (or edit the config) to fall back to GPT-2's GELU
MLP for an apples-to-apples comparison; everything else is identical.

## Methodology notes (read before reporting results)

### 1. Use MLP-relative param percentage as the primary metric

At TinyStories scale, the token embedding alone is 26 M params (51% of
total) — that's noise that makes prism look like a 2% effect when the
change is actually concentrated in the MLP. Report the MLP-relative view
as the headline, total as a secondary number. At GPT-2 scale (where MLP
is ~46% of total) the two views converge.

TinyStories config (`n_layer=8, n_embd=512, group_size=64, reconn_sz=8`):

| Config | MLP params | vs vanilla 4× | Total params |
|---|---|---|---|
| vanilla 4× | 16.78 M | (baseline) | 51.19 M |
| prism 4× | 17.86 M | **+6.4%** | 52.27 M |
| prism 3× | 13.39 M | **−20.2%** | 47.81 M |
| prism 2× | 8.93 M | **−46.8%** | 43.34 M |

Per-block math (with `gs=64, r=8`):

```
Vanilla MLP params = 8 · d²
Prism   MLP params = (2x + x·r/gs) · d²   where x is mlp_expansion (default 4)
                   = 2.125 · x · d²       at x=4 → 8.5·d² (+6.25% vs vanilla)
```

### 2. The vanilla baseline is NOT independently tuned

Both vanilla and prism share the same fixed training recipe. This is
fair for an A/B comparison of the architecture, but does **not**
establish that either is at its own optimum.

Hyperparameters (inherited from nanoGPT defaults; not swept):

| Param | Value | Rationale |
|---|---|---|
| `learning_rate` | 6e-4 | nanoGPT default; not swept |
| `min_lr` | 6e-5 | LR/10, nanoGPT convention |
| `warmup_iters` | 1000 | ≈2% of total iters |
| `max_iters` | 50,000 | Long enough to plateau at TinyStories scale |
| `lr_decay_iters` | 50,000 | Cosine decay reaches `min_lr` at end |
| effective batch | 65,536 tok/iter (bs=32, accum=4, seq=512) | Fits one GH200 in bf16 |
| total tokens | 3.28 B | ≈7 epochs over TinyStories (~470 M tokens) |
| optimizer | AdamW(0.9, 0.95), wd=0.1, grad_clip=1.0 | Standard |
| dtype | bfloat16 | Required by `cute_prism` cublas backend |

**4× expansion** is the GPT-2 default (Radford et al. 2019) inherited by
GPT-3/4, Llama, Mistral. It is the *convention*, not the *optimum* —
e.g. Shazeer 2020 ("GLU Variants Improve Transformer") used 8/3 ≈ 2.67×
for SwiGLU to keep params constant.

The defensible claim from fixed-recipe runs is:

> "At fixed training recipe (LR=6e-4, 50k iters, bs=128 seqs of 512
> tokens, AdamW(0.9, 0.95), wd=0.1, cosine decay), `prism-3×` achieves
> val loss X with 20% fewer MLP params than `vanilla-4×`."

To upgrade to *"at each config's best-known recipe"*, run the sweep in
Step 5 below for both vanilla and prism, then compare.

## Practical comparison summary (params vs FLOPs vs wallclock vs loss)

The thesis "fewer params, similar performance" requires looking at four
axes together. Below: numbers for TinyStories (50 M-param-scale) and
GPT-2-small / FineWeb. Wallclock is the median ms/iter from
[`tools/parse_iter_times.py`](tools/parse_iter_times.py); FLOPs are
from [`tools/count_flops.py`](tools/count_flops.py) (the prism
reconn+B@x matmuls are added back analytically since they're hidden
inside the `cute_prism::forward` custom op).

### TinyStories (50 M params, bs=128 × seq=512)

| Config | Params | MLP params | Fwd FLOPs | Best val_loss | ms/iter |
|---|---|---|---|---|---|
| vanilla 4× | 51.19 M | 16.78 M | 30.12 G | **1.2145** | **78** |
| prism 4× | 52.27 M | 17.86 M (+6.4%) | 31.19 G (+3.6%) | ~1.35 *(tuned LR n/a)* | ~480 |
| prism 3× | 47.81 M (−6.6%) | 13.39 M (−20%) | 26.63 G (−12%) | 1.3100 (+0.10) | 486 (6.2×) |
| prism 2× | 43.34 M (−15%) | 8.93 M (−47%) | 22.06 G (−27%) | 1.3813 (+0.17) | 389 (5.0×) |

### GPT-2 small / FineWeb-Edu (124 M params, bs=480 × seq=1024)

| Config | Params | MLP params | Fwd FLOPs | val_loss @ 100k | ms/iter |
|---|---|---|---|---|---|
| vanilla 4× | 124.37 M | 56.62 M | 212.68 G | **2.9653** | **312** |
| prism 4× | 127.99 M | 60.24 M (+6.4%) | 219.93 G (+3.4%) | *running* | 3153 (10.1×) |
| prism 3× | 112.93 M (−9%) | 45.30 M (−20%) | 189.12 G (−11%) | *running* | 2639 (8.5×) |
| prism 2× | 97.87 M (−21%) | 30.20 M (−47%) | 158.32 G (−26%) | *running* | 2121 (6.8×) |

### Takeaways for any writeup

1. **Prism's slight FLOPs+param advantage** at 3× and 2× expansion is
   real but moderate (~11% / ~26% fewer FLOPs respectively).
2. **The wallclock penalty dominates** any per-iter speedup expected
   from fewer FLOPs. cute_prism's training-compatible cublas backend is
   5–10× slower than `nn.Linear`. The fast `cute` backend exists
   (see `cute_mma/FUTURE_OPTIMIZATIONS.md`) but doesn't yet support
   training.
3. **At fixed iter count**, prism is worse by ~0.1 val_loss across
   both architectures and both data scales. The gap widens slightly at
   GPT-2 scale (in progress, but already visible at iter 24-35 k).
4. **At fixed wallclock**, vanilla would do 6–10× more iters than
   prism, so prism's parameter advantage doesn't translate into a
   training-time advantage today.

The defensible single-sentence claim:

> "At fixed training recipe and iter count, prism-3× achieves val
> loss within 8% of vanilla-4× at TinyStories scale while using 20%
> fewer MLP params and 12% fewer forward FLOPs, but at the current
> cublas-backed training kernel, it pays a 6× wallclock penalty."

Whether that's a win depends on the audience. For inference (where the
faster cute backend is usable), it's a real story. For training,
it's not yet.

## Project roadmap

### Step 1–3 — done (TinyStories baseline)

- [x] Vanilla 4× run: `nanogpt-vanilla-tinystories` (wandb `u0krjgq3`)
- [x] Prism 4× run: `nanogpt-prism-tinystories` (wandb `oua4pbt1`,
      reached iter 45000, val loss 1.3566)
- [x] Logging scaffolding: prism summary panels (`ar_std_mean`,
      `dead_gate_frac_max`, `out_norm_ratio_max`, etc.)

### Step 4 — param-matched ablation (done)

Results at fixed recipe (LR=6e-4, 50k iters):

| Run | MLP params | Total params | Final val loss | vs vanilla |
|---|---|---|---|---|
| vanilla 4× | 16.78 M | 51.19 M | **1.2459** | (baseline) |
| prism 4× | 17.86 M | 52.27 M | 1.3534 | +0.108 |
| prism 3× | 13.39 M | 47.81 M | 1.3612 | +0.115 |
| prism 2× | 8.93 M | 43.34 M | 1.3813 | +0.135 |

Two findings:

1. **Prism loses to vanilla by 0.11 val_loss** at fixed recipe — even
   prism-4× (which has *more* MLP params than vanilla-4×) is worse. This
   strongly implies prism is mistuned or undertrained, not structurally
   inferior.
2. **Prism is extremely robust to MLP-width shrinking**: going 4× → 2×
   (50% fewer MLP params) costs only 0.028 val_loss. The reconnection
   matrix `R` is doing most of the work; the `B` projection adds little
   once `R` is in place.

If Step 5 closes the 0.11 vanilla–prism gap, prism-3× (or even 2×)
becomes a real "fewer params, same performance" win. If it doesn't,
the structural cost is unrecoverable at this scale.

### Step 5 — per-config hyperparam sweep

After the rough runs, each config gets its own tuning pass. ≈8–12 runs per
config to find the best operating point, then re-compare across configs.

#### Step 5a — prism-3× LR × r_init_scale sweep (done)

11 runs covering `lr ∈ {3e-4, 6e-4, 1e-3, 1.5e-3} × r_init ∈ {0.5, 1.0, 2.0}`
(baseline `(6e-4, 1.0)` skipped):

| LR | r_init=0.5 | r_init=1.0 | r_init=2.0 |
|---|---|---|---|
| 3e-4 | 1.4917 | 1.4948 | 1.4948 |
| 6e-4 | 1.3624 | 1.3612 *(baseline)* | 1.3620 |
| 1e-3 | 1.3258 | 1.3301 | 1.3298 |
| **1.5e-3** | **1.3100** ★ | 1.3144 | 1.3170 |

Findings:

- **LR is the dominant knob.** Default 6e-4 was way too low for
  prism-3×. Raising LR to 1.5e-3 dropped val_loss by 0.051.
- **`r_init_scale` barely matters.** At any fixed LR the spread across
  {0.5, 1.0, 2.0} is only 0.005-0.011 val_loss.
- **Peak not found yet.** Val_loss was still decreasing with higher LR
  at 1.5e-3, so the optimum may be at 2e-3 or 3e-3. Extension sweep
  prepared in [`sweeps/lr_prism_3x_extend.sh`](sweeps/lr_prism_3x_extend.sh).

Gap to vanilla closed from **0.115 → 0.064** (about half).

#### Step 5b — vanilla LR sweep (done)

Sweep `lr ∈ {3e-4, 6e-4, 1e-3, 1.5e-3, 2e-3}` on vanilla-4×:

| LR | val_loss |
|---|---|
| 3e-4 | 1.3136 |
| 6e-4 *(orig baseline)* | 1.2459 |
| 1e-3 | 1.2248 |
| 1.5e-3 | 1.2177 |
| **2e-3** | **1.2145** ★ |

Vanilla also wanted higher LR — and the new best is **1.2145**.

#### Step 5 — TinyStories conclusion (tuned vs tuned)

| Config | Best val_loss | At LR | MLP params | vs vanilla |
|---|---|---|---|---|
| **vanilla 4× (tuned)** | **1.2145** | 2e-3 | 16.78 M | (baseline) |
| prism 3× (tuned) | 1.3100 | 1.5e-3, r_init=0.5 | 13.39 M (−20%) | +0.096 |

The vanilla-prism gap stays at **~0.1 val_loss** even with both
architectures at their sweep-best operating points. At TinyStories
scale, **prism-3× trades off perplexity for parameter savings** —
it does not beat vanilla per-param. The remaining question is whether
this is a TinyStories saturation artifact (the dataset is too easy for
50 M-param models, see [Datasets section](#datasets)) or a genuine
structural limitation. **Step 6 (FineWeb at GPT-2 scale) tests this.**

Open knobs we haven't swept (defer until FineWeb results suggest they
might matter):

| Knob | Default | Sweep range | Why |
|---|---|---|---|
| `learning_rate` | 6e-4 | {3e-4, 6e-4, 1e-3, 1.5e-3} | Smaller MLPs often prefer higher LR; prism may have its own optimum |
| `r_lr_mult` | 1.0 | {0.5, 1.0, 2.0} | R is small and sensitive — separate LR scale may help |
| `prism_internal_bias` | False | {False, True} | Adds learnable bias inside gate; trades tiny params for plasticity |
| `prism_input_shuffle` | False | {False, True} | Per-group butterfly shuffle of A before A@R; increases cross-group view diversity. **Requires `reconn_sz=16`** (validated at construction). |
| `r_init_scale` | 1.0 | {0.5, 1.0, 2.0} | Init magnitude of R; affects whether reconnection contributes early |
| `reconn_sz` | 8 | {4, 8, 16} | Larger r → more R capacity, more params, slower kernel |

`prism_input_shuffle` and `prism_shuffle_blk_k` are now exposed in
`GPTConfig`. Constraints confirmed empirically against the cute_prism
build that's loaded inside the `nanogpt` container env:
- `prism_input_shuffle=True` requires `reconn_sz=16` (PrismLinear raises)
- `n_embd` must be divisible by `prism_shuffle_blk_k` (default 128)
- backend must be `cublas` or `pytorch`; the `cute` kernel in this build
  rejects input_shuffle (the host-side source at `~/cute_mma/...` shows
  a different version than what's actually loaded — check
  `inspect.getsourcefile(cute_prism._module)` for the live path).
  Training uses `cublas` per TRAINING_GATED_PRISM.md §9 so this doesn't
  affect us.
- works with `bf16` + `silu_gate` (our training setup; both forward and
  backward verified)

Smallest ablation to test the shuffle (2 jobs, ~5 h each):
[`sweeps/input_shuffle_prism_3x.sh`](sweeps/input_shuffle_prism_3x.sh).
Runs reconn_sz=16 with and without shuffle at the tuned recipe
(lr=1.5e-3, r_init=0.5) to separate "bigger R" from "shuffle helps".

**Sweep pattern: one base config + bash loop with `EXTRA` overrides.** No
need for N config files; `submit.sh` reads `--out_dir=` from `EXTRA` so
each sweep point has its own checkpoint directory and wandb run.

Ready-made launcher: [`sweeps/lr_prism_3x.sh`](sweeps/lr_prism_3x.sh)
runs a 2-D sweep over `learning_rate × r_init_scale` (4 LRs × 3 init
scales = 12 cells, skips the `(6e-4, 1.0)` baseline duplicate → 11
jobs). Edit `LRS=(...)` and `R_INITS=(...)` at the top to shrink the
grid.

```bash
bash sweeps/lr_prism_3x.sh        # the default 11-job sweep
```

For ad-hoc sweeps along any other axis (e.g. `r_lr_mult`,
`prism_internal_bias`) copy the script and swap the inner loop.

### Step 6 — replicate at GPT-2 scale on FineWeb-Edu

If Step 5 closes the vanilla-prism gap, scale up to confirm it survives.
At GPT-2 small (`n_layer=12, n_head=12, n_embd=768`) on FineWeb-Edu
(10 B tokens), the embedding becomes a smaller fraction of total params
and the MLP delta becomes more visible.

Configs (all in [`config/`](config/)):

| File | Variant | Total params | MLP params | Hidden width `d_ff` |
|---|---|---|---|---|
| `train_fineweb_vanilla.py` | vanilla 4× | 124.37 M | 56.62 M | 3072 |
| `train_fineweb.py` | prism 4× | 127.99 M | 60.24 M (+6.4%) | 3072 |
| `train_fineweb_prism_3x.py` | prism 3× | 113.05 M | 45.30 M (−20.0%) | 2304 |
| `train_fineweb_prism_2x.py` | prism 2× | 97.95 M | 30.20 M (−46.7%) | 1536 |

Launchers:

```bash
# One-shot: download + tokenize FineWeb-Edu sample-10BT (~30-60 min,
# writes ~20 GB to /capstor/scratch via the symlinked data/fineweb/).
sbatch sweeps/prep_fineweb.sh

# Step-6 ablation: submit all 4 GPT-2-scale runs in parallel.
# ~17-20 h wallclock each on 4× GH200 with bf16, 100k iters, ~50 B
# tokens trained per run (~5 epochs over the 10 B-token corpus).
bash sweeps/fineweb_param_matched.sh
```

Before launching the FineWeb ablation, update the prism configs with
the best `learning_rate` / `r_init_scale` found in Step 5 — don't pay
20 GPU-hours per run for a recipe we already know is suboptimal.

### Step 7 — architectural variants (optional, paper-grade)

Beyond `mlp_expansion`, the natural next axes are:

- **Prism the `c_fc` only or the `c_proj` only** — currently `up` is prism
  and `down` is plain Linear. Mirror-imaging or doubling up changes the
  param budget materially.
- **`(group_size, reconn_sz)` joint sweep** — current default
  `(64, 8)` gives `n_groups = d_ff / 64` pathways. Tighter
  groups (`gs=32`) give more independent pathways but smaller per-group
  rank.
- **Multi-token / Mixture-of-Experts pairing** — if prism's diversity-
  per-param wins on dense MLPs, it composes naturally with MoE.

### Step 8 — known bugs / polish (anytime)

- [ ] `grad_norm` is currently logged as 0 in [train.py:295-323](train.py#L295-L323) — pre-existing
      bug, doesn't affect training but breaks the dashboard. Fix before
      writing up results.
- [x] Expose `prism_input_shuffle` and `prism_shuffle_blk_k` in
      `GPTConfig`. Done with validation: enforces `reconn_sz=16` and
      `n_embd % shuffle_blk_k == 0`. Sweep script in
      [`sweeps/input_shuffle_prism_3x.sh`](sweeps/input_shuffle_prism_3x.sh).
- [ ] `data/`, `out-*` directories are symlinked to `/capstor/scratch`
      to keep `/users/lshuhao` (50 GB quota) clean. The slurm container
      env (`~/.edf/nanogpt.toml`) mounts `/capstor/scratch` so they
      resolve inside the container — confirmed.

## Sweep scripts

All multi-job launchers live in [`sweeps/`](sweeps/). They use one base
config plus `EXTRA` overrides — no need for N nearly-identical config
files.

| Script | What it does | When to run |
|---|---|---|
| [`sweeps/lr_prism_3x.sh`](sweeps/lr_prism_3x.sh) | Step 5a — 2-D sweep `learning_rate × r_init_scale` on TinyStories prism-3× (11 jobs). **Done.** Best: lr=1.5e-3, r_init=0.5 → val 1.3100. | Already run. |
| [`sweeps/lr_prism_3x_extend.sh`](sweeps/lr_prism_3x_extend.sh) | Step 5a-extend — push prism-3× LR to {2e-3, 3e-3} at r_init=0.5 (2 jobs). The first sweep ended on a still-decreasing trend. | Optional — only if you want to find the actual prism LR peak. |
| [`sweeps/lr_vanilla.sh`](sweeps/lr_vanilla.sh) | Step 5b — vanilla 4× LR sweep on TinyStories at `{3e-4, 1e-3, 1.5e-3, 2e-3}` (4 jobs). | Run before Step 6 — needed for a fair vanilla-vs-prism baseline. |
| [`sweeps/prep_fineweb.sh`](sweeps/prep_fineweb.sh) | One-shot data prep — downloads FineWeb-Edu sample-10BT and tokenizes to `data/fineweb/{train,val}.bin` (~20 GB on scratch). **Done.** | Already run. |
| [`sweeps/fineweb_param_matched.sh`](sweeps/fineweb_param_matched.sh) | Step 6 — submits all four GPT-2-scale runs (vanilla 4× + prism 4×/3×/2×) on FineWeb in parallel. | After Step 5b. Override LR via EXTRA per config when launching (e.g. `EXTRA="--learning_rate=1.5e-3 --min_lr=1.5e-4"` for the prism configs). |
| [`sweeps/measure_flops.sh`](sweeps/measure_flops.sh) | One-shot — runs [`tools/count_flops.py`](tools/count_flops.py) inside the container env to produce the FLOPs comparison table. | Any time the model code changes. ~30 s. |
| [`sweeps/input_shuffle_prism_3x.sh`](sweeps/input_shuffle_prism_3x.sh) | 2 jobs at the tuned recipe: prism-3× with reconn_sz=16 only vs reconn_sz=16 + `input_shuffle=True`. Separates "bigger R" from "shuffle helps". | After FineWeb runs settle. Or now — they're independent. |

To write a new sweep, copy `lr_prism_3x.sh`, change the loop variable,
and update the `out_dir` / `wandb_run_name` template. The pattern:

```bash
sbatch --time=$WALLCLOCK --gpus-per-node=$GPUS_PER_NODE \
    --export=ALL,CONFIG=$CONFIG,EXTRA="--<knob>=$val --out_dir=... --wandb_run_name=..." \
    submit.sh
```

## Known constraints

* Training requires `prism_backend=cublas` + bf16 (per
  `cute_mma/TRAINING_GATED_PRISM.md` §9).
* `mlp_expansion * n_embd` must be divisible by `group_size`; `n_embd`
  must be divisible by `reconn_sz`.
* `torch.compile` is opt-in (`compile=True`). The `cute_prism::forward`
  custom op is opaque to the compiler — surrounding ops still trace, but
  the layer itself stays in eager.

See `cute_mma/TRAINING_GATED_PRISM.md` for the full set of rules around
weight decay, dropout, gradient clipping, and monitors.
