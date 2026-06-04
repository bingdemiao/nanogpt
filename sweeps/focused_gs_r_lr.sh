#!/bin/bash
# Step 11 — focused 7-job sweep closing three gaps in the (gs, r) x LR space.
#
# Existing data only LR-tunes (gs=32, r=32). All other (gs, r) points
# tested at LR=1.5e-3 only, which we now know is ~2.7x too low for
# at least the (gs=32, r=32) shape. This sweep adds:
#
#   (A) LR tuning at the current "best" shape (gs=16, r=16)
#       — 3 jobs at LR in {2.5e-3, 4e-3, 6e-3}
#   (B) the never-tested (gs=64, r=32) point at vanilla-budget params
#       — 2 jobs at LR in {1.5e-3, 4e-3}
#   (C) re-test (gs=64, r=8) baseline at higher LR
#       — 2 jobs at LR in {2.5e-3, 4e-3}
#
# All at: r_init=0.5, r_lr_mult=1.0, input_shuffle=False,
# prism_internal_bias=False, dropout=0, max_iters=30000.
# NaN early-stop in train.py:566-572 keeps unstable points cheap.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/focused_gs_r_lr.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000
R_INIT=0.5

# (gs, r, lr, min_lr) tuples
RUNS=(
    # (A) LR tune at (gs=16, r=16) — the current 50k "best" shape
    "16 16 2.5e-3 2.5e-4"
    "16 16 4e-3   4e-4"
    "16 16 6e-3   6e-4"
    # (B) the never-tested (gs=64, r=32) point
    "64 32 1.5e-3 1.5e-4"
    "64 32 4e-3   4e-4"
    # (C) re-test the original baseline at higher LR
    "64 8  2.5e-3 2.5e-4"
    "64 8  4e-3   4e-4"
)

echo "=== sweep plan ==="
echo "fixed:    r_init=$R_INIT  r_lr_mult=1.0  input_shuffle=False  ibF  dropout=0"
echo "iters:    $MAX_ITERS"
echo "n_jobs:   ${#RUNS[@]}"
echo ""

for tup in "${RUNS[@]}"; do
    set -- $tup
    gs=$1; r=$2; lr=$3; minlr=$4
    lr_tag=$(echo "$lr" | tr -d '.')
    out_dir="out-tinystories-prism-3x-gs${gs}-r${r}-lr${lr_tag}"
    run_name="prism-3x-gs${gs}-r${r}-lr${lr}"
    extra="--learning_rate=$lr --min_lr=$minlr --r_init_scale=$R_INIT --group_size=$gs --reconn_sz=$r --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> gs=$gs r=$r LR=$lr  out_dir=$out_dir"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
squeue -u "$USER" 2>/dev/null | tail -10
