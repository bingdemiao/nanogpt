#!/bin/bash
# Step 6 — cheap, high-info mini-sweep of (group_size, reconn_sz) at the
# tuned LR=1.5e-3 / r_init_scale=0.5 anchor. NO input_shuffle, NO internal_bias.
#
#  run  gs   r    r/gs   MLP/d²   notes
#  1    64   8    0.125  6.375    existing baseline (val 1.31)              SKIP
#  2    32   32   1.0    9.0      existing (val 1.29)                       SKIP
#  3    16   16   1.0    9.0      same params as run 2, finer groups
#  4    32    8   0.25   6.75     nearly-vanilla cost with double groups
#
# Run 3 vs Run 2 isolates `gs` at matched cost — does granularity help, or only
# total R-rank? Run 4 vs Run 1 isolates `gs` at matched r — does cheap "more
# groups" help by itself?
#
# Skipping runs 1 and 2 since their checkpoints already exist
# (out-tinystories-prism-3x-lr1.5e-3-r0.5 and
#  out-tinystories-prism-3x-gs32-r32).
#
# Wallclock: 50k iters at this scale was ~1h08m for gs=32/r=16 — using
# 8h cap as in earlier sweeps to leave headroom.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/gs_r_mini_prism_3x.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

LR=1.5e-3
MIN_LR=1.5e-4
R_INIT=0.5

# (gs, r) pairs to launch
PAIRS=(
    "16 16"
    "32 8"
)

echo "=== sweep plan ==="
echo "config:        $CONFIG"
echo "fixed:         LR=$LR  min_lr=$MIN_LR  r_init_scale=$R_INIT  (no shuffle, no internal_bias)"
echo "(gs, r) grid:  ${PAIRS[*]}"
echo "n_jobs:        ${#PAIRS[@]}"
echo ""

for pair in "${PAIRS[@]}"; do
    set -- $pair
    gs=$1; r=$2
    out_dir="out-tinystories-prism-3x-gs${gs}-r${r}"
    run_name="prism-3x-gs${gs}-r${r}"
    extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --group_size=$gs --reconn_sz=$r --out_dir=$out_dir --wandb_run_name=$run_name"

    echo ">>> submitting gs=$gs  r=$r"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
