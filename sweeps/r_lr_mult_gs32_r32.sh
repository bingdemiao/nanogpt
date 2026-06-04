#!/bin/bash
# Step 7 — r_lr_mult sweep at the current best (gs, r) point: (gs=32, r=32).
#
# Hypothesis ("R as router"): R should learn more slowly than B. At
# (gs=32, r=32), R has 8× more parameters than at the original baseline
# (gs=64, r=8) — so unconstrained R growth is more likely to be a
# problem here than in the prior r_lr_mult sweep (which was at gs=64,
# r=8 and showed no benefit from lowering r_lr_mult).
#
# r_lr_mult applies only to the R parameter group's LR ([model.py:523]).
# B, internal_bias, and no_decay groups keep the full learning_rate.
#
# Grid: r_lr_mult ∈ {0.1, 0.3, 1.0}.
#   1.0 = control (matches existing 50k run 3474784 at val 1.2874, but at
#         the shorter 30k schedule for apples-to-apples comparison).
#
# max_iters = 30000.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/r_lr_mult_gs32_r32.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000

# Locked best (gs, r) point.
LR=1.5e-3
MIN_LR=1.5e-4
R_INIT=0.5
GS=32
RSZ=32

MULTS=(0.1 0.3 1.0)

echo "=== sweep plan ==="
echo "config:        $CONFIG"
echo "fixed:         LR=$LR  gs=$GS  r=$RSZ  r_init=$R_INIT"
echo "max_iters:     $MAX_ITERS"
echo "r_lr_mult:     ${MULTS[*]}"
echo ""

for mult in "${MULTS[@]}"; do
    out_dir="out-tinystories-prism-3x-gs${GS}-r${RSZ}-rmult${mult}"
    run_name="prism-3x-gs${GS}-r${RSZ}-rmult${mult}"
    extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$RSZ --r_lr_mult=$mult --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> r_lr_mult=$mult"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER" 2>/dev/null
