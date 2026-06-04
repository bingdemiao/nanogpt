#!/bin/bash
# Step 12 — verify the "optimal LR scales with r" hypothesis by holding
# gs FIXED at 32 and sweeping r x LR. This isolates r's effect on the
# optimal LR with no gs confound.
#
# The r=32 row is already done (wider LR sweep + r_lr_mult sweep):
#   (gs=32, r=32): LR 1.5e-3 -> 1.3098, 2.5e-3 -> 1.2926, 4e-3 -> 1.2919
# This script fills the r=8 and r=16 rows:
#
#         LR=1.5e-3   LR=2.5e-3   LR=4e-3
#   r=8     new         new         new
#   r=16    new         new         new       (also: r=16 no-shuffle never run)
#   r=32    done        done        done
#
# Prediction (if LR ~ r holds): per-row optimum shifts right as r grows
#   r=8  peaks ~1.5e-3
#   r=16 peaks ~2.5e-3
#   r=32 peaks ~4e-3 (already)
#
# Fixed: gs=32, r_init=0.5, r_lr_mult=1.0, input_shuffle=False, ibF,
#        dropout=0, max_iters=30000. NaN early-stop guards high-LR points.
#
#     bash sweeps/verify_lr_vs_r.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=06:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000
R_INIT=0.5
GS=32

# (r, lr, min_lr)
RUNS=(
    "8  1.5e-3 1.5e-4"
    "8  2.5e-3 2.5e-4"
    "8  4e-3   4e-4"
    "16 1.5e-3 1.5e-4"
    "16 2.5e-3 2.5e-4"
    "16 4e-3   4e-4"
)

echo "=== verify LR vs r (fixed gs=$GS) ==="
echo "fixed: gs=$GS r_init=$R_INIT r_lr_mult=1.0 shuffle=off ibF dropout=0 iters=$MAX_ITERS"
echo "n_jobs: ${#RUNS[@]} (r=32 row already complete)"
echo ""

for tup in "${RUNS[@]}"; do
    set -- $tup
    r=$1; lr=$2; minlr=$3
    lr_tag=$(echo "$lr" | tr -d '.')
    out_dir="out-tinystories-prism-3x-gs${GS}-r${r}-lr${lr_tag}-verify"
    run_name="prism-3x-gs${GS}-r${r}-lr${lr}-verify"
    extra="--learning_rate=$lr --min_lr=$minlr --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$r --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> gs=$GS r=$r LR=$lr"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
squeue -u "$USER" 2>/dev/null | tail -12
