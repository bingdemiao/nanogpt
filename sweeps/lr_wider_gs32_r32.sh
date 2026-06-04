#!/bin/bash
# Step 8 — wider LR sweep at (gs=32, r=32).
#
# The 1.5e-3 LR was inherited from the old shape (gs=64, r=8). Never
# verified to be the peak at the new shape. r_lr_mult sweep below 1.0
# was monotonically worse, so R-vs-B ratio isn't the issue; the
# question is whether the absolute base LR is right for this shape.
#
# Grid: log-spaced from 1e-4 up to 4e-3 (40× range). Existing control
# at LR=1.5e-3 is job 3493213 (val 1.3123 at 30k). Six new jobs:
#
#   LR ∈ {1e-4, 3e-4, 6e-4, 1e-3, 2.5e-3, 4e-3}
#   min_lr = LR/10  (matches existing pattern; cosine decay endpoint
#                    shifts with LR, so all runs see same relative decay)
#
# Other settings identical to job 3493213:
#   gs=32, r=32, r_lr_mult=1.0, r_init=0.5, max_iters=30000,
#   no shuffle, ibF, dropout=0
#
# train.py now early-stops on non-finite loss so high-LR points that
# diverge waste only a few iterations of wallclock, not the full 6h.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/lr_wider_gs32_r32.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000

# Locked shape (gs=32, r=32) at the current best point.
R_INIT=0.5
GS=32
RSZ=32

# (lr, min_lr) pairs — min_lr = lr/10
PAIRS=(
    "1e-4 1e-5"
    "3e-4 3e-5"
    "6e-4 6e-5"
    "1e-3 1e-4"
    "2.5e-3 2.5e-4"
    "4e-3 4e-4"
)

echo "=== sweep plan ==="
echo "config:        $CONFIG"
echo "fixed:         gs=$GS  r=$RSZ  r_init=$R_INIT  r_lr_mult=1.0"
echo "max_iters:     $MAX_ITERS"
echo "control:       LR=1.5e-3 → job 3493213 (val 1.3123)"
echo "n_jobs:        ${#PAIRS[@]}"
echo ""

for pair in "${PAIRS[@]}"; do
    set -- $pair
    lr=$1; minlr=$2
    # Replace dots in out_dir tag for cleanliness
    lr_tag=$(echo "$lr" | tr -d '.')
    out_dir="out-tinystories-prism-3x-gs${GS}-r${RSZ}-lr${lr_tag}"
    run_name="prism-3x-gs${GS}-r${RSZ}-lr${lr}"
    extra="--learning_rate=$lr --min_lr=$minlr --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$RSZ --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> LR=$lr  min_lr=$minlr  out_dir=$out_dir"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER" 2>/dev/null
