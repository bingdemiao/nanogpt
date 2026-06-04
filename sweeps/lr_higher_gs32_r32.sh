#!/bin/bash
# Step 9 — push LR higher at (gs=32, r=32).
#
# Previous wider LR sweep at this shape (30k iters):
#   LR=1.5e-3 (inherited)  val 1.3098
#   LR=2.5e-3              val 1.2926
#   LR=4e-3                val 1.2919
# Curve is essentially flat at the top — either at the peak or about to
# hit a stability cliff. Two probes:
#
#   LR=6e-3   probably still improves marginally if 4e-3 was not the peak
#   LR=1e-2   stress test — likely to NaN; train.py early-stop will cut
#             wallclock if so
#
# All other settings identical to job 3493213 / the wider-LR sweep:
#   gs=32, r=32, r_lr_mult=1.0, r_init=0.5, max_iters=30000,
#   no shuffle, ibF, dropout=0.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000
R_INIT=0.5
GS=32
RSZ=32

PAIRS=(
    "6e-3 6e-4"
    "1e-2 1e-3"
)

echo "=== sweep plan ==="
echo "shape:    gs=$GS r=$RSZ  r_init=$R_INIT  r_lr_mult=1.0"
echo "iters:    $MAX_ITERS"
echo "n_jobs:   ${#PAIRS[@]}"
echo ""

for pair in "${PAIRS[@]}"; do
    set -- $pair
    lr=$1; minlr=$2
    lr_tag=$(echo "$lr" | tr -d '.')
    out_dir="out-tinystories-prism-3x-gs${GS}-r${RSZ}-lr${lr_tag}"
    run_name="prism-3x-gs${GS}-r${RSZ}-lr${lr}"
    extra="--learning_rate=$lr --min_lr=$minlr --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$RSZ --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> LR=$lr  min_lr=$minlr"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

squeue -u "$USER" 2>/dev/null | tail -10
