#!/bin/bash
# Step 5a-extend — push prism-3× LR higher to find the actual peak.
#
# The first sweep showed val_loss monotonically decreasing through
# lr=1.5e-3 (best: 1.3100 at r_init=0.5). This extends the LR axis at
# the best r_init=0.5 only — 2 jobs.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/lr_prism_3x_extend.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

LRS=(2e-3 3e-3)
R_INIT=0.5

echo "=== extension plan ==="
echo "config:    $CONFIG"
echo "LRs:       ${LRS[*]}"
echo "r_init:    $R_INIT (best from first sweep)"
echo "n_jobs:    ${#LRS[@]}"
echo ""

for lr in "${LRS[@]}"; do
    minlr=$(awk "BEGIN{printf \"%g\", $lr/10}")
    out_dir="out-tinystories-prism-3x-lr$lr-r$R_INIT"
    run_name="prism-3x-lr$lr-r$R_INIT"
    extra="--learning_rate=$lr --min_lr=$minlr --r_init_scale=$R_INIT --out_dir=$out_dir --wandb_run_name=$run_name"

    echo ">>> submitting lr=$lr r_init=$R_INIT"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" \
        submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
