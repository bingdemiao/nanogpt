#!/bin/bash
# Step 5b — vanilla 4× LR sweep on TinyStories.
#
# We swept prism-3×'s LR (it wanted 1.5e-3, not the default 6e-4).
# To make the prism-vs-vanilla comparison fair we have to verify vanilla
# at its own optimum, not just the inherited nanoGPT default. The existing
# vanilla run at lr=6e-4 (val 1.2459) is the "control".
#
# Run from the nanogpt repo root::
#
#     bash sweeps/lr_vanilla.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_vanilla.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

# Skip 6e-4 (baseline already exists). Sweep above and below it.
LRS=(3e-4 1e-3 1.5e-3 2e-3)

echo "=== sweep plan ==="
echo "config:        $CONFIG"
echo "wallclock:     $WALLCLOCK"
echo "GPUs/job:      $GPUS_PER_NODE"
echo "LRs:           ${LRS[*]}"
echo "n_jobs:        ${#LRS[@]}"
echo ""

for lr in "${LRS[@]}"; do
    minlr=$(awk "BEGIN{printf \"%g\", $lr/10}")
    out_dir="out-tinystories-vanilla-lr$lr"
    run_name="vanilla-lr$lr"
    extra="--learning_rate=$lr --min_lr=$minlr --out_dir=$out_dir --wandb_run_name=$run_name"

    echo ">>> submitting lr=$lr"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" \
        submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
