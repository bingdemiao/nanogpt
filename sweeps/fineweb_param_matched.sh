#!/bin/bash
# Step 6 — param-matched ablation at GPT-2 scale on FineWeb-Edu.
#
# Submits 4 sbatch jobs in parallel:
#   vanilla 4×, prism 4×, prism 3×, prism 2×
#
# Each runs 100k iters (~50B tokens, ~5 epochs over the 10B-token corpus)
# on 4× GH200 with bf16. ~17-20 hours wallclock per run.
#
# Prerequisite: bash sweeps/prep_fineweb.sh   (run once; produces ~20GB
# of tokenized data on /capstor/scratch).
#
# Run from the nanogpt repo root::
#
#     bash sweeps/fineweb_param_matched.sh
#
# Compare in wandb (project "nanogpt"):
#   nanogpt-vanilla-fineweb
#   nanogpt-prism-fineweb
#   nanogpt-prism-3x-fineweb
#   nanogpt-prism-2x-fineweb

set -euo pipefail
cd "$(dirname "$0")/.."

WALLCLOCK=20:00:00
GPUS_PER_NODE=4

CONFIGS=(
    config/train_fineweb_vanilla.py
    config/train_fineweb.py
    config/train_fineweb_prism_3x.py
    config/train_fineweb_prism_2x.py
)

# Pre-flight check: don't waste a slot if the data isn't ready.
TRAIN_BIN=/capstor/scratch/cscs/lshuhao/nanogpt_data/data/fineweb/train.bin
if [[ ! -f "$TRAIN_BIN" ]]; then
    echo "ERROR: $TRAIN_BIN missing. Run 'sbatch sweeps/prep_fineweb.sh' first." >&2
    exit 1
fi

echo "=== launching FineWeb param-matched ablation ==="
echo "wallclock: $WALLCLOCK"
echo "GPUs/job:  $GPUS_PER_NODE"
echo "n_jobs:    ${#CONFIGS[@]}"
echo ""

for cfg in "${CONFIGS[@]}"; do
    echo ">>> submitting $cfg"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG="$cfg" \
        submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
