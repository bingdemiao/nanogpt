#!/bin/bash
# Step 5d — learnable ar_scale (LoRA-α-style fix for gate saturation).
#
# Hypothesis: with ar_scale_init=0.1, gate inputs at iter 0 are ~10×
# smaller, putting SiLU in its informative band where SGD has good
# gradients. The scale is learnable, so SGD can adjust over training.
#
# Sweeping ar_scale_init ∈ {0.05, 0.1, 0.25} at the tuned recipe.
# Baseline to beat: prism-3x-lr1.5e-3-r0.5 (val 1.3100).
#
# Requires: cute_prism with ar_scale_init support (our patched version
# loaded via PYTHONPATH=/opt/cute_mma/src in submit.sh). Confirmed working
# via job 3427098 smoke test.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

LR=1.5e-3
MIN_LR=1.5e-4
R_INIT=0.5

ARSCALES=(0.05 0.1 0.25)

echo "=== ar_scale sweep plan ==="
echo "config:    $CONFIG"
echo "fixed LR:  $LR (min $MIN_LR)"
echo "fixed r_init_scale: $R_INIT"
echo "ar_scale_inits: ${ARSCALES[*]}"
echo "n_jobs:    ${#ARSCALES[@]}"
echo ""

for arscale in "${ARSCALES[@]}"; do
    out_dir="out-tinystories-prism-3x-arscale$arscale"
    run_name="prism-3x-arscale$arscale"
    extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --prism_ar_scale_init=$arscale --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> submitting ar_scale_init=$arscale"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
