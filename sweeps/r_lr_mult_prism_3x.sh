#!/bin/bash
# Step 5c — slow R's learning rate to keep it from saturating the gate.
#
# Diagnostic finding (tools/check_norm_growth.py @ iter 50000):
#   R.std grew 4.03× from init (0.177 → 0.712)
#   LN γ stayed at 1.0, A.std stayed at 1.0
#   AR.std blew up to 16, putting SiLU gates deep in linear regime
#
# Hypothesis: lowering r_lr_mult constrains R growth, keeps AR.std in the
# [1, 4] target band, gates stay in the informative middle (target
# `gate_in_band_frac > 0.4` from tools/contribution_ratio.py).
#
# Baseline to beat: prism-3x-lr1.5e-3-r0.5 (r_lr_mult=1.0 default) → val 1.3100
# Run from the nanogpt repo root::
#
#     bash sweeps/r_lr_mult_prism_3x.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

LR=1.5e-3
MIN_LR=1.5e-4
R_INIT=0.5

# Sweep r_lr_mult only. Defaults: r_lr_mult=1.0 already exists (val=1.3100).
RMULTS=(0.1 0.25 0.5)

echo "=== r_lr_mult sweep plan ==="
echo "config:      $CONFIG"
echo "fixed LR:    $LR (min $MIN_LR)"
echo "fixed r_init: $R_INIT"
echo "r_lr_mults:  ${RMULTS[*]}"
echo "n_jobs:      ${#RMULTS[@]}"
echo ""

for rmult in "${RMULTS[@]}"; do
    out_dir="out-tinystories-prism-3x-rmult$rmult"
    run_name="prism-3x-rmult$rmult"
    extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --r_lr_mult=$rmult --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> submitting r_lr_mult=$rmult"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
