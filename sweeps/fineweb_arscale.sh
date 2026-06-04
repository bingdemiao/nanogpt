#!/bin/bash
# Step 6 follow-up — verify ar_scale fix at GPT-2 / FineWeb scale.
#
# Launches a single FineWeb prism-3× run with prism_ar_scale_init=0.1
# (everything else identical to train_fineweb_prism_3x.py).
#
# Prerequisite:
#   1. tools/check_norm_growth.py + tools/contribution_ratio.py confirm
#      ar_scale fix works at TinyStories (val_loss drops below 1.31 with
#      gate_in_band_frac > 0.3).
#   2. PYTHONPATH=/opt/cute_mma/src is set in submit.sh (already done) so
#      the patched cute_prism with ar_scale_init support is loaded.
#
# Cost: ~17-20 h wallclock on 4× GH200.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/fineweb_arscale.sh
#
# Compare in wandb (project "nanogpt"):
#   nanogpt-vanilla-fineweb           (val 2.9653)
#   nanogpt-prism-3x-fineweb          (no ar_scale, baseline)
#   nanogpt-prism-3x-arscale-fineweb  (this run)

set -euo pipefail
cd "$(dirname "$0")/.."

# Pre-flight: tokenized FineWeb data must exist.
TRAIN_BIN=/capstor/scratch/cscs/lshuhao/nanogpt_data/data/fineweb/train.bin
if [[ ! -f "$TRAIN_BIN" ]]; then
    echo "ERROR: $TRAIN_BIN missing. Run 'sbatch sweeps/prep_fineweb.sh' first." >&2
    exit 1
fi

echo "=== launching FineWeb prism-3× + ar_scale=0.1 ==="
sbatch --time=20:00:00 --gpus-per-node=4 \
    --export=ALL,CONFIG=config/train_fineweb_prism_3x_arscale.py \
    submit.sh

echo ""
echo "=== queue ==="
squeue -u "$USER"
