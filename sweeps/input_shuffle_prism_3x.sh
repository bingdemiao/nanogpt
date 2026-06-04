#!/bin/bash
# Test input_shuffle on prism-3× at the tuned recipe.
#
# input_shuffle requires reconn_sz=16 (vs the default 8). To attribute
# any improvement we run two ablations:
#
#   A. reconn_sz=16, input_shuffle=False  → isolates the "bigger R" effect
#   B. reconn_sz=16, input_shuffle=True   → adds the per-group butterfly
#
# Compared against existing prism-3x-lr1.5e-3-r0.5 (reconn=8, val 1.3100)
# and vanilla 4× (val 1.2145), we'll be able to say:
#   - if A alone helps  → R capacity was bottlenecking
#   - if B helps over A → cross-group view diversity matters
#   - if neither helps  → shuffle isn't the missing piece either
#
# Uses the best LR/init from Step 5a (lr=1.5e-3, r_init_scale=0.5).
#
# Run from the nanogpt repo root::
#
#     bash sweeps/input_shuffle_prism_3x.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

LR=1.5e-3
MIN_LR=1.5e-4
R_INIT=0.5

# ----- A. reconn=16, no shuffle -----
extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --reconn_sz=16 --out_dir=out-tinystories-prism-3x-r16-noshuf --wandb_run_name=prism-3x-r16-noshuf"
echo ">>> A: reconn_sz=16, input_shuffle=False"
sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
    --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh

# ----- B. reconn=16 + input_shuffle -----
extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --reconn_sz=16 --prism_input_shuffle=True --out_dir=out-tinystories-prism-3x-r16-shuf --wandb_run_name=prism-3x-r16-shuf"
echo ">>> B: reconn_sz=16, input_shuffle=True"
sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
    --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh

echo ""
echo "=== queue ==="
squeue -u "$USER"
