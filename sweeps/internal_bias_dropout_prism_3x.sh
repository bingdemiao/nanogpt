#!/bin/bash
# Step 5e — sweep internal_bias × dropout at the best-so-far prism config.
#
# Best previous config:
#   LR=1.5e-3, min_lr=1.5e-4, r_init_scale=0.5,
#   mlp_expansion=3, group_size=32, reconn_sz=16, input_shuffle=True
#
# What we vary here:
#   prism_internal_bias ∈ {False, True}
#   dropout             ∈ {0.0, 0.05, 0.1}
# = 6 runs.
#
# max_iters is reduced from 50000 → 30000 (still ~4 epochs over TinyStories,
# which is enough to compare variants since the relative ordering of
# variants stabilizes early).
#
# Wallclock: gs=32, r=16 ran 50k iters in 1h 8min — so 30k iters ≈ 40 min
# per run. Six runs parallel = ~40 min total wallclock.
#
# Run from the nanogpt repo root::
#
#     bash sweeps/internal_bias_dropout_prism_3x.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=04:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000

# Best-found config (locked).
LR=1.5e-3
MIN_LR=1.5e-4
R_INIT=0.5
GS=32
RSZ=16
SHUFFLE=True

# What we sweep.
IBS=(False True)
DROPOUTS=(0.0 0.05 0.1)

echo "=== sweep plan ==="
echo "config base:        $CONFIG"
echo "fixed:              LR=$LR, r_init_scale=$R_INIT, gs=$GS, r=$RSZ, input_shuffle=$SHUFFLE"
echo "max_iters:          $MAX_ITERS"
echo "internal_bias vals: ${IBS[*]}"
echo "dropout vals:       ${DROPOUTS[*]}"
echo "n_jobs:             $(( ${#IBS[@]} * ${#DROPOUTS[@]} ))"
echo ""

for ib in "${IBS[@]}"; do
    for d in "${DROPOUTS[@]}"; do
        # Build a compact tag like "ibT-d0.1" or "ibF-d0.05".
        ib_tag=$([ "$ib" = "True" ] && echo "ibT" || echo "ibF")
        d_tag="d${d}"
        out_dir="out-tinystories-prism-3x-${ib_tag}-${d_tag}"
        run_name="prism-3x-${ib_tag}-${d_tag}"
        extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$RSZ --prism_input_shuffle=$SHUFFLE --prism_internal_bias=$ib --dropout=$d --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
        echo ">>> submitting internal_bias=$ib  dropout=$d"
        sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
            --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
    done
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
