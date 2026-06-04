#!/bin/bash
# Step 5 — 2-D sweep for prism-3×: learning_rate × r_init_scale.
#
# Submits len(LRS) × len(R_INITS) sbatch jobs, each with its own out_dir
# and wandb run name. Default grid is 4 × 3 = 12 jobs.
#
# Edit the LRS / R_INITS arrays below to add/remove points.
# Run from the nanogpt repo root::
#
#     bash sweeps/lr_prism_3x.sh
#
# Each run shows up in wandb (project "nanogpt") as
# "prism-3x-lr{lr}-r{r_init}". The existing baseline run
# "prism-3x-tinystories" (lr=6e-4, r_init=1.0, val_loss=1.3612) is the
# control to beat.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1

LRS=(3e-4 6e-4 1e-3 1.5e-3)
R_INITS=(0.5 1.0 2.0)

# Skip combos that match the existing baseline (lr=6e-4, r_init=1.0) so we
# don't re-pay for a run we already have. Set to 0 to submit the duplicate.
SKIP_BASELINE=1

echo "=== sweep plan ==="
echo "config:        $CONFIG"
echo "wallclock:     $WALLCLOCK"
echo "GPUs/job:      $GPUS_PER_NODE"
echo "LRs:           ${LRS[*]}"
echo "r_init_scales: ${R_INITS[*]}"
echo "skip baseline: $SKIP_BASELINE"
echo "n_jobs:        $(( ${#LRS[@]} * ${#R_INITS[@]} - SKIP_BASELINE ))"
echo ""

for lr in "${LRS[@]}"; do
    minlr=$(awk "BEGIN{printf \"%g\", $lr/10}")
    for r in "${R_INITS[@]}"; do
        if [[ "$SKIP_BASELINE" -eq 1 && "$lr" == "6e-4" && "$r" == "1.0" ]]; then
            echo "--- skipping baseline (lr=$lr, r_init=$r) — already exists"
            continue
        fi
        out_dir="out-tinystories-prism-3x-lr$lr-r$r"
        run_name="prism-3x-lr$lr-r$r"
        extra="--learning_rate=$lr --min_lr=$minlr --r_init_scale=$r --out_dir=$out_dir --wandb_run_name=$run_name"

        echo ">>> submitting lr=$lr r_init=$r"
        sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
            --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" \
            submit.sh
    done
done

echo ""
echo "=== queue ==="
squeue -u "$USER"
