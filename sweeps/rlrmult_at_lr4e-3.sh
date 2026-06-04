#!/bin/bash
# Step 10 — r_lr_mult sweep at the new base LR=4e-3, the LR-sweep winner
# at (gs=32, r=32).
#
# Decouples "B needs bigger LR" vs "R needs bigger LR" — both groups
# share LR at r_lr_mult=1.0 (the LR-sweep default), so the gain from
# LR=1.5e-3 → 4e-3 could be from either or both. By holding B at 4e-3
# and dropping R's LR, we isolate which group's higher LR did the work.
#
#   r_lr_mult  R's effective LR  what it tests
#   ─────────  ────────────────  ─────────────────────────────────────
#   0.25       1.0e-3            sharp slow-down of R, fast B
#   0.375      1.5e-3            R held at OLD inherited LR, B raised
#   0.625      2.5e-3            partial slow-down
#   1.0        4.0e-3            control — both at LR* (job 3501275, val 1.2940)
#
# All else identical to LR-sweep: gs=32, r=32, r_init=0.5, max_iters=30000,
# no shuffle, ibF, dropout=0.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=08:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000

LR=4e-3
MIN_LR=4e-4
R_INIT=0.5
GS=32
RSZ=32

MULTS=(0.25 0.375 0.625)

echo "=== sweep plan ==="
echo "base LR:   $LR   (control LR with r_lr_mult=1.0 → job 3501275, val 1.2940)"
echo "min_lr:    $MIN_LR"
echo "shape:     gs=$GS  r=$RSZ"
echo "r_lr_mult: ${MULTS[*]}"
echo ""

for mult in "${MULTS[@]}"; do
    mult_tag=$(echo "$mult" | tr -d '.')
    out_dir="out-tinystories-prism-3x-gs${GS}-r${RSZ}-lr4e-3-rmult${mult}"
    run_name="prism-3x-gs${GS}-r${RSZ}-lr4e-3-rmult${mult}"
    extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$RSZ --r_lr_mult=$mult --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> r_lr_mult=$mult  (R at $(echo "$LR * $mult" | bc -l | head -c 8))"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

squeue -u "$USER" 2>/dev/null | tail -10
