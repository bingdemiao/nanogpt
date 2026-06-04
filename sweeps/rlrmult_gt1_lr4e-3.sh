#!/bin/bash
# Step 13 — r_lr_mult > 1 probe at base LR=4e-3, (gs=32, r=32).
#
# Motivation: the update-ratio instrumentation (jobs 3523560-62) shows R's
# relative step ||ΔR||/||R|| is only ~6% of B's at equal LR (r_lr_mult=1.0)
# -> R is the chronic under-learner. Earlier r_lr_mult sweeps only went
# BELOW 1.0 (all worse). This probes ABOVE 1.0: let R move faster than B.
#
# Base LR = 4e-3 (B's LR). r_lr_mult scales R's LR:
#   r_lr_mult=1.5 -> R=6e-3
#   r_lr_mult=2.0 -> R=8e-3
#   r_lr_mult=4.0 -> R=16e-3   (aggressive; NaN early-stop guards it)
#
# Direct comparison with the existing base=4e-3 r_lr_mult<=1 curve (min val):
#   0.25->1.3132  0.375->1.3070  0.625->1.2990  1.0->1.2919
# If the optimum is at r_lr_mult>1, R was under-trained and benefits from a
# higher LR than B (despite the earlier prior that R should be slower).
#
# Fixed: gs=32, r=32, r_init=0.5, max_iters=30000, no shuffle, ibF, dropout=0.
# Runs also log update_ratio/{R,B_up,R_over_B} so we can see whether R/B
# moves toward 1 as r_lr_mult rises.
#
#     bash sweeps/rlrmult_gt1_lr4e-3.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=config/train_tinystories_prism_3x.py
WALLCLOCK=06:00:00
GPUS_PER_NODE=1
MAX_ITERS=30000
LR=4e-3
MIN_LR=4e-4
R_INIT=0.5
GS=32
RSZ=32

MULTS=(1.5 2.0 4.0)

echo "=== r_lr_mult>1 probe at base LR=$LR, gs=$GS r=$RSZ ==="
echo "control (r_lr_mult=1.0) = 1.2919 (min val, 30k)"
echo "r_lr_mult values: ${MULTS[*]}  -> R LR = ${LR}*mult"
echo ""

for mult in "${MULTS[@]}"; do
    mult_tag=$(echo "$mult" | tr -d '.')
    out_dir="out-tinystories-prism-3x-gs${GS}-r${RSZ}-lr4e-3-rmult${mult}"
    run_name="prism-3x-gs${GS}-r${RSZ}-lr4e-3-rmult${mult}"
    extra="--learning_rate=$LR --min_lr=$MIN_LR --r_init_scale=$R_INIT --group_size=$GS --reconn_sz=$RSZ --r_lr_mult=$mult --max_iters=$MAX_ITERS --lr_decay_iters=$MAX_ITERS --out_dir=$out_dir --wandb_run_name=$run_name"
    echo ">>> r_lr_mult=$mult  (R LR = $(awk "BEGIN{printf \"%.1e\", 4e-3*$mult}"))"
    sbatch --time="$WALLCLOCK" --gpus-per-node="$GPUS_PER_NODE" \
        --export=ALL,CONFIG=$CONFIG,EXTRA="$extra" submit.sh
done

echo ""
squeue -u "$USER" 2>/dev/null | tail -8
