#!/bin/bash
# Run the three thesis-analysis tools in one batch job.
#
# Provide checkpoint paths via env vars or edit the defaults below.
# Outputs land in /workspace/tools/out_*/  (your local checkout).
#
# Submit::
#
#     # Defaults — best TinyStories runs:
#     sbatch sweeps/run_analysis_tools.sh
#
#     # Custom:
#     sbatch \\
#         --export=ALL,VANILLA_CKPT=/path/A,PRISM_CKPT=/path/B \\
#         sweeps/run_analysis_tools.sh

#SBATCH --job-name=analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --environment=nanogpt

set -euo pipefail
cd /workspace
export PATH="/opt/oft/.venv/bin:$PATH"

# Defaults: the tuned TinyStories runs.
VANILLA_CKPT="${VANILLA_CKPT:-/capstor/scratch/cscs/lshuhao/nanogpt_out/out-tinystories-vanilla-lr2e-3/ckpt.pt}"
PRISM_CKPT="${PRISM_CKPT:-/capstor/scratch/cscs/lshuhao/nanogpt_out/out-tinystories-prism-3x-lr1.5e-3-r0.5/ckpt.pt}"
DATASET="${DATASET:-tinystories}"

echo "VANILLA_CKPT=$VANILLA_CKPT"
echo "PRISM_CKPT=$PRISM_CKPT"
echo "DATASET=$DATASET"
echo ""

# --- 1. Effective rank ---------------------------------------------------
echo "================ effective_rank.py ================"
python tools/effective_rank.py \
    --ckpt vanilla "$VANILLA_CKPT" \
    --ckpt prism   "$PRISM_CKPT" \
    --output_dir tools/out_effective_rank

# --- 2. Contribution ratio (prism only) ---------------------------------
echo ""
echo "================ contribution_ratio.py ================"
python tools/contribution_ratio.py \
    --ckpt "$PRISM_CKPT" \
    --dataset "$DATASET" \
    --n_batches 8 --batch_size 16 \
    --output_dir tools/out_contribution_ratio

# --- 3. Inference benchmark ---------------------------------------------
echo ""
echo "================ inference_bench.py ================"
python tools/inference_bench.py \
    --ckpt vanilla:"$VANILLA_CKPT" \
    --ckpt prism:"$PRISM_CKPT" \
    --batch_sizes 1,4,16,32 \
    --warmup 20 --iters 100

echo ""
echo "done. outputs in tools/out_*/"
