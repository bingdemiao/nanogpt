#!/bin/bash
# Quick diagnostic on a trained prism checkpoint.
#
# Submit::
#
#     sbatch sweeps/check_norm_growth.sh
#
#     # or with a custom ckpt:
#     sbatch --export=ALL,CKPT=/path/to/ckpt.pt,DATASET=fineweb sweeps/check_norm_growth.sh

#SBATCH --job-name=normgrowth
#SBATCH --output=logs/normgrowth_%j.out
#SBATCH --error=logs/normgrowth_%j.err
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --environment=nanogpt

set -euo pipefail
cd /workspace
export PATH="/opt/oft/.venv/bin:$PATH"

CKPT="${CKPT:-/capstor/scratch/cscs/lshuhao/nanogpt_out/out-tinystories-prism-3x-lr1.5e-3-r0.5/ckpt.pt}"
DATASET="${DATASET:-tinystories}"

python tools/check_norm_growth.py --ckpt "$CKPT" --dataset "$DATASET" --n_batches 4
