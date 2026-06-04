#!/bin/bash
# Run tools/count_flops.py inside the container env to get per-config
# forward FLOPs for both TinyStories and GPT-2-small scales.
#
# Submit::
#
#     sbatch sweeps/measure_flops.sh
#
# Watch::
#
#     tail -f logs/flops_<JOBID>.out

#SBATCH --job-name=flops
#SBATCH --output=logs/flops_%j.out
#SBATCH --error=logs/flops_%j.err
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --environment=nanogpt

set -euo pipefail
cd /workspace
export PATH="/opt/oft/.venv/bin:$PATH"

python tools/count_flops.py
