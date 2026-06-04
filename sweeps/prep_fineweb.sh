#!/bin/bash
# Run once before any FineWeb training. Downloads HuggingFaceFW/fineweb-edu
# sample-10BT and writes train.bin (~20GB) + val.bin (~10MB) to
# /workspace/data/fineweb/  (symlinked to /capstor/scratch/cscs/lshuhao/
# nanogpt_data/data/fineweb/, so plenty of space).
#
# ~30-60 min total: ~5-10 min download + ~20-40 min tokenization with 8
# workers. HF_TOKEN is sourced from /workspace/.env.
#
# Submit from the host shell::
#
#     sbatch sweeps/prep_fineweb.sh
#
# Watch progress with::
#
#     tail -f logs/prep-fineweb_<JOBID>.out

#SBATCH --job-name=prep-fineweb
#SBATCH --output=logs/prep-fineweb_%j.out
#SBATCH --error=logs/prep-fineweb_%j.err
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --environment=nanogpt
#SBATCH --open-mode=append

set -euo pipefail

cd /workspace

export PATH="/opt/oft/.venv/bin:$PATH"
export HF_HOME=${HF_HOME:-/tmp/hf-cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

[[ -f /workspace/.env ]] && set -a && source /workspace/.env && set +a

# oft.sqfs ships numpy/tqdm/huggingface_hub but not `datasets` or
# `tiktoken`. The squashfs rootfs is read-only so installs are per-job;
# uv cache makes the second run cheap.
python -c "import datasets, tiktoken" 2>/dev/null || \
    uv pip install datasets tiktoken

python -c "import datasets, tiktoken, numpy, tqdm; print('deps ok')"

echo "[prep_fineweb] starting at $(date)"
python data/fineweb/prepare.py
echo "[prep_fineweb] done at $(date)"

echo "=== output ==="
ls -lh data/fineweb/
