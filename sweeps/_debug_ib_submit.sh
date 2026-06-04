#!/bin/bash
#SBATCH --job-name=ib_nan_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --environment=nanogpt
#SBATCH --time=00:20:00

set -euo pipefail
cd /workspace

export VIRTUAL_ENV=/opt/oft/.venv
export CUDA_HOME=/usr/local/cuda
export PATH="/opt/oft/.venv/bin:${CUDA_HOME}/bin:$PATH"
export PYTHONPATH="/opt/cute_mma/src:${PYTHONPATH:-}"
export TORCH_HOME=/users/lshuhao/.cache/torch
export CUTE_PRISM_CACHE_DIR=${CUTE_PRISM_CACHE_DIR:-/tmp/cute_prism_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
mkdir -p "$UV_CACHE_DIR" "$CUTE_PRISM_CACHE_DIR"
unset CC CXX CUDAHOSTCXX
export LD_LIBRARY_PATH="/root/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/lib:${LD_LIBRARY_PATH:-}"

python -c "import cute_prism" 2>/dev/null || \
    uv pip install --no-build-isolation -e "/opt/cute_mma[mup]"

python tools/debug_internal_bias_nan.py
