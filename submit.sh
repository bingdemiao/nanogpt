#!/bin/bash
#SBATCH --job-name=nanogpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=g34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=nanogpt
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --open-mode=append
#SBATCH --requeue

# Slurm wrapper for nanogpt/train.py. Mirrors oft/submit.sh:
#  * uses the `nanogpt` container env (squashfs at ~/.edf/nanogpt.toml,
#    same image as oft.sqfs but with /users/lshuhao/nanogpt mounted)
#  * forwards SIGUSR1 to the python process for graceful checkpoint+requeue
#  * launches with torchrun for DDP
#
# Usage::
#
#     sbatch --time=4:00:00 \
#         --export=ALL,CONFIG=config/train_fineweb.py,EXTRA="--batch_size=8" \
#         submit.sh
#
# CONFIG     — path to a config/train_*.py file (default: config/train_shakespeare_char.py)
# EXTRA      — extra --key=value flags forwarded to train.py
# NPROC      — torchrun --nproc_per_node (default: $SLURM_GPUS_PER_NODE)

set -euo pipefail

# The `nanogpt` container env mounts the repo at /workspace (not the host path).
cd /workspace
mkdir -p logs

# Activate the container venv (mirrors oft/run.sh).
export VIRTUAL_ENV=/opt/oft/.venv
export CUDA_HOME=/usr/local/cuda
export PATH="/opt/oft/.venv/bin:${CUDA_HOME}/bin:$PATH"
# Prefer the toml-mounted host cute_mma (with our ar_scale + input_shuffle
# work) over the container's baked-in copy at /users/lshuhao/cute_mma.
export PYTHONPATH="/opt/cute_mma/src:${PYTHONPATH:-}"
export TORCH_HOME=/users/lshuhao/.cache/torch
export CUTE_PRISM_CACHE_DIR=${CUTE_PRISM_CACHE_DIR:-/tmp/cute_prism_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
export HF_HOME=${HF_HOME:-/tmp/hf-cache}
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$CUTE_PRISM_CACHE_DIR"
unset CC CXX CUDAHOSTCXX
export LD_LIBRARY_PATH="/root/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/lib:${LD_LIBRARY_PATH:-}"

# Source WANDB_API_KEY / WANDB_ENTITY / HF_TOKEN from the repo's .env.
[[ -f /workspace/.env ]] && set -a && source /workspace/.env && set +a

# Make sure cute_prism is installed in the container venv (lazy install
# from the live mount at /opt/cute_mma).
python -c "import cute_prism" 2>/dev/null || \
    uv pip install --no-build-isolation -e "/opt/cute_mma[mup]"
# muP package (only needed when use_mup=True; cheap no-op if already present).
python -c "import mup" 2>/dev/null || uv pip install mup

CONFIG=${CONFIG:-config/train_shakespeare_char.py}
EXTRA=${EXTRA:-}
NPROC=${NPROC:-${SLURM_GPUS_PER_NODE:-1}}

# Auto-resume from checkpoint on requeue.
# Parse out_dir from the config; if EXTRA contains --out_dir=X it wins, so
# sweep runs (one base config + EXTRA overrides) each check their own dir.
# Skip auto-resume if user explicitly passes --init_from=... in EXTRA.
OUT_DIR=$(grep -oP '^out_dir\s*=\s*"\K[^"]+' "$CONFIG" 2>/dev/null || true)
EXTRA_OUT_DIR=$(echo " $EXTRA " | grep -oP -- '--out_dir=\K\S+' | tail -1 || true)
if [[ -n "${EXTRA_OUT_DIR:-}" ]]; then
    OUT_DIR="$EXTRA_OUT_DIR"
fi
if echo " $EXTRA " | grep -q -- '--init_from='; then
    echo "[submit.sh] --init_from override in EXTRA; skipping auto-resume check"
elif [[ -n "${OUT_DIR:-}" && -f "$OUT_DIR/ckpt.pt" ]]; then
    echo "[submit.sh] existing checkpoint at $OUT_DIR/ckpt.pt — resuming"
    EXTRA="$EXTRA --init_from=resume"
fi

cleanup() {
    echo "[submit.sh] caught SIGUSR1; relaying to python and requeuing job $SLURM_JOB_ID"
    kill -SIGUSR1 "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap cleanup SIGUSR1

# Proxy unset for wandb (matches oft/run.sh)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

torchrun --standalone --nproc_per_node="$NPROC" \
    train.py "$CONFIG" $EXTRA &
child=$!
wait "$child"
