# Shared environment for all SplatAD-NFF jobs. Source inside the container.
export W=/capstor/scratch/cscs/yitiwang/splatad-nff-repro
export VENV=$W/venv
export REPO=/users/yitiwang/projects/foggyfields-av-baselines
export CODE=$W/code
export DATA_ROOT=$W/data/NVIDIA_AV_Fog
export OUT_ROOT=$W/outputs
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="9.0"
export TCNN_CUDA_ARCHITECTURES="90"
export TORCH_EXTENSIONS_DIR=$W/cache/torch_extensions
export PIP_CACHE_DIR=$W/cache/pip
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
export PIP_NO_INPUT=1
export XDG_CACHE_HOME=$W/cache
export TORCH_HOME=$W/cache/torch
export TORCHDYNAMO_DISABLE=1
export MAX_JOBS=32
export WANDB_PROJECT=foggyfields-nff
export WANDB_DIR=$W/cache/wandb
mkdir -p "$WANDB_DIR"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
