#!/bin/bash
#SBATCH --job-name=tcnn_build
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/tcnn_build_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad
export PYTHONNOUSERSITE=1
export CUDA_HOME=$CONDA_PREFIX
export TCNN_CUDA_ARCHITECTURES=89
export MAX_JOBS=2

echo "[$(date)] node=$(hostname)  (CPU-only build; arch pinned to sm_89)"
nvcc --version | tail -3
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[$(date)] starting tcnn pip install"
pip install --no-build-isolation -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "[$(date)] import test deferred to a GPU job (cannot import on CPU node)"
echo "[$(date)] DONE"
