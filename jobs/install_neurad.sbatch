#!/bin/bash
#SBATCH --job-name=ns_install
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/ns_install_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad
export PYTHONNOUSERSITE=1
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"      # gsplat fork uses this var
export MAX_JOBS=2                       # avoid OOM in parallel nvcc

echo "[$(date)] node=$(hostname)"
nvcc --version | tail -3
# Note: skip tcnn import here — tcnn probes the GPU at import time and this is a CPU-only node
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[$(date)] installing neurad-studio (will also build carlinds/splatad gsplat fork)"
cd ~/ETH/Code/SplatAD/neurad-studio
pip install --no-build-isolation -e .

echo "[$(date)] light import test (no GPU here, so skip gsplat/tcnn whose __init__ touch the device)"
python -c "
import nerfstudio
print('nerfstudio:', nerfstudio.__file__)
import importlib.util
print('gsplat module path:', importlib.util.find_spec('gsplat').origin)
"

echo "[$(date)] DONE — full GPU import verification will happen at smoke-test job"
