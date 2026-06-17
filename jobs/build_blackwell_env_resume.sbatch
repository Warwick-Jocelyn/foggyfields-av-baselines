#!/bin/bash
#SBATCH --job-name=bw_env2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_env2_%j.log

# NOTE: dropped `-u` from set: a conda gcc-activation hook references
# _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED unset → killed the previous job
# at stage 4 even though the install itself succeeded.
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1
export CUDA_HOME=$CONDA_PREFIX
export TCNN_CUDA_ARCHITECTURES=120        # Blackwell sm_120 (RTX PRO 6000 Blackwell)
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2

echo "[$(date)] env state at resume:"
which python && python --version
which nvcc && nvcc --version | tail -3
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[$(date)] === STAGE 5: tinycudann (HEAD — older commits do not know sm_120) ==="
python -m pip install --no-build-isolation -v \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "[$(date)] === STAGE 6: neurad-studio + gsplat fork ==="
cd ~/ETH/Code/SplatAD/neurad-studio
python -m pip install --no-build-isolation -e .

echo "[$(date)] === STAGE 7: light import (no GPU here) ==="
python -c "
import nerfstudio, importlib.util
print('nerfstudio:', nerfstudio.__file__)
print('gsplat:', importlib.util.find_spec('gsplat').origin)
"

echo "[$(date)] === STAGE 8: snapshot env ==="
mkdir -p ~/ETH/Code/SplatAD/env
DATE_TAG=$(date +%Y%m%d_%H%M)
python -m pip freeze > ~/ETH/Code/SplatAD/env/blackwell_requirements_${DATE_TAG}.txt
conda env export > ~/ETH/Code/SplatAD/env/blackwell_conda_${DATE_TAG}.yaml

echo "[$(date)] BLACKWELL_ENV_BUILD_DONE"
