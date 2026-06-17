#!/bin/bash
#SBATCH --job-name=bw_env
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_env_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=splatad-blackwell

echo "[$(date)] === STAGE 1: recreate $ENV_NAME ==="
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate "$ENV_NAME"
echo "active env: $(which python)"

echo "[$(date)] === STAGE 2: pip baseline ==="
python -m pip install --upgrade pip "setuptools<70.0"
python -m pip install typing_extensions networkx sympy filelock numpy pillow dill huggingface_hub requests ninja

echo "[$(date)] === STAGE 3: PyTorch 2.7.0 + cu128 (Blackwell needs cu12.8+) ==="
python -m pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "[$(date)] === STAGE 4: CUDA toolkit 12.8 (nvcc) ==="
conda install -y -c "nvidia/label/cuda-12.8.0" cuda-toolkit
nvcc --version | tail -3

export CUDA_HOME=$CONDA_PREFIX
export TCNN_CUDA_ARCHITECTURES=120        # Blackwell sm_120 (RTX PRO 6000 Blackwell)
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2

echo "[$(date)] === STAGE 5: tinycudann (HEAD — older commits do not know sm_120) ==="
python -m pip install --no-build-isolation -v \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "[$(date)] === STAGE 6: neurad-studio + gsplat fork (will recompile gsplat for cu128) ==="
cd ~/ETH/Code/SplatAD/neurad-studio
python -m pip install --no-build-isolation -e .

echo "[$(date)] === STAGE 7: light import (no GPU here — defer tcnn/gsplat verification to smoke job) ==="
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
