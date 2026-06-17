#!/bin/bash
#SBATCH --job-name=bw_fixcuda
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_fixcuda_%j.log

# PROBLEM the previous bw_env2 hit:
#   conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit  → actually installed CUDA 13.2
#   (NVIDIA appears to have repointed the label to "latest")
# PyTorch was built with cu12.8, so cpp_extension._check_cuda_version refused 13.2.
# FIX: explicitly pin cuda-toolkit to 12.8.* from the plain `nvidia` channel.

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1

echo "[$(date)] === STAGE A: replace cuda-13.2 with cuda-toolkit 12.8.* ==="
echo "before:"; nvcc --version | tail -3 || true
conda install -y -c nvidia 'cuda-toolkit=12.8.*'
echo "after:"; nvcc --version | tail -3

# Now nvcc must report 12.8.x
NVCC_MAJ_MIN=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
if [ "$NVCC_MAJ_MIN" != "12.8" ]; then
    echo "ERROR: nvcc reports $NVCC_MAJ_MIN, expected 12.8"
    exit 2
fi

export CUDA_HOME=$CONDA_PREFIX
export TCNN_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2

python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

echo "[$(date)] === STAGE 5: tinycudann (HEAD, sm_120) ==="
python -m pip install --no-build-isolation -v \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "[$(date)] === STAGE 6: neurad-studio + gsplat fork ==="
cd ~/ETH/Code/SplatAD/neurad-studio
python -m pip install --no-build-isolation -e .

echo "[$(date)] === STAGE 7: light import ==="
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
