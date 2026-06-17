#!/bin/bash
#SBATCH --job-name=bw_nvccwrap
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_nvccwrap_%j.log

# Resume from previous bw_fixcuda (env has torch 2.7+cu128 and nvcc 12.8 already).
# Install an nvcc wrapper that dedupes double-`-c` (PyTorch 2.7 + tcnn passes both),
# then complete stages 5-8.

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1

ENV_BIN=$CONDA_PREFIX/bin
WRAP=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/bin/nvcc

echo "[$(date)] === STAGE B: install nvcc wrapper ==="
echo "before: $(readlink -f $ENV_BIN/nvcc)"
if [ ! -e "$ENV_BIN/nvcc.real" ]; then
    mv "$ENV_BIN/nvcc" "$ENV_BIN/nvcc.real"
fi
ln -sf "$WRAP" "$ENV_BIN/nvcc"
echo "after: $(ls -la $ENV_BIN/nvcc)"
echo "real:  $(ls -la $ENV_BIN/nvcc.real)"

# sanity test the wrapper: feed it `-c -c -o /tmp/foo file.cu` and verify dedupe
echo "wrapper trace (REAL_NVCC will fail because no input):"
bash -x "$WRAP" -c -c --version 2>&1 | head -5 || true

nvcc --version | tail -3

export CUDA_HOME=$CONDA_PREFIX
export TCNN_CUDA_ARCHITECTURES=120
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2

python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

echo "[$(date)] === STAGE 5: tinycudann (HEAD, sm_120) — with nvcc dedupe wrapper ==="
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
