#!/bin/bash
#SBATCH --job-name=bw_render
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_render_%j.log

# Usage:  sbatch --export=ALL,SCENE=011  smoke_render_blackwell.sbatch
#         sbatch --export=ALL,SCENE=078  smoke_render_blackwell.sbatch
SCENE="${SCENE:?must set SCENE=011 or SCENE=078 via --export}"

# drop `-u`: conda's cuda-nvcc activation hook references NVCC_PREPEND_FLAGS unset
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1
: "${NVCC_PREPEND_FLAGS:=}"; export NVCC_PREPEND_FLAGS

# conda cuda-toolkit puts CUDA headers/libs under $CONDA_PREFIX/targets/x86_64-linux/{include,lib}
# torch's cpp_extension JIT only adds $CONDA_PREFIX/include → cuda_runtime_api.h not found
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"

echo "[$(date)] node=$(hostname)  scene=$SCENE"
nvidia-smi -L
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda_ok', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0))
import tinycudann; print('tcnn ok')
import gsplat
from gsplat.rendering import rasterization, lidar_rasterization
print('gsplat + lidar_rasterization ok')
"

TEST_ROOT=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset
OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/blackwell_${SCENE}

mkdir -p "$OUT_DIR"
cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === train SplatAD on scene $SCENE (2000 steps, save eval renders) ==="
ns-train splatad \
    --output-dir "$OUT_DIR" \
    --vis tensorboard \
    --max-num-iterations 2000 \
    --steps-per-save 1000 \
    --steps-per-eval-batch 500 \
    --steps-per-eval-image 1000 \
    --steps-per-eval-all-images 2000 \
    pandaset-data \
    --data "$TEST_ROOT" \
    --sequence "$SCENE"

echo "[$(date)] DONE — outputs at $OUT_DIR"
ls -la "$OUT_DIR"
find "$OUT_DIR" -name "*.png" -o -name "*.jpg" 2>/dev/null | head -20
