#!/bin/bash
#SBATCH --job-name=splatad_smoke
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=shard:L40:16        # 1/4 of an L40 (~12 GB) — bypass the "all full L40 taken" queue
#SBATCH --time=00:45:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/smoke_train_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad
export PYTHONNOUSERSITE=1
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"

echo "[$(date)] node=$(hostname)"
nvidia-smi -L | head -2
python -c "import torch, tinycudann, gsplat; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'tcnn ok')"

TEST_ROOT=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset
OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/smoke_011

mkdir -p "$OUT_DIR"
cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] launching SplatAD smoke (200 steps) on scene 011"
ns-train splatad \
    --output-dir "$OUT_DIR" \
    --vis tensorboard \
    --max-num-iterations 200 \
    --steps-per-save 200 \
    --steps-per-eval-batch 100 \
    --steps-per-eval-image 200 \
    pandaset-data \
    --data "$TEST_ROOT" \
    --sequence 011

echo "[$(date)] DONE — check $OUT_DIR"
