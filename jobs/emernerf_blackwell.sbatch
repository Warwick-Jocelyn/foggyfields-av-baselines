#!/bin/bash
#SBATCH --job-name=bw_emernerf
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_emernerf_%j.log

# Phase 0.2-extension — EmerNeRF (RGB-only) baseline on PandaSet 011/078.
# Matches our SplatAD/NeuRAD protocol: 30k iter, test_image_stride=2 (= 0.5 split).
#
# Usage:
#   sbatch --export=ALL,SCENE=011 emernerf_blackwell.sbatch
#   sbatch --export=ALL,SCENE=078 emernerf_blackwell.sbatch

SCENE="${SCENE:?must set SCENE=... via --export=ALL,SCENE=...}"
NUM_ITERS="${NUM_ITERS:-30000}"
# CONFIG = "pandaset" (RGB-only) or "pandaset_lidar" (RGB + LiDAR depth sup, paper default)
CONFIG="${CONFIG:-pandaset}"
RUN_SUFFIX="${RUN_SUFFIX:-6cam_dyn_flow}"

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1
: "${NVCC_PREPEND_FLAGS:=}"; export NVCC_PREPEND_FLAGS
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"          # Blackwell sm_120 for tcnn
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

OUT_ROOT=/networkhome/WMGDS/wang3_y/ETH/Code/outputs/emernerf_${SCENE}_split05
mkdir -p "$OUT_ROOT"

echo "[$(date)] node=$(hostname) scene=$SCENE iter=$NUM_ITERS"
nvidia-smi -L

cd ~/ETH/Code/EmerNeRF

# scene_idx is the numeric form (011 → 11) — PandaSetDataset zero-pads back to 3 digits.
SCENE_INT=$(echo "$SCENE" | sed 's/^0*//')

echo "[$(date)] === STEP 1: train_emernerf.py (${NUM_ITERS} iter, ${CONFIG}.yaml, 0.5 split) ==="
python train_emernerf.py \
    --config_file configs/${CONFIG}.yaml \
    --output_root "$OUT_ROOT" \
    --project foggyfields_emernerf \
    --run_name "scene_${SCENE}_${RUN_SUFFIX}" \
    --enable_wandb \
    --entity haonan_zhao \
    data.scene_idx=${SCENE_INT} \
    data.start_timestep=0 \
    data.end_timestep=-1 \
    logging.saveckpt_freq=${NUM_ITERS} \
    optim.num_iters=${NUM_ITERS}

LOG_DIR=$(find "$OUT_ROOT" -type d -name "scene_${SCENE}" | sort | tail -1)
echo "[$(date)] log_dir=$LOG_DIR"

echo "[$(date)] === STEP 2: aggregate metrics + update comparison table ==="
python ~/ETH/Code/SplatAD/jobs/emernerf_collect_metrics.py "$LOG_DIR" "$SCENE" || true

echo "[$(date)] DONE — outputs:"
ls -lh "$LOG_DIR"/metrics/ 2>/dev/null | tail -5
ls -lh "$LOG_DIR"/videos/ 2>/dev/null | tail -10
