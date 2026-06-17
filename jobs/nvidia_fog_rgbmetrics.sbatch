#!/bin/bash
#SBATCH --job-name=nvfog_rgbm
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=00:20:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/rgbmetrics_%j.log

# Compute PSNR/SSIM/LPIPS from rendered val PNGs for a SplatAD run.
#   sbatch --export=ALL,METHOD=splatad,SEQ=004 nvidia_fog_rgbmetrics.sbatch
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1

METHOD="${METHOD:?set METHOD}"; SEQ="${SEQ:?set SEQ}"
SPLIT_DIR="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/${METHOD}/${SEQ}/renders/val"
OUT="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/${METHOD}/${SEQ}/rgb_metrics.json"
echo "[$(date)] rgb metrics $METHOD/$SEQ from $SPLIT_DIR"
cd ~/ETH/Code/SplatAD/jobs
python compute_rgb_metrics.py "$SPLIT_DIR" --output "$OUT"
