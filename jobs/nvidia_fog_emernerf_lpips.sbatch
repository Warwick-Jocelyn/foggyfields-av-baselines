#!/bin/bash
#SBATCH --job-name=nvfog_emerlpips
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/emerlpips_%j.log

# Compute final-eval LPIPS for EmerNeRF on one clip using test_videos/25000_*.mp4.
#   sbatch --export=ALL,SEQ=002 nvidia_fog_emernerf_lpips.sbatch
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1

SEQ="${SEQ:?set SEQ=002|003|004}"
ROOT="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/emernerf/${SEQ}/foggyfields_emernerf/nvidia_fog_${SEQ}_7cam"
PRED="$ROOT/test_videos/25000_rgbs.mp4"
GT="$ROOT/test_videos/25000_gt_rgbs.mp4"
OUT="$ROOT/metrics_lpips.json"

echo "[$(date)] LPIPS emernerf/$SEQ  pred=$PRED  gt=$GT"
cd ~/ETH/Code/SplatAD/jobs
python compute_lpips_emernerf.py "$PRED" "$GT" --output "$OUT"
