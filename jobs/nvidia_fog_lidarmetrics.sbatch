#!/bin/bash
#SBATCH --job-name=nvfog_lidarm
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/lidarm_%j.log
# Compute LiDAR depth metrics from raw-depth renders + GT sweeps.
#   sbatch --export=ALL,METHOD=splatad,SEQ=002 nvidia_fog_lidarmetrics.sbatch
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1
METHOD="${METHOD:?set METHOD=splatad|neurad}"
SEQ="${SEQ:?set SEQ=002|003|004}"
OUT=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/${METHOD}/${SEQ}/lidar_metrics.json
cd ~/ETH/Code/SplatAD/jobs
python compute_lidar_metrics.py --method "$METHOD" --seq "$SEQ" --output "$OUT"
