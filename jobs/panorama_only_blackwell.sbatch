#!/bin/bash
#SBATCH --job-name=bw_pano
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_pano_%j.log

# Encode 3x2 panorama MP4 (front-left | front | front-right / left | back |
# right) for each existing rendered output (rgb, gt-rgb, depth, rgb_bbox if
# present). No GPU needed; pure CPU re-encoding of already-rendered PNGs.
#
# Usage:  sbatch --export=ALL,SCENE=011 panorama_only_blackwell.sbatch
SCENE="${SCENE:?must set SCENE=011 or SCENE=078 via --export=ALL,SCENE=...}"

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1

OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/paper_${SCENE}
RENDER_DIR=$OUT_DIR/renders
VIDEO_DIR=$OUT_DIR/videos

echo "[$(date)] scene=$SCENE"
for SPLIT in train val train+val; do
    if [ -d "$RENDER_DIR/$SPLIT" ]; then
        echo "[$(date)] -- split=$SPLIT --"
        python ~/ETH/Code/SplatAD/jobs/pngs_to_mp4.py "$RENDER_DIR" "$VIDEO_DIR/$SPLIT" \
            --split "$SPLIT" --fps 10 --panorama-only
    fi
done

echo "[$(date)] DONE — panorama outputs:"
ls -lh "$VIDEO_DIR"/*/panorama_*.mp4 2>/dev/null
