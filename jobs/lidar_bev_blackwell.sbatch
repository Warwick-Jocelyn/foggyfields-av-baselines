#!/bin/bash
#SBATCH --job-name=bw_lidar
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_lidar_%j.log

# Use the trained SplatAD checkpoint to render per-scan LiDAR point-cloud PNGs
# (predicted + GT, via plot_lidar_points), then stitch into mp4.
#
# Usage:  sbatch --export=ALL,SCENE=011 lidar_bev_blackwell.sbatch
SCENE="${SCENE:?must set SCENE=... via --export=ALL,SCENE=...}"

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1
: "${NVCC_PREPEND_FLAGS:=}"; export NVCC_PREPEND_FLAGS
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/paper_${SCENE}
RENDER_DIR=$OUT_DIR/renders
VIDEO_DIR=$OUT_DIR/videos
RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml

echo "[$(date)] scene=$SCENE  cfg=$CFG"
ls -lh "$RUN_DIR/nerfstudio_models/"

# plot_lidar_points uses plotly + kaleido for image export
python -c "import kaleido; print('kaleido', kaleido.__version__)" 2>/dev/null || \
    pip install --no-cache-dir "kaleido==0.2.1"
python -c "import plotly; print('plotly', plotly.__version__)" 2>/dev/null || \
    pip install --no-cache-dir plotly

cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === ns-render dataset --render-point-clouds (skip camera outputs) --pose-source train+val ==="
ns-render dataset \
    --load-config "$CFG" \
    --output-path "$RENDER_DIR" \
    --image-format png \
    --rendered-output-names none \
    --render-point-clouds True \
    --pose-source train+val

echo "[$(date)] === PNG → MP4 ==="
for SPLIT in train val train+val; do
    if [ -d "$RENDER_DIR/$SPLIT/lidar" ]; then
        python ~/ETH/Code/SplatAD/jobs/lidar_bev_pngs_to_mp4.py "$RENDER_DIR" \
            "$VIDEO_DIR/$SPLIT" --split "$SPLIT" --fps 10
    fi
done

echo "[$(date)] DONE — outputs:"
ls -lh "$VIDEO_DIR"/*/lidar_*.mp4 2>/dev/null
