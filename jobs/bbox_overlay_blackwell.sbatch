#!/bin/bash
#SBATCH --job-name=bw_bbox
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=01:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_bbox_%j.log

# Overlay trained DynamicActors cuboids on already-rendered PNGs and re-encode
# per-camera MP4s with an extra "rgb_bbox" track and a 4-panel compare.
#
# Usage:  sbatch --export=ALL,SCENE=011 bbox_overlay_blackwell.sbatch
#         sbatch --export=ALL,SCENE=078 bbox_overlay_blackwell.sbatch
#         sbatch --export=ALL,SCENE=011,SUFFIX=_split05_paperfaithful bbox_overlay_blackwell.sbatch
# SUFFIX (optional, default ""): selects between paper_${SCENE}${SUFFIX}/ output trees.
SCENE="${SCENE:?must set SCENE=011 or SCENE=078 via --export=ALL,SCENE=...}"
SUFFIX="${SUFFIX:-}"

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

echo "[$(date)] node=$(hostname)  scene=$SCENE"
nvidia-smi -L

OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/paper_${SCENE}${SUFFIX}
RENDER_DIR=$OUT_DIR/renders
VIDEO_DIR=$OUT_DIR/videos

# Locate the run dir (the one with a checkpoint)
RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml
echo "[$(date)] config = $CFG"
ls -lh "$RUN_DIR/nerfstudio_models/"

cd ~/ETH/Code/SplatAD/neurad-studio

# Make sure opencv-python-headless is available (small enough to install on the fly
# if missing — keep the env build script lean).
python -c "import cv2; print('cv2', cv2.__version__)" 2>/dev/null || \
    pip install --no-cache-dir opencv-python-headless

# ───────────────── STEP 1 : project cuboids onto rgb/ PNGs ─────────────────
echo "[$(date)] === STEP 1: draw_bbox_overlay (train + val) ==="
python ~/ETH/Code/SplatAD/jobs/draw_bbox_overlay.py "$CFG" "$RENDER_DIR" \
    --splits train val

# ───────────────── STEP 2 : re-encode MP4 (now with rgb_bbox track) ─────────────────
echo "[$(date)] === STEP 2: PNG → MP4 (with rgb_bbox + compare_bbox) ==="
for SPLIT in train val train+val; do
    if [ -d "$RENDER_DIR/$SPLIT" ]; then
        python ~/ETH/Code/SplatAD/jobs/pngs_to_mp4.py "$RENDER_DIR" "$VIDEO_DIR/$SPLIT" \
            --split "$SPLIT" --fps 10
    fi
done

echo "[$(date)] DONE — bbox overlay artefacts:"
ls -lh "$RENDER_DIR"/train/rgb_bbox/front_camera/ 2>/dev/null | head -3 || true
ls -lh "$VIDEO_DIR"/*/*_compare_bbox.mp4 2>/dev/null | head -20 || true
