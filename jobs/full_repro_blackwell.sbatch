#!/bin/bash
#SBATCH --job-name=bw_full
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_full_%j.log

# Paper-faithful SplatAD reproduction on one PandaSet scene:
#   1) train 30k steps with train-split-fraction=0.8 (hold every 5)
#   2) ns-render dataset --pose-source train+val  (full 8-sec sequence)
#   3) build per-camera MP4 (rgb / gt / depth / 3-panel comparison)
#
# Usage:  sbatch --export=ALL,SCENE=011 full_repro_blackwell.sbatch
#         sbatch --export=ALL,SCENE=078 full_repro_blackwell.sbatch
SCENE="${SCENE:?must set SCENE=011 or SCENE=078 via --export=ALL,SCENE=...}"

# drop `-u`: conda's cuda-nvcc activation hook references NVCC_PREPEND_FLAGS unset
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
python -c "import torch, tinycudann, gsplat; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"

TEST_ROOT=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset
OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/paper_${SCENE}
RENDER_DIR=$OUT_DIR/renders
VIDEO_DIR=$OUT_DIR/videos
mkdir -p "$OUT_DIR"

cd ~/ETH/Code/SplatAD/neurad-studio

# ───────────────────── STEP 1 : train (paper defaults: 30000 steps) ─────────────────────
echo "[$(date)] === STEP 1: ns-train splatad (30000 steps, train_split_fraction=0.8) ==="
ns-train splatad \
    --output-dir "$OUT_DIR" \
    --vis tensorboard \
    pandaset-data \
    --data "$TEST_ROOT" \
    --sequence "$SCENE" \
    --train-split-fraction 0.8

# locate latest run dir (the one with a final checkpoint)
RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml
echo "[$(date)] training done; latest run = $RUN_DIR"
ls -lh "$RUN_DIR/nerfstudio_models/"

# ───────────────────── STEP 2 : ns-render dataset (train+val) ─────────────────────
echo "[$(date)] === STEP 2: ns-render dataset --pose-source train+val ==="
ns-render dataset \
    --load-config "$CFG" \
    --output-path "$RENDER_DIR" \
    --image-format png \
    --rendered-output-names rgb gt-rgb depth \
    --pose-source train+val

# ─────────────────── STEP 3 : encode MP4 per camera ───────────────────────
echo "[$(date)] === STEP 3: PNG → MP4 ==="
for SPLIT in train val train+val; do
    if [ -d "$RENDER_DIR/$SPLIT" ]; then
        python ~/ETH/Code/SplatAD/jobs/pngs_to_mp4.py "$RENDER_DIR" "$VIDEO_DIR/$SPLIT" --split "$SPLIT" --fps 10
    fi
done

echo "[$(date)] DONE — outputs:"
echo "  checkpoint: $RUN_DIR/nerfstudio_models/"
echo "  renders:    $RENDER_DIR"
echo "  videos:     $VIDEO_DIR"
ls -lh "$VIDEO_DIR"/*/*.mp4 2>/dev/null | head -20 || true
