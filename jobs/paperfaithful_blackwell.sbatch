#!/bin/bash
#SBATCH --job-name=bw_paperfaith
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=03:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_paperfaith_%j.log

# Paper-faithful SplatAD repro on PandaSet 011:
#   train_split_fraction = 0.5  (every other frame, per SplatAD CVPR'25 §5/supp B.1)
#   30000 iterations, six cameras + Pandar64, back camera bottom-260 crop (dataparser default)
#
# Output goes to outputs/paper_011_split05_paperfaithful/ — does NOT touch the older
# split=0.8 hold-every-5 run at outputs/paper_011/.

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

SCENE="${SCENE:-011}"
TEST_ROOT=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset
OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/paper_${SCENE}_split05_paperfaithful
RENDER_DIR=$OUT_DIR/renders
VIDEO_DIR=$OUT_DIR/videos
mkdir -p "$OUT_DIR"

echo "[$(date)] node=$(hostname)  scene=$SCENE  train_split_fraction=0.5 (paper-faithful)"
nvidia-smi -L

cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === STEP 1: ns-train splatad (30k iter, split=0.5) ==="
ns-train splatad \
    --output-dir "$OUT_DIR" \
    --vis tensorboard \
    pandaset-data \
    --data "$TEST_ROOT" \
    --sequence "$SCENE" \
    --train-split-fraction 0.5

RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml
echo "[$(date)] run=$RUN_DIR"
ls -lh "$RUN_DIR/nerfstudio_models/"

echo "[$(date)] === STEP 2: ns-render dataset --pose-source train+val (rgb + gt-rgb + depth) ==="
ns-render dataset \
    --load-config "$CFG" \
    --output-path "$RENDER_DIR" \
    --image-format png \
    --rendered-output-names rgb gt-rgb depth \
    --pose-source train+val

echo "[$(date)] === STEP 3: 3x2 panorama mp4 per split ==="
for SPLIT in train val train+val; do
    if [ -d "$RENDER_DIR/$SPLIT" ]; then
        python ~/ETH/Code/SplatAD/jobs/pngs_to_mp4.py "$RENDER_DIR" "$VIDEO_DIR/$SPLIT" \
            --split "$SPLIT" --fps 10 --panorama-only
    fi
done

echo "[$(date)] DONE"
ls -lh "$VIDEO_DIR"/*/panorama_*.mp4 2>/dev/null
