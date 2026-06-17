#!/bin/bash
#SBATCH --job-name=bw_neurad
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=04:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_neurad_%j.log

# Phase 0.2 — NeuRAD baseline on PandaSet, paper-faithful + extended.
#
#   method:                neurad-paper      (use_temporal_appearance=False, flip_prob=0)
#   train_split_fraction:  0.5               (every other frame, matches SplatAD CVPR'25 §4)
#   max_num_iterations:    30000             (paper recipe is 20k; we run to 30k and snapshot @20k
#                                              via steps_per_eval_all_images=5000 + steps_per_save=2000)
#   eval_all @ 20k / 30k:  enables BOTH paper-comparable (20k) and same-iter-vs-SplatAD (30k)
#
# Output → outputs/neurad_paper_${SCENE}_split05/
#
# Usage:
#   sbatch --export=ALL,SCENE=011 neurad_paper_blackwell.sbatch
#   sbatch --export=ALL,SCENE=078 neurad_paper_blackwell.sbatch

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
OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/neurad_paper_${SCENE}_split05
RENDER_DIR=$OUT_DIR/renders
VIDEO_DIR=$OUT_DIR/videos
mkdir -p "$OUT_DIR"

echo "[$(date)] node=$(hostname)  scene=$SCENE  method=neurad-paper  max_iter=30000  split=0.5"
nvidia-smi -L

cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === STEP 1: ns-train neurad-paper (30k iter, split=0.5, eval @ every 5k) ==="
# Why these overrides:
#   --max-num-iterations 30000           : extend past paper's 20k for SplatAD-iter-match
#   --steps-per-eval-all-images 5000     : eval @ 5/10/15/20/25/30k → 20k snapshot is comparable to paper
#   --steps-per-save 5000                : save ckpt at the same cadence so we can re-render @ 20k later
ns-train neurad-paper \
    --output-dir "$OUT_DIR" \
    --vis tensorboard \
    --max-num-iterations 30000 \
    --steps-per-eval-all-images 5000 \
    --steps-per-save 5000 \
    pandaset-data \
    --data "$TEST_ROOT" \
    --sequence "$SCENE" \
    --train-split-fraction 0.5

RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml
echo "[$(date)] run=$RUN_DIR"
ls -lh "$RUN_DIR/nerfstudio_models/"

echo "[$(date)] === STEP 2: ns-render dataset (final 30k ckpt → rgb + gt-rgb + depth) ==="
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
