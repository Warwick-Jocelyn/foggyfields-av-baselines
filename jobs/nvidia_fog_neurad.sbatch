#!/bin/bash
#SBATCH --job-name=nvfog_neurad
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/neurad_%j.log

# NeuRAD-paper reproduction on NVIDIA_AV_Fog clip $SEQ.
# Same dataparser as SplatAD; just different method config.

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

# NeuRAD's image_lidar parallel datamanager spawns num_processes=8 workers that
# share cached tensors via file descriptors; the default soft nofile limit (1024)
# is exhausted ("Too many open files (24)") when several jobs co-reside on a node.
# Raise the soft limit to the node's hard max.
ulimit -n "$(ulimit -Hn)" || ulimit -n 51200
echo "[$(date)] nofile soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

# Disable torch.compile / dynamo. NeuRAD hits `_dynamo recompile_limit (8)` on
# `transform_points_pairwise` (lidars.py:549) early in training; the recompile
# cache then thrashes and the job freezes at 0% GPU CPU-burning. Eager is
# slower but stable.
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

SEQ="${SEQ:?must set SEQ=002|003|004 via --export=ALL,SEQ=...}"
MAX_ITER="${MAX_ITER:-30000}"
OUT_ROOT="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/neurad/${SEQ}"
RENDER_DIR="$OUT_ROOT/renders"
VIDEO_DIR="$OUT_ROOT/videos"
mkdir -p "$OUT_ROOT"

echo "[$(date)] node=$(hostname) seq=$SEQ method=neurad-paper iter=$MAX_ITER"
nvidia-smi -L

cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === STEP 1: ns-train neurad-paper ($MAX_ITER iter, nvs_50_50) ==="
ns-train neurad-paper \
    --output-dir "$OUT_ROOT" \
    --vis wandb \
    --max-num-iterations "$MAX_ITER" \
    --steps-per-eval-all-images 999999 \
    --steps-per-save 10000 \
    nvidia-av-fog-data \
    --sequence "$SEQ" \
    --train-split-fraction 0.5
# NOTE: --steps-per-eval-all-images 999999 means full eval only at final step.
# With correct (216 m) NVIDIA-Fog scene scale, the intermediate eval-all-images
# at step 5000 freezes mid-prep (~step 4900) — saw it with both 30 k step pose-fix
# runs even with TORCHDYNAMO_DISABLE=1. Final eval is produced by ns-eval after
# training (already wired in jobs/nvidia_fog_eval.sbatch).

RUN_DIR=$(find "$OUT_ROOT" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG="$RUN_DIR/config.yml"
echo "[$(date)] run_dir=$RUN_DIR"

echo "[$(date)] === STEP 2: ns-render dataset (rgb + gt-rgb + depth) ==="
ns-render dataset \
    --load-config "$CFG" \
    --output-path "$RENDER_DIR" \
    --image-format png \
    --rendered-output-names rgb gt-rgb depth \
    --pose-source train+val

echo "[$(date)] === STEP 3: 7-cam 3x3 panorama mp4 per split ==="
for SPLIT in train val train+val; do
    if [ -d "$RENDER_DIR/$SPLIT" ]; then
        python ~/ETH/Code/SplatAD/jobs/pngs_to_mp4_nvidia_fog.py \
            "$RENDER_DIR" "$VIDEO_DIR/$SPLIT" \
            --split "$SPLIT" --fps 10 --panorama-only
    fi
done

echo "[$(date)] DONE — videos in $VIDEO_DIR"
ls -lh "$VIDEO_DIR"/*/panorama_*.mp4 2>/dev/null
