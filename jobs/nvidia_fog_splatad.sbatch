#!/bin/bash
#SBATCH --job-name=nvfog_splatad
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=12:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/splatad_%j.log

# SplatAD reproduction on NVIDIA_AV_Fog clip $SEQ.
# Usage:
#   sbatch --export=ALL,SEQ=002 nvidia_fog_splatad.sbatch
#   sbatch --export=ALL,SEQ=003 nvidia_fog_splatad.sbatch
#   sbatch --export=ALL,SEQ=004 nvidia_fog_splatad.sbatch
#
# Protocol: nvs_50_50 (train_split_fraction=0.5 LINSPACE), 30k iter, all 7 cams + 1 lidar.

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

SEQ="${SEQ:?must set SEQ=002|003|004 via --export=ALL,SEQ=...}"
MAX_ITER="${MAX_ITER:-30000}"
OUT_ROOT="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/splatad/${SEQ}"
RENDER_DIR="$OUT_ROOT/renders"
VIDEO_DIR="$OUT_ROOT/videos"
mkdir -p "$OUT_ROOT"

echo "[$(date)] node=$(hostname) seq=$SEQ method=splatad iter=$MAX_ITER"
nvidia-smi -L

# Disable torch.compile / dynamo. With the larger NVIDIA-Fog scene (216 m extent,
# 7 heterogeneous cams, 5M-cap Gaussians) the dynamo recompile cache thrashes
# after step 2500 (MCMC post-cap relocations), freezing the job CPU-burning at
# 0% GPU. Eager mode is ~15% slower but reliable.
export TORCHDYNAMO_DISABLE=1
# Make python print flush per-line so SLURM stdout reflects real progress.
export PYTHONUNBUFFERED=1

cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === STEP 1: ns-train splatad ($MAX_ITER iter, nvs_50_50) ==="
# Overrides for NVIDIA-Fog (vs PandaSet defaults):
#   --strategy=default       : MCMC strategy's `Relocated N GSs` op at step 2500
#                              hangs on millions of Gaussians on Blackwell (verified
#                              with both 5M and 10M init point clouds). The default
#                              densify/split strategy (standard 3DGS) is stable.
#   --max-num-seed-points=5000000 : NVIDIA-Fog has ~50M raw lidar pts (128-line LiDAR
#                                   × 199 sweeps); PandaSet default 2M starves init.
#                                   Capped at 5M (vs 10M) for memory headroom on 80GB.
#   --densify-grad-thresh=0.0003 : halved to compensate for mixed-resolution cams
#                                  (low-res 480×320 has lower gradient magnitudes)
ns-train splatad \
    --output-dir "$OUT_ROOT" \
    --vis wandb \
    --max-num-iterations "$MAX_ITER" \
    --steps-per-eval-image 999999 \
    --steps-per-eval-all-images 999999 \
    --steps-per-save 30000 \
    --pipeline.model.strategy mcmc \
    --pipeline.model.mcmc-cap-max 5000000 \
    --pipeline.model.max-num-seed-points 2000000 \
    nvidia-av-fog-data \
    --sequence "$SEQ" \
    --train-split-fraction 0.5
# MCMC re-run: cap=3M, max-seed=1.5M to keep init Gaussians below ~5M (after
# flip_actors_at_init doubling + per-actor min_points_per_actor=500). Previous
# Blackwell hang at step 2500 was with init=6.4M GSs → relocation kernel
# launched 6.25M blocks × 1024 threads = 6.4B threads in flight, hitting Blackwell's
# persistent-thread-occupancy limit. With init<5M, kernel grid stays within scale.
# MCMC noise injection + opacity-weighted relocation is the paper's headline
# strategy; default-strategy bleeds GS count (5.8M → 1.6M) and costs ~5 dB PSNR.
# steps-per-eval-image was previously default=500; full 7-cam render through
# millions of Gaussians at 216 m extent dominated wall time. Eval-image renders
# now only at end. steps-per-save raised to one final save (no intermediate
# checkpoints needed for our pipeline).
# NOTE: steps-per-eval-all-images=999999 prevents mid-training eval-cache freeze
# (correct 216 m scene scale makes eval rendering an order of magnitude heavier
# than PandaSet). Final eval via ns-eval / compute_rgb_metrics.py after training.

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
