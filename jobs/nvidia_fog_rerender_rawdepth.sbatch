#!/bin/bash
#SBATCH --job-name=nvfog_rawdepth
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=02:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/rawdepth_%j.log

# Re-render val frames with raw-depth output for LiDAR metric computation.
# ns-render's default `depth` is 8-bit Turbo colormapped — unusable for metrics.
# `raw-depth` gives the float32 depth in meters.
#   sbatch --export=ALL,METHOD=splatad,SEQ=002 nvidia_fog_rerender_rawdepth.sbatch
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad-blackwell
export PYTHONNOUSERSITE=1
: "${NVCC_PREPEND_FLAGS:=}"; export NVCC_PREPEND_FLAGS
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
ulimit -n "$(ulimit -Hn)" || true

METHOD="${METHOD:?set METHOD=splatad|neurad}"
SEQ="${SEQ:?set SEQ=002|003|004}"
OUT_ROOT="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/${METHOD}/${SEQ}"
RAWDEPTH_DIR="$OUT_ROOT/renders_rawdepth"

# Use the MOST-RECENT trained config (the MCMC cap=5M SplatAD or the post-fix NeuRAD).
RUN_DIR=$(find "$OUT_ROOT" -type d -name nerfstudio_models -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
CFG="$RUN_DIR/config.yml"
[ -f "$CFG" ] || { echo "no config at $CFG"; exit 1; }
echo "[$(date)] re-render $METHOD/$SEQ from $RUN_DIR"

cd ~/ETH/Code/SplatAD/neurad-studio
ns-render dataset \
    --load-config "$CFG" \
    --output-path "$RAWDEPTH_DIR" \
    --rendered-output-names raw-depth \
    --pose-source val

echo "[$(date)] DONE — output at $RAWDEPTH_DIR"
ls "$RAWDEPTH_DIR/val/" 2>/dev/null | head
