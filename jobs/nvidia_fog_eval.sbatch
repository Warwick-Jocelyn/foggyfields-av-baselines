#!/bin/bash
#SBATCH --job-name=nvfog_eval
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=06:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/eval_%j.log

# Run ns-eval on a finished SplatAD/NeuRAD run to dump held-out (val) metrics
# (PSNR/SSIM/LPIPS + depth) to a JSON. The training sbatch only renders videos;
# this computes the comparison-table numbers.
#   sbatch --export=ALL,METHOD=neurad,SEQ=003 nvidia_fog_eval.sbatch

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
ulimit -n "$(ulimit -Hn)" || true
# SplatAD keeps ~90GB of Gaussians/buffers resident at eval; the LiDAR
# chamfer_distance (torch.cdist) then OOMs on the last ~0.7GB. expandable_segments
# reclaims the reserved-but-unallocated fragmentation so the metric fits.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

METHOD="${METHOD:?set METHOD=splatad|neurad}"
SEQ="${SEQ:?set SEQ=002|003|004}"
OUT_ROOT="/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/${METHOD}/${SEQ}"

# Pick the most-recent complete run (has nerfstudio_models). Avoids stale smoke runs.
RUN_DIR=$(find "$OUT_ROOT" -type d -name nerfstudio_models -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
CFG="$RUN_DIR/config.yml"
echo "[$(date)] eval METHOD=$METHOD SEQ=$SEQ run=$RUN_DIR"
[ -f "$CFG" ] || { echo "no config.yml under $OUT_ROOT"; exit 1; }

cd ~/ETH/Code/SplatAD/neurad-studio
ns-eval --load-config "$CFG" --output-path "$OUT_ROOT/eval_metrics.json"
echo "[$(date)] wrote $OUT_ROOT/eval_metrics.json"
cat "$OUT_ROOT/eval_metrics.json"
