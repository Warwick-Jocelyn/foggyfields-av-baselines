#!/bin/bash
#SBATCH --job-name=bw_disk
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_disk_%j.log

# Usage:  sbatch --export=ALL,SCENE=011 render_disk_blackwell.sbatch
SCENE="${SCENE:?must set SCENE=011 or SCENE=078 via --export}"

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

OUT_BASE=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/blackwell_${SCENE}
# pick the most recently completed run dir (the one with a checkpoint)
RUN_DIR=$(find $OUT_BASE -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml
RENDER_OUT=$OUT_BASE/renders

echo "[$(date)] scene=$SCENE  run=$RUN_DIR"
echo "[$(date)] checkpoint exists?"; ls -lh $RUN_DIR/nerfstudio_models/

mkdir -p "$RENDER_OUT"
cd ~/ETH/Code/SplatAD/neurad-studio

echo "[$(date)] === ns-render dataset (val split, png, rgb + gt-rgb + depth) ==="
ns-render dataset \
    --load-config "$CFG" \
    --output-path "$RENDER_OUT" \
    --image-format png \
    --rendered-output-names rgb gt-rgb depth \
    --pose-source val

echo "[$(date)] === outputs ==="
ls -la "$RENDER_OUT"
find "$RENDER_OUT" -type f | head -20

echo "[$(date)] DONE — disk renders at $RENDER_OUT"
