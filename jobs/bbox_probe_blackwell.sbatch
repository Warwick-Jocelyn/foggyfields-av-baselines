#!/bin/bash
#SBATCH --job-name=bw_probe
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=00:20:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_probe_%j.log

SCENE="${SCENE:?must set SCENE=011 via --export=ALL,SCENE=...}"
CAM="${CAM:-front_camera}"
FRAME="${FRAME:-30}"

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
RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml
OUT=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bbox_probe_${SCENE}_${CAM}_${FRAME}.png

echo "[$(date)] probe scene=$SCENE cam=$CAM frame=$FRAME"
cd ~/ETH/Code/SplatAD/neurad-studio
python ~/ETH/Code/SplatAD/jobs/bbox_projection_probe.py "$CFG" "$RENDER_DIR" "$OUT" \
    --split train --camera "$CAM" --frame "$FRAME"

echo "[$(date)] DONE -> $OUT"
ls -lh "$OUT"
