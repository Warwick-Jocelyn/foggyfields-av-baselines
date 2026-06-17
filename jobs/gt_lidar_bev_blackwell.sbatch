#!/bin/bash
#SBATCH --job-name=bw_gtlidar
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1
#SBATCH --time=00:30:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/bw_gtlidar_%j.log

# GT-only LiDAR BEV: render each PandaSet lidar scan via plotly + stitch into mp4.
# Needs a GPU because eval_setup loads the pipeline (which initialises gsplat).
#
# Usage:
#   sbatch --export=ALL,SCENE=011 gt_lidar_bev_blackwell.sbatch
#   sbatch --export=ALL,SCENE=011,SUFFIX=_split05_paperfaithful gt_lidar_bev_blackwell.sbatch
SCENE="${SCENE:?must set SCENE=... via --export=ALL,SCENE=...}"
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

OUT_DIR=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/outputs/paper_${SCENE}${SUFFIX}
RUN_DIR=$(find "$OUT_DIR" -type d -name "nerfstudio_models" | xargs -I {} dirname {} | sort | tail -1)
CFG=$RUN_DIR/config.yml

echo "[$(date)] gt-lidar BEV  scene=$SCENE  cfg=$CFG"
python -c "import kaleido; print('kaleido', kaleido.__version__)" 2>/dev/null || pip install --no-cache-dir "kaleido==0.2.1"
python -c "import plotly" 2>/dev/null || pip install --no-cache-dir plotly

cd ~/ETH/Code/SplatAD/neurad-studio
python ~/ETH/Code/SplatAD/jobs/gt_lidar_bev_render.py "$CFG" "$OUT_DIR"

echo "[$(date)] DONE"
ls -lh "$OUT_DIR"/videos/*/lidar_gt.mp4 2>/dev/null
