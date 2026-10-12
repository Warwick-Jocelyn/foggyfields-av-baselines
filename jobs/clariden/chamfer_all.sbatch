#!/bin/bash
#SBATCH --job-name=splatad_cd
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:20:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/splatad-nff-repro/logs/chamfer_all_%j.log
# Native-LiDAR Chamfer + inference timing for the 6 models (2 per GPU, sequential within a GPU).
set -o pipefail
source /capstor/scratch/cscs/yitiwang/splatad-nff-repro/jobs/env.sh
pip install -q scipy 2>/dev/null || true
cd $CODE/neurad-studio; mkdir -p $W/logs/chamfer
cfg(){ ls $OUT_ROOT/$1/$2/*/splatad/*/config.yml 2>/dev/null | tail -1; }
run(){ # variant clip gpu
  local C; if [ "$1" = hf_ckpt ]; then C=$OUT_ROOT/hf_ckpt/$2/config.yml; else C=$(cfg $1 $2); fi
  echo "[$(date '+%T')] start $1/$2 on GPU $3 cfg=$C"
  CUDA_VISIBLE_DEVICES=$3 python $W/jobs/lidar_chamfer_eval.py --config "$C" --output $OUT_ROOT/$1/$2/lidar_chamfer.json > $W/logs/chamfer/$1_$2.log 2>&1 \
    && echo "[$(date '+%T')] done  $1/$2" || echo "!! $1/$2 failed (see logs/chamfer/$1_$2.log)"
}
( run faithful NFF-02 0; run hf_ckpt NFF-02 0 ) &
( run faithful NFF-03 1; run hf_ckpt NFF-03 1 ) &
( run corrected NFF-02 2 ) &
( run corrected NFF-03 3 ) &
wait
python $W/jobs/collect_results.py >/dev/null && echo "[$(date)] CHAMFER ALL DONE"
