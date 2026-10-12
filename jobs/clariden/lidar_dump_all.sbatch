#!/bin/bash
#SBATCH --job-name=splatad_ldump
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/splatad-nff-repro/logs/lidar_dump_%j.log
set -o pipefail
source /capstor/scratch/cscs/yitiwang/splatad-nff-repro/jobs/env.sh
cd $CODE/neurad-studio; mkdir -p $W/logs/ldump
cfg(){ ls $OUT_ROOT/$1/$2/*/splatad/*/config.yml 2>/dev/null | tail -1; }
run(){ local C; if [ "$1" = hf_ckpt ]; then C=$OUT_ROOT/hf_ckpt/$2/config.yml; else C=$(cfg $1 $2); fi
  echo "[$(date '+%T')] start $1/$2 GPU $3"; CUDA_VISIBLE_DEVICES=$3 python $W/jobs/lidar_dump.py --config "$C" --out $OUT_ROOT/$1/$2/lidar_dump > $W/logs/ldump/$1_$2.log 2>&1 && echo "[$(date '+%T')] done $1/$2" || echo "!! $1/$2 failed"; }
run corrected NFF-02 0 & run corrected NFF-03 1 & run hf_ckpt NFF-02 2 & run hf_ckpt NFF-03 3 & wait
echo "[$(date)] LIDAR DUMP ALL DONE"
