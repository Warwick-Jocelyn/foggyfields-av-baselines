#!/bin/bash
#SBATCH --job-name=splatad_lidar
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_nffdiag/splatad_lidar_%j.log
# LiDAR eval (jobs/lidar_lane.sh, same protocol as the corrected bench) of the fixed SplatAD runs.
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash -c '
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
one(){ v=$1 c=$2 g=$3; d=$W/data_nff; case $c in NCF-*) d=$W/data_all;; esac
  BENCH_OUT=$W/outputs_nffdiag/$v BENCH_RES=$W/results_nffdiag/$v BENCH_LOG=$W/logs_nffdiag/$v BENCH_DATA=$d DUMP_CLIPS= \
    bash $W/jobs/lidar_lane.sh splatad $c $g > $W/logs_nffdiag/lidar_${v}_$c.log 2>&1; echo "$v $c rc=$?"; }
one opreg5e4 NCF-02 0 & one opreg5e4 NFF-13 1 & one opreg5e4 NFF-15 2 & one allfix NCF-02 3 & wait'
