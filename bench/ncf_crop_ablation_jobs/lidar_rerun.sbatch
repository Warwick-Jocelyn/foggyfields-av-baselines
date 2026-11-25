#!/bin/bash
#SBATCH --job-name=lidar_rerun
#SBATCH --partition=debug
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@120
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_bench/lidar_rerun_%j.log
# One-off: re-run the LiDAR eval (FORCE_LIDAR=1, jobs/lidar_lane.sh) of the lanes whose eval ran twice at
# once before lidar_lane.sh held the run-dir lock: neurad NFF-05, unisim NCF-02/07/10/15.
# Rows "<gpu> <method> <clip>"; one GPU runs its rows in order.
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
export BENCH_DATA=$W/data_all BENCH_OUT=$W/outputs_bench BENCH_RES=$W/results_bench BENCH_LOG=$W/logs_bench
export DUMP_CLIPS="NCF-02 NFF-05" FORCE_LIDAR=1
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash -c '
  run_gpu(){ g=$1; shift; while [ $# -gt 0 ]; do
      lg=$BENCH_LOG/lanes/lidar_ns__$1__safe__$2.log
      bash '"$W"'/jobs/lidar_lane.sh $1 $2 $g > $lg 2>&1; echo "[$(date -Is)] gpu $g $1/$2 rc=$?"; shift 2; done; }
  run_gpu 0 neurad NFF-05 &
  run_gpu 1 unisim NCF-02 &
  run_gpu 2 unisim NCF-07 &
  run_gpu 3 unisim NCF-15 unisim NCF-10 &
  wait'
