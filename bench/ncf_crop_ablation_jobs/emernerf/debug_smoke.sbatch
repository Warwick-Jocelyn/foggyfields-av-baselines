#!/bin/bash
#SBATCH --job-name=emer_smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:25:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_emernerf_smoke/debug_%j.log
# ONE debug allocation for: (1) venv build, (2) 200-iter plumbing test, (3) the NCF-02 smoke
# (3000 iters, 8 held-out timesteps) through every stage of lane_emernerf.sh.
# STEPS env var selects a subset, e.g. STEPS="smoke".
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
STEPS=${STEPS:-"build quick smoke"}
T=$W/logs_emernerf_smoke/steps_${SLURM_JOB_ID}.txt
step(){ echo "$(date -Is) $1 $2" | tee -a "$T"; }
for s in $STEPS; do
  t0=$(date +%s)
  case $s in
    build) bash $W/jobs/emernerf/build_env.sh || { step build FAIL; exit 1; } ;;
    quick) env CLIPS=NCF-02 ITERS=200 MAX_EVAL_FRAMES=1 EMER_EXTRA="optim.cache_rgb_freq=100 logging.print_freq=50" \
             OUT_ROOT=$W/outputs_emernerf_dbg RES_ROOT=$W/results_emernerf_dbg VIZ_ROOT=$W/viz_emernerf_dbg LOG_ROOT=$W/logs_emernerf_dbg \
             bash $W/jobs/emernerf/lane_emernerf.sh NCF-02 0 || { step quick FAIL; exit 2; } ;;
    smoke) env CLIPS=NCF-02 ITERS=${SMOKE_ITERS:-3000} MAX_EVAL_FRAMES=${SMOKE_FRAMES:-8} \
             OUT_ROOT=$W/outputs_emernerf_smoke RES_ROOT=$W/results_emernerf_smoke VIZ_ROOT=$W/viz_emernerf_smoke LOG_ROOT=$W/logs_emernerf_smoke \
             bash $W/jobs/emernerf/lane_emernerf.sh NCF-02 0 || { step smoke FAIL; exit 3; } ;;
  esac
  step "$s" "OK $(( $(date +%s) - t0 ))s"
done
