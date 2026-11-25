#!/bin/bash
# One node of jobs/fixbench/run.sbatch (inside the container). Plan rows: "<gpu> <method> <clip>",
# method = neurad | unisim | emernerf. Corrected flags come from config.sh; outputs go to the
# *_fix roots (or $FIX_TAG, e.g. smoke), never into the old bench dirs.
#   neurad|unisim  jobs/lane.sh (train, render, RGB metrics, videos), then jobs/lidar_lane.sh (LiDAR eval)
#   emernerf       jobs/emernerf/lane_emernerf.sh (has its own LiDAR stages)
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
: "${PLAN:?}"
TAG=${FIX_TAG:-fix}
say(){ echo "[$(date '+%F %T')] [$(hostname)] $*"; }
run_bg(){ "$@" & LANE=$!; wait $LANE; }   # child in the background so a SIGTERM is handled at once
lane(){  # gpu method clip
  local gpu=$1 m=$2 c=$3 data=$W/data_nff rc
  case "$c" in NCF-*) data=$W/data_all ;; esac
  source "$W/jobs/fixbench/config.sh"
  export DATA_ROOT=$data OUT_ROOT=$W/outputs_$TAG RES_ROOT=$W/results_$TAG VIZ_ROOT=$W/viz_$TAG LOG_ROOT=$W/logs_$TAG
  mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"
  local lg=$LOG_ROOT/lanes/${m}__safe__$c.log
  trap '[ -n "$LANE" ] && kill -TERM $LANE 2>/dev/null; wait $LANE 2>/dev/null; exit 143' TERM
  sleep $((gpu * 20))
  case "$m" in
    neurad|unisim)
      export TIMINGS=$RES_ROOT/timings.csv
      local v=${m^^}_EXTRA d=${m^^}_DP_EXTRA
      { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $gpu $m $c"; echo "extra: ${!v}"; echo "dp: ${!d} ########"; } >> "$lg"
      run_bg bash "$W/jobs/lane.sh" "$m" safe "$c" "$gpu" >> "$lg" 2>&1; rc=$?
      if [ $rc -eq 0 ]; then
        export BENCH_OUT=$OUT_ROOT BENCH_RES=$RES_ROOT BENCH_LOG=$LOG_ROOT BENCH_DATA=$data DUMP_CLIPS="NCF-02 NFF-05"
        run_bg bash "$W/jobs/lidar_lane.sh" "$m" "$c" "$gpu" >> "$LOG_ROOT/lanes/lidar_${m}__safe__$c.log" 2>&1; rc=$?
      fi ;;
    emernerf)
      export TIMINGS=$RES_ROOT/timings_emernerf.csv
      { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $gpu emernerf $c ########"; } >> "$lg"
      run_bg bash "$W/jobs/emernerf/lane_emernerf.sh" "$c" "$gpu" >> "$lg" 2>&1; rc=$? ;;
    *) say "!! unknown method $m"; return 2 ;;
  esac
  say "lane $m/$c gpu $gpu rc=$rc"
  return $rc
}
PIDS=()
while read -r g m c; do
  [[ "$g" =~ ^# ]] || [ -z "$c" ] && continue
  lane "$g" "$m" "$c" & PIDS+=($!)
done < "$PLAN"
trap 'say "!! SIGTERM"; kill -TERM "${PIDS[@]}" 2>/dev/null; wait; exit 143' TERM
wait "${PIDS[@]}"
say "node done"
