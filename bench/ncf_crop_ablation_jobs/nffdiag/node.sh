#!/bin/bash
# One node of jobs/nffdiag/run.sbatch (inside the container). Plan rows: "<gpu> <variant> <clip>".
# Every variant gets its own output/result/viz roots, so lane.sh's resume and locks work per variant.
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
source "$W/jobs/nffdiag/variants.sh"
: "${PLAN:?}"
say(){ echo "[$(date '+%F %T')] [$(hostname)] $*"; }
lane(){  # gpu variant clip
  local g=$1 v=$2 c=$3 flags
  flags=$(variant_flags "$v") || { say "!! unknown variant $v"; return 2; }
  local data=$W/data_nff; case "$c" in NCF-*) data=$W/data_all ;; esac   # NCF clips: same data as the crop ablation
  export DATA_ROOT=$data OUT_ROOT=$W/outputs_nffdiag/$v RES_ROOT=$W/results_nffdiag/$v \
         VIZ_ROOT=$W/viz_nff/$v LOG_ROOT=$W/logs_nffdiag/$v SPLATAD_EXTRA="$flags" \
         SPLATAD_DP_EXTRA="$(variant_dp_flags "$v")"
  export TIMINGS=$RES_ROOT/timings.csv
  mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"
  local lg=$LOG_ROOT/lanes/splatad__safe__$c.log
  { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $g variant $v flags: $flags dp: $SPLATAD_DP_EXTRA ########"; } >> "$lg"
  sleep $((g * 20))
  bash "$W/jobs/lane.sh" splatad safe "$c" "$g" >> "$lg" 2>&1 & LANE=$!
  trap 'kill -TERM $LANE 2>/dev/null; wait $LANE; exit 143' TERM
  wait $LANE; local rc=$?
  say "lane $v/$c gpu $g rc=$rc"
  return $rc
}
PIDS=()
while read -r g v c; do
  [[ "$g" =~ ^# ]] || [ -z "$c" ] && continue
  lane "$g" "$v" "$c" & PIDS+=($!)
done < "$PLAN"
trap 'say "!! SIGTERM"; kill -TERM "${PIDS[@]}" 2>/dev/null; wait; exit 143' TERM
wait "${PIDS[@]}"
say "node done"
