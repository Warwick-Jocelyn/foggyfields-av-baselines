#!/bin/bash
# One node of jobs/neuraddiag/run.sbatch (NeuRAD variants) (inside the container). Plan rows: "<gpu> <variant> <clip>".
# Every variant gets its own output/result/viz roots, so lane.sh's resume and locks work per variant.
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
source "$W/jobs/neuraddiag/variants.sh"
: "${PLAN:?}"
say(){ echo "[$(date '+%F %T')] [$(hostname)] $*"; }
lane(){  # gpu variant clip
  local g=$1 v=$2 c=$3 flags
  flags=$(variant_flags "$v") || { say "!! unknown variant $v"; return 2; }
  export DATA_ROOT=$W/data_all OUT_ROOT=$W/outputs_neuraddiag/$v RES_ROOT=$W/results_neuraddiag/$v \
         VIZ_ROOT=$W/viz_nff/neurad_diag/$v LOG_ROOT=$W/logs_neuraddiag/$v NEURAD_EXTRA="$flags" NEURAD_DP_EXTRA="$(variant_dp_flags "$v")"
  export TIMINGS=$RES_ROOT/timings.csv
  mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"
  local lg=$LOG_ROOT/lanes/neurad__safe__$c.log
  { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $g variant $v flags: $flags dp: $NEURAD_DP_EXTRA ########"; } >> "$lg"
  sleep $((g * 20))
  bash "$W/jobs/lane.sh" neurad safe "$c" "$g" >> "$lg" 2>&1 & LANE=$!
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
