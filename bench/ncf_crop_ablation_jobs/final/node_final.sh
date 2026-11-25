#!/bin/bash
# One node of jobs/final/run_final.sbatch (inside the container). Plan rows: "<gpu> <kind> <args...>";
# rows with the same gpu run one after another. Kinds:
#   splatad <clip>      SplatAD allfix (opacity reg 5e-4 + F1b/F2b), lane.sh + lidar_lane.sh + lidar_viz
#   splatad-opreg <clip> SplatAD fallback variant (opacity reg 5e-4, pre-fix data path)
#   neurad <clip>       NeuRAD standard 30k (fixbench flags), lane.sh + lidar_lane.sh + lidar_viz
#   emernerf <clip>     EmerNeRF 25k (lane_emernerf.sh: has its own LiDAR stages)
#   v13 <label> <clip> <config.yml>   score one FoggyField V1.3 run with the common evaluator (v13_lane.sh)
#   v13poll             keep scoring V1.3 runs from v13_queue.txt / RESULTS_LANES.md until the deadline (v13_poll.sh)
# A job-level owner lock (one live job per plan) lets the same plan be submitted on two accounts.
set -o pipefail
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/final/config_final.sh
: "${PLAN:?}"
ulimit -c 0
say(){ echo "[$(date '+%F %T')] [$(hostname) job ${SLURM_JOB_ID:-?}] $*"; }
PLAN_NAME=$(basename "$PLAN" .txt)${FINAL_TAG:+_$FINAL_TAG}
R0=$BW/outputs_final${FINAL_TAG:+_$FINAL_TAG}; S0=$BW/results_final${FINAL_TAG:+_$FINAL_TAG}; Z0=$BW/viz_final${FINAL_TAG:+_$FINAL_TAG}; L0=$BW/logs_final${FINAL_TAG:+_$FINAL_TAG}
# ---- owner lock (same scheme as jobs/lanelock.sh, per plan) ----
LOCK=$BW/outputs_final/.owner_$PLAN_NAME; mkdir -p "$BW/outputs_final"
if ! mkdir "$LOCK" 2>/dev/null; then
  hb=$(stat -c %Y "$LOCK/heartbeat" 2>/dev/null || stat -c %Y "$LOCK" 2>/dev/null || echo 0)
  age=$(( $(date +%s) - hb ))
  if [ "$age" -lt 300 ]; then say "plan $PLAN_NAME is being run by $(cat "$LOCK/owner" 2>/dev/null) (heartbeat ${age}s ago) -> this duplicate exits"; touch "$BW/outputs_final/.dup_exit_${SLURM_JOB_ID:-local}"; exit 0; fi
  say "stale owner lock (${age}s) -> taking over"; rm -rf "$LOCK"; mkdir "$LOCK" || exit 0
fi
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) since=$(date -Is) acct=${SLURM_JOB_ACCOUNT:-?}" > "$LOCK/owner"; touch "$LOCK/heartbeat"
( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat"; sleep 60; done ) >/dev/null 2>&1 & HB=$!
trap 'kill $HB 2>/dev/null; rm -rf "$LOCK"' EXIT
# Deadline for pollers: job end time from Slurm (fallback: now + 6 h), minus 25 min.
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null | head -1)
DEADLINE=$(( $( [ -n "$END" ] && date -d "$END" +%s 2>/dev/null || echo $(( $(date +%s) + 21600 )) ) - 1500 ))
export DEADLINE
say "plan $PLAN_NAME restart=${SLURM_RESTART_COUNT:-0} deadline=$(date -d @$DEADLINE '+%F %T')"; cat "$PLAN"

run_bg(){ "$@" & LANE=$!; wait $LANE; }
one(){  # gpu kind args...
  local g=$1 k=$2 c=${3:-} rc=0; shift $(( $# < 3 ? $# : 3 ))
  export CUDA_VISIBLE_DEVICES=$g
  export OUT_ROOT=$R0/std RES_ROOT=$S0/std VIZ_ROOT=$Z0/std LOG_ROOT=$L0/std
  case "$k" in
    splatad|splatad-opreg)
      local v=allfix; [ "$k" = splatad-opreg ] && v=opreg5e4
      export OUT_ROOT=$R0/$v RES_ROOT=$S0/$v VIZ_ROOT=$Z0/$v LOG_ROOT=$L0/$v
      if [ "$v" = allfix ]; then export SPLATAD_EXTRA="$SPLATAD_ALLFIX_EXTRA" SPLATAD_DP_EXTRA="$SPLATAD_ALLFIX_DP"
      else export SPLATAD_EXTRA="$SPLATAD_OPREG_EXTRA" SPLATAD_DP_EXTRA="$SPLATAD_OPREG_DP"; fi
      mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"; export TIMINGS=$RES_ROOT/timings.csv
      local lg=$LOG_ROOT/lanes/splatad__safe__$c.log
      { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $g splatad/$v $c extra: $SPLATAD_EXTRA dp: $SPLATAD_DP_EXTRA code: $NS_REPO ########"; } >> "$lg"
      run_bg bash "$BW/jobs/lane.sh" splatad safe "$c" "$g" >> "$lg" 2>&1; rc=$?
      [ $rc -eq 0 ] && { export BENCH_OUT=$OUT_ROOT BENCH_RES=$RES_ROOT BENCH_LOG=$LOG_ROOT BENCH_DATA=$DATA_ROOT DUMP_CLIPS=""
        run_bg bash "$BW/jobs/lidar_lane.sh" splatad "$c" "$g" >> "$LOG_ROOT/lanes/lidar_splatad__safe__$c.log" 2>&1; rc=$?; }
      ;;
    neurad)
      mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"; export TIMINGS=$RES_ROOT/timings.csv
      local lg=$LOG_ROOT/lanes/neurad__safe__$c.log
      { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $g neurad $c extra: $NEURAD_EXTRA dp: $NEURAD_DP_EXTRA code: $NS_REPO ########"; } >> "$lg"
      run_bg bash "$BW/jobs/lane.sh" neurad safe "$c" "$g" >> "$lg" 2>&1; rc=$?
      [ $rc -eq 0 ] && { export BENCH_OUT=$OUT_ROOT BENCH_RES=$RES_ROOT BENCH_LOG=$LOG_ROOT BENCH_DATA=$DATA_ROOT DUMP_CLIPS=""
        run_bg bash "$BW/jobs/lidar_lane.sh" neurad "$c" "$g" >> "$LOG_ROOT/lanes/lidar_neurad__safe__$c.log" 2>&1; rc=$?; }
      ;;
    unisim)
      mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"; export TIMINGS=$RES_ROOT/timings.csv
      local lg=$LOG_ROOT/lanes/unisim__safe__$c.log
      { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $g unisim $c extra: $UNISIM_EXTRA dp: $UNISIM_DP_EXTRA code: $NS_REPO plugin: $BW/code/_src/unisim ########"; } >> "$lg"
      run_bg flock "$BW/.unisim_install.lock" bash "$BW/jobs/final/unisim_install.sh" >> "$lg" 2>&1; rc=$?
      [ $rc -eq 0 ] || { say "unisim plugin install failed rc=$rc"; return $rc; }
      run_bg bash "$BW/jobs/lane.sh" unisim safe "$c" "$g" >> "$lg" 2>&1; rc=$?
      [ $rc -eq 0 ] && { export BENCH_OUT=$OUT_ROOT BENCH_RES=$RES_ROOT BENCH_LOG=$LOG_ROOT BENCH_DATA=$DATA_ROOT DUMP_CLIPS=""
        run_bg bash "$BW/jobs/lidar_lane.sh" unisim "$c" "$g" >> "$LOG_ROOT/lanes/lidar_unisim__safe__$c.log" 2>&1; rc=$?; } ;;
    emernerf)
      mkdir -p "$LOG_ROOT/lanes" "$RES_ROOT"; export TIMINGS=$RES_ROOT/timings_emernerf.csv
      local lg=$LOG_ROOT/lanes/emernerf__safe__$c.log
      { echo; echo "######## $(date -Is) job $SLURM_JOB_ID gpu $g emernerf $c ########"; } >> "$lg"
      # venv-emernerf self-repair (purge damage), serialised across the lanes of this node
      run_bg flock "$BW/venv-emernerf/.fix.lock" bash "$BW/jobs/final/emer_fix_env.sh" >> "$lg" 2>&1; rc=$?
      [ $rc -eq 0 ] || { say "emernerf env repair failed rc=$rc (see $lg)"; return $rc; }
      run_bg bash "$BW/jobs/emernerf/lane_emernerf.sh" "$c" "$g" >> "$lg" 2>&1; rc=$? ;;
    v13)   # c = label, $1 = clip, $2 = config
      mkdir -p "$BW/logs_final/v13"
      run_bg bash "$BW/jobs/final/v13_lane.sh" "$c" "$1" "$2" "$g" >> "$BW/logs_final/v13/${c}__$1.log" 2>&1; rc=$? ;;
    v13poll)
      mkdir -p "$BW/logs_final/v13"
      # the poller alone must not keep a plan open: run it until every OTHER GPU chain of this plan is done
      # (.chain_ok markers, persisted across chunks) or until the deadline; the lane then counts as finished.
      bash "$BW/jobs/final/v13_poll.sh" "$g" >> "$BW/logs_final/v13/poll_gpu${g}_job${SLURM_JOB_ID}.log" 2>&1 & local PP=$!
      while kill -0 $PP 2>/dev/null; do
        local others=1 og; for og in 0 1 2 3; do [ "$og" = "$g" ] && continue
          grep -qE "^$og " "$PLAN" && [ ! -f "$R0/.chain_ok_${PLAN_NAME}_gpu$og" ] && others=0; done
        [ $others -eq 1 ] && { say "v13poll gpu $g: other chains done -> stopping poller"; sleep 120; kill -TERM $PP 2>/dev/null; break; }
        sleep 60
      done
      wait $PP 2>/dev/null; rc=0 ;;
    *) say "!! unknown kind $k"; rc=2 ;;
  esac
  say "lane $k $c $* gpu $g rc=$rc"
  return $rc
}
lidar_fig(){  # method clip -> BEV etc. from the dump (CPU)
  local d=$OUT_ROOT/_lidar_dump/${1}_safe__$2 o=$VIZ_ROOT/$1/safe/$2/lidar
  [ -d "$d" ] && [ ! -s "$o/lidar_viz_stats.json" ] && { source "$BW/jobs/env.sh" >/dev/null 2>&1
    MPLBACKEND=Agg python "$BW/jobs/lidar_viz.py" --dump "$d" -o "$o" >> "$LOG_ROOT/lanes/lidarviz_${1}__$2.log" 2>&1 || say "lidar_viz $1/$2 failed (metrics unaffected)"; }
  return 0
}
gpu_chain(){  # gpu ; writes $R0/.chain_ok_<plan>_gpu<g> when every lane of this GPU finished with rc 0 (persists across chunks)
  local g=$1 bad=0
  trap '[ -n "$LANE" ] && kill -TERM $LANE 2>/dev/null; wait $LANE 2>/dev/null; exit 143' TERM
  sleep $((g * 15))
  while read -r pg k c rest; do
    [[ "$pg" =~ ^# ]] || [ -z "$k" ] && continue
    [ "$pg" = "$g" ] || continue
    one "$g" "$k" "$c" $rest || bad=1
  done < "$PLAN"
  [ $bad -eq 0 ] && touch "$R0/.chain_ok_${PLAN_NAME}_gpu$g"
}
PIDS=()
for g in 0 1 2 3; do grep -qE "^$g " "$PLAN" && { gpu_chain "$g" & PIDS+=($!); }; done
trap 'say "!! SIGTERM"; kill -TERM "${PIDS[@]}" 2>/dev/null; wait; exit 143' TERM
wait "${PIDS[@]}"
ALLOK=1; for g in 0 1 2 3; do grep -qE "^$g " "$PLAN" && [ ! -f "$R0/.chain_ok_${PLAN_NAME}_gpu$g" ] && ALLOK=0; done
[ $ALLOK -eq 1 ] && { touch "$R0/.plan_done_$PLAN_NAME"; say "PLAN DONE (all GPU chains finished)"; }
say "node done"
