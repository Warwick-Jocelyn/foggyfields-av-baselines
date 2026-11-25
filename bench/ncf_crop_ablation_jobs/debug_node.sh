#!/bin/bash
# One node of jobs/debug_chunk.sbatch (runs INSIDE the container, one task per node).
# Starts this node's lanes from $PLAN (lines "<node> <gpu> <method> <clip>"), one per GPU:
#   neurad|unisim  jobs/lane.sh <method> safe <clip> <gpu>, then the LiDAR eval (jobs/lidar_lane.sh)
#   emernerf       jobs/emernerf/lane_emernerf.sh <clip> <gpu>  (has its own LiDAR stages)
# On SIGTERM (wall clock) the lanes are told to stop; they log the partial segment and the next
# chunk resumes them.
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
JOBS=$W/jobs
NODE=${PLAN_NODE:-${SLURM_NODEID:-0}}   # PLAN_NODE=k: a 1-node job runs the plan rows of node k (debug-qos: max 90 node-minutes per job)
PLAN=${PLAN:-$JOBS/bench_plan.txt}
STEAL_PLAN=${STEAL_PLAN:-$JOBS/bench_plan.txt}   # the full bench, for work stealing
: "${BENCH_OUT:?}" "${BENCH_RES:?}" "${BENCH_VIZ:?}" "${BENCH_LOG:?}" "${BENCH_LOG_EMER:?}" "${BENCH_DATA:?}"
DUMP_CLIPS=${DUMP_CLIPS:-}
say(){ echo "[$(date '+%F %T')] [node $NODE $(hostname)] $*"; }

mapfile -t MINE < <(awk -v n="$NODE" '$1 !~ /^#/ && NF == 4 && $1 == n' "$PLAN")
say "${#MINE[@]} lanes: $(printf '%s; ' "${MINE[@]}")"
[ ${#MINE[@]} -gt 0 ] || exit 0
nvidia-smi -L | sed "s/^/[node $NODE] /"
say "cpus visible: $(nproc)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# The UniSim plugin is an editable install in the shared venv; refuse to start a UniSim lane without it.
if printf '%s\n' "${MINE[@]}" | awk '{print $3}' | grep -qx unisim; then
  ( source "$JOBS/env.sh" >/dev/null && python -c "from unisim.unisim_config import unisim" ) \
    || { say "!! unisim plugin does not import -- the UniSim lanes of this node are skipped"; NO_UNISIM=1; }
fi

ns_lane(){  # method clip gpu
  local m=$1 c=$2 g=$3
  export DATA_ROOT=$BENCH_DATA OUT_ROOT=$BENCH_OUT RES_ROOT=$BENCH_RES VIZ_ROOT=$BENCH_VIZ LOG_ROOT=$BENCH_LOG
  export TIMINGS=$BENCH_RES/timings.csv
  local lg=$BENCH_LOG/lanes/${m}__safe__${c}.log
  mkdir -p "$BENCH_LOG/lanes"
  { echo; echo "######## $(date -Is) job $SLURM_JOB_ID node $NODE gpu $g ########"; } >> "$lg"
  bash "$JOBS/lane.sh" "$m" safe "$c" "$g" >> "$lg" 2>&1 & LANE=$!
  wait $LANE; local rc=$?
  [ $rc -eq 0 ] || return $rc
  # LiDAR eval on the SAME config the RGB stages used (TRAIN_DONE), under the run-dir lock (jobs/lidar_lane.sh)
  bash "$JOBS/lidar_lane.sh" "$m" "$c" "$g" > "$BENCH_LOG/lanes/lidar_ns__${m}__safe__${c}.log.$SLURM_JOB_ID.$g" 2>&1 & LANE=$!
  wait $LANE; rc=$?
  # the log of the job that actually ran it (not BUSY/SKIP) becomes the lane's LiDAR log
  local lg2=$BENCH_LOG/lanes/lidar_ns__${m}__safe__${c}.log
  grep -q "\] start (dump" "$lg2.$SLURM_JOB_ID.$g" && mv "$lg2.$SLURM_JOB_ID.$g" "$lg2" || { cat "$lg2.$SLURM_JOB_ID.$g" >> "$lg"; rm -f "$lg2.$SLURM_JOB_ID.$g"; }
  return $rc
}
emer_lane(){  # clip gpu
  local c=$1 g=$2
  export DATA_ROOT=$BENCH_DATA OUT_ROOT=$BENCH_OUT RES_ROOT=$BENCH_RES VIZ_ROOT=$BENCH_VIZ LOG_ROOT=$BENCH_LOG_EMER
  export TIMINGS=$BENCH_RES/timings_emernerf.csv
  local lg=$BENCH_LOG_EMER/lanes/emernerf__${c}.log
  mkdir -p "$BENCH_LOG_EMER/lanes"
  { echo; echo "######## $(date -Is) job $SLURM_JOB_ID node $NODE gpu $g ########"; } >> "$lg"
  bash "$JOBS/emernerf/lane_emernerf.sh" "$c" "$g" >> "$lg" 2>&1 & LANE=$!
  wait $LANE
}
lane_once(){  # gpu method clip -> rc of the lane (75 = held by another job)
  local g=$1 m=$2 c=$3 rc
  say "START $m/$c gpu $g"
  case "$m" in
    neurad)   ns_lane neurad "$c" "$g" ;;
    unisim)   [ -z "$NO_UNISIM" ] && ns_lane unisim "$c" "$g" ;;
    emernerf) emer_lane "$c" "$g" ;;
    *) say "!! unknown method $m"; false ;;
  esac
  rc=$?
  if [ $rc -eq 0 ]; then say "DONE  $m/$c"; elif [ $rc -eq 75 ]; then say "BUSY  $m/$c (another job holds it)"; else say "FAIL  $m/$c rc=$rc"; echo "$(date -Is) $m $c rc=$rc node $NODE" >> "$BENCH_LOG/FAILURES_${SLURM_JOB_ID}.txt"; fi
  return $rc
}
lane_done(){  # method clip  (same test as debug_chunk.sbatch)
  local run=$BENCH_OUT/$1/safe/$2
  [ -f "$run/RUN_DONE" ] || return 1
  [ "$1" = emernerf ] && return 0
  [ "$BENCH_RES/lidar/lidar_${1}_safe__${2}.json" -nt "$run/TRAIN_DONE" ]
}
run_one(){  # gpu method clip  (subshell: its own env and its own TERM handler)
  local g=$1 m=$2 c=$3 LANE= failed
  trap '[ -n "$LANE" ] && kill -TERM $LANE 2>/dev/null; wait $LANE 2>/dev/null; exit 143' TERM
  sleep $((g * 20))                       # stagger the image loads (host RAM / Lustre)
  lane_once "$g" "$m" "$c"; local rc0=$?
  # Work stealing: when this GPU's lane is finished, take the next unfinished lane of the full bench
  # plan that nobody holds (the lane lock arbitrates; BUSY -> try the next one) instead of idling until
  # the wall clock. Whatever is cut off is resumed by the next chunk. STEAL=0 disables it.
  [ "${STEAL:-1}" = 1 ] || return 0
  failed=" "; case $rc0 in 0|75) ;; *) failed=" $m/$c " ;; esac   # a failed lane is not retried here
  while :; do
    local next= busy=0
    while read -r _ _ m2 c2; do
      [[ "$failed" == *" $m2/$c2 "* ]] && continue
      lane_done "$m2" "$c2" && continue
      [ -n "$NO_UNISIM" ] && [ "$m2" = unisim ] && continue
      local hb=$BENCH_OUT/$m2/safe/$c2/.lane_lock/heartbeat
      if [ -f "$hb" ] && [ $(( $(date +%s) - $(stat -c %Y "$hb" 2>/dev/null || echo 0) )) -lt "${LOCK_STALE_S:-300}" ]; then
        busy=$((busy + 1)); continue
      fi
      next="$m2 $c2"; break
    done < <(awk '$1 !~ /^#/ && NF == 4' "$STEAL_PLAN")
    if [ -z "$next" ]; then
      # Everything left is held elsewhere. A holder that died leaves a lock that goes stale after
      # LOCK_STALE_S, so look again later instead of leaving this GPU idle for the rest of the job.
      [ "$busy" -gt 0 ] || { say "gpu $g: nothing left to steal"; return 0; }
      say "gpu $g: $busy unfinished lanes all held elsewhere -- rescanning in 5 min"
      sleep 300 & LANE=$!; wait $LANE; LANE=; continue
    fi
    read -r m2 c2 <<< "$next"
    say "gpu $g steals $m2/$c2"
    lane_once "$g" "$m2" "$c2"
    case $? in 0|75) ;; *) failed="$failed $m2/$c2 " ;; esac
  done
}

PIDS=()
for line in "${MINE[@]}"; do
  read -r _ g m c <<< "$line"
  run_one "$g" "$m" "$c" & PIDS+=($!)
done
on_term(){
  say "!! SIGTERM (wall clock): stopping lanes; the next chunk resumes them"
  kill -TERM "${PIDS[@]}" 2>/dev/null
  for _ in $(seq 1 90); do
    alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 2
  done
  exit 143
}
trap on_term TERM
wait "${PIDS[@]}"
say "all lanes of this node finished"
