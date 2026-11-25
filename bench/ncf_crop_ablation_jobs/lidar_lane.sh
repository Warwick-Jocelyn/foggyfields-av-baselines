#!/bin/bash
# LiDAR eval of a finished neurad/unisim lane (lidar_dump_ns.py), run by jobs/debug_node.sh after lane.sh.
#   lidar_lane.sh <method> <clip> <gpu>      (env: BENCH_OUT BENCH_RES BENCH_LOG BENCH_DATA [DUMP_CLIPS] [FORCE_LIDAR=1])
# Holds the run-dir lock (jobs/lanelock.sh) like lane.sh does: lane.sh releases it at RUN_DONE, and a job
# that stole the lane in that gap used to start a second LiDAR eval writing the same JSON/log/dump dir.
# The JSON and the dump are written next to their target and moved into place only when complete, so a
# wall-clock kill never leaves a result that the "newer than TRAIN_DONE" test would take as done.
set -o pipefail
m=$1 c=$2 g=$3
JOBS=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs
: "${BENCH_OUT:?}" "${BENCH_RES:?}" "${BENCH_LOG:?}" "${BENCH_DATA:?}"
RUN=$BENCH_OUT/$m/safe/$c
log(){ echo "[$(date '+%F %T')] [lidar $m/$c gpu $g] $*"; }
source "$JOBS/lanelock.sh"; lane_lock "$RUN"
[ -f "$RUN/TRAIN_DONE" ] || { log "!! no TRAIN_DONE"; exit 1; }
res=$BENCH_RES/lidar/lidar_${m}_safe__${c}.json
# (newer than TRAIN_DONE: a retrained run must not keep the LiDAR numbers of the previous model)
if [ "${FORCE_LIDAR:-0}" != 1 ] && [ -s "$res" ] && [ "$res" -nt "$RUN/TRAIN_DONE" ]; then log "SKIP (done)"; exit 0; fi
mkdir -p "$BENCH_RES/lidar"
args=(--config "$(head -1 "$RUN/TRAIN_DONE")" --output "$res.tmp" --n-cameras 70 --set "data=$BENCH_DATA")
dump=
for dc in ${DUMP_CLIPS:-}; do [ "$dc" = "$c" ] && dump=$BENCH_OUT/_lidar_dump/${m}_safe__${c}; done
[ -n "$dump" ] && { rm -rf "$dump.tmp"; args+=(--dump "$dump.tmp"); }
rm -f "$res.tmp"
log "start (dump: ${dump:-no})"
( source "$JOBS/env.sh" >/dev/null; cd "$(cd -P "$CODE/neurad-studio" && pwd)" && \
  CUDA_VISIBLE_DEVICES=$g exec python "$JOBS/lidar_dump_ns.py" "${args[@]}" ) & CHILD=$!
trap 'log "!! SIGTERM: stopping"; kill -TERM $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; exit 143' TERM
wait $CHILD; rc=$?
[ $rc -eq 0 ] && [ -s "$res.tmp" ] || { log "!! lidar_dump_ns.py rc=$rc"; exit $(( rc ? rc : 1 )); }
if [ -n "$dump" ]; then rm -rf "$dump"; mv "$dump.tmp" "$dump"; fi
mv "$res.tmp" "$res"
log "done -> $res"
