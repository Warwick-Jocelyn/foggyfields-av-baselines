# Run-directory lock, sourced by jobs/lane.sh and jobs/emernerf/lane_emernerf.sh after RUN and log()
# are defined.  lane_lock <run dir>
#
# The same run can be picked up by a debug chunk (jobs/debug_chunk.sbatch) and by a normal-partition
# matrix job at the same time; two trainings writing one run dir would corrupt both. The lock is a
# directory (mkdir is atomic on Lustre, across nodes) with an owner line and a heartbeat file touched
# every 60 s. A lock whose heartbeat is older than LOCK_STALE_S is taken over (its holder died
# without the EXIT trap, e.g. SIGKILL at the wall clock).
# A busy run exits 75 (EX_TEMPFAIL); the drivers report that as BUSY, not as a failure.
LOCK_STALE_S=${LOCK_STALE_S:-300}
lane_lock(){
  LOCK=$1/.lane_lock
  local hb= age
  if ! mkdir "$LOCK" 2>/dev/null; then
    hb=$(stat -c %Y "$LOCK/heartbeat" 2>/dev/null || stat -c %Y "$LOCK" 2>/dev/null)
    if [ -z "$hb" ]; then   # released between our mkdir and stat (holder's EXIT trap): just retry once
      mkdir "$LOCK" 2>/dev/null || { log "BUSY: lock changed hands while checking -- skipped"; exit 75; }
      hb=fresh
    fi
  fi
  if [ "$hb" != fresh ] && [ -n "$hb" ]; then
    age=$(( $(date +%s) - hb ))
    if [ "$age" -lt "$LOCK_STALE_S" ]; then
      log "BUSY: held by $(cat "$LOCK/owner" 2>/dev/null || echo '?') (heartbeat ${age}s ago) -- skipped"
      exit 75
    fi
    log "stale lock (heartbeat ${age}s ago, owner $(cat "$LOCK/owner" 2>/dev/null || echo '?')) -> taking over"
    rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || { log "BUSY: lost the takeover race -- skipped"; exit 75; }
  fi
  echo "job=${SLURM_JOB_ID:-local} host=$(hostname) pid=$$ since=$(date -Is)" > "$LOCK/owner"
  touch "$LOCK/heartbeat"
  # (detached from the lane log; its sleep is killed with it so nothing lingers after the lane exits)
  ( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat" 2>/dev/null; sleep 60; done ) >/dev/null 2>&1 &
  LOCK_HB=$!
  trap 'pkill -P $LOCK_HB 2>/dev/null; kill $LOCK_HB 2>/dev/null; rm -rf "$LOCK"' EXIT
}
