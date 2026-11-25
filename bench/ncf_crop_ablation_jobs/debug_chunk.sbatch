#!/bin/bash
#SBATCH --job-name=bench_dbg
#SBATCH --partition=debug
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
# (--exclusive/--mem=800G: debug hands out whole nodes anyway, but preemptable/normal allocate only what is
#  asked -- without these a job got cpu=1 and its four lanes stalled in image loading.)
#SBATCH --time=01:30:00
#SBATCH --signal=TERM@240
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_bench/debug_chunk_%j.log
#
# One 1.5 h chunk of the safe_crop cross-method bench (NeuRAD / UniSim / EmerNeRF x 5 clips) on the
# debug partition, which -- unlike normal -- is not behind the 400-node QOS cap.
# Lanes: jobs/bench_plan.txt (one per GPU, 15 lanes on 4 nodes); per node: jobs/debug_node.sh.
# Submit the same script again (or chain with --dependency=afterany:<id>) until every lane is DONE:
# finished stages are skipped and trainings resume from their newest checkpoint, so a chain of
# chunks trains exactly the full iteration budget of each method.
#
# This script runs on the HOST (no #SBATCH --environment) so that srun can start one container
# per node; --signal=TERM@240 (no B:) reaches every node's debug_node.sh.
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
TOML=/users/yitiwang/infra/projects/splatad-nff-repro.toml
export BENCH_DATA=${BENCH_DATA:-$W/data_all}
export BENCH_OUT=${BENCH_OUT:-$W/outputs_bench}
export BENCH_RES=${BENCH_RES:-$W/results_bench}
export BENCH_VIZ=${BENCH_VIZ:-$W/viz_bench}
export BENCH_LOG=${BENCH_LOG:-$W/logs_bench}
export BENCH_LOG_EMER=${BENCH_LOG_EMER:-$W/logs_emernerf}
export PLAN=${PLAN:-$W/jobs/bench_plan.txt}
export DUMP_CLIPS=${DUMP_CLIPS:-"NCF-02 NFF-05"}
# The iteration budgets come from env.sh / env_emernerf.sh; a leaked smoke override must not reach
# them. (BENCH_KEEP_OVERRIDES=1 is for the resume test only, with its own BENCH_* roots.)
[ "$BENCH_KEEP_OVERRIDES" = 1 ] || \
  unset ITERS END_FRACTION MAX_EVAL_FRAMES NEURAD_EXTRA NEURAD_DP_EXTRA UNISIM_EXTRA UNISIM_DP_EXTRA EMER_EXTRA FORCE SAVE_EVERY
mkdir -p "$BENCH_LOG" "$BENCH_RES"
# debug-qos caps a job at 90 node-minutes (MaxTRESMinsPerJob node=90): 4 nodes x 01:30 never starts
# (PENDING, QOSMaxNodeMinutesPerJob). Within 1.5 h only --nodes=1 fits; PLAN_NODE=k picks the plan rows.
[ -n "$PLAN_NODE" ] && [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ] && { echo "PLAN_NODE=$PLAN_NODE needs --nodes=1"; exit 2; }

# ---- automatic plan + self-chaining (default when PLAN_NODE is not given) ----------------------
# Pick up to 4 lanes of $PLAN that are neither finished nor held by another job (jobs/lanelock.sh),
# runs already in progress first, and run them as node 0 of a one-node plan. Then queue the next
# chunk behind this one (afterany), so the chain keeps going until nothing is left.
# Stop the chain:  touch $BENCH_LOG/STOP_CHAIN   (CHAIN_LEFT also bounds it).
if [ -z "$PLAN_NODE" ]; then
  lane_done(){  # method clip
    local run=$BENCH_OUT/$1/safe/$2
    [ -f "$run/RUN_DONE" ] || return 1
    [ "$1" = emernerf ] && return 0
    [ "$BENCH_RES/lidar/lidar_${1}_safe__${2}.json" -nt "$run/TRAIN_DONE" ]
  }
  lane_busy(){  # method clip
    local hb=$BENCH_OUT/$1/safe/$2/.lane_lock/heartbeat
    [ -f "$hb" ] && [ $(( $(date +%s) - $(stat -c %Y "$hb") )) -lt "${LOCK_STALE_S:-300}" ]
  }
  STARTED=(); FRESH=(); NLEFT=0
  while read -r _ _ m c; do
    lane_done "$m" "$c" && continue
    NLEFT=$((NLEFT + 1))
    lane_busy "$m" "$c" && { echo "  busy elsewhere: $m/$c"; continue; }
    if [ -f "$BENCH_OUT/$m/safe/$c/TRAIN_SIG" ]; then STARTED+=("$m $c"); else FRESH+=("$m $c"); fi
  done < <(awk '$1 !~ /^#/ && NF == 4' "$PLAN")
  PICK=("${STARTED[@]}" "${FRESH[@]}"); PICK=("${PICK[@]:0:4}")
  echo "[$(date -Is)] $NLEFT lanes not finished; this chunk takes ${#PICK[@]}: $(printf '%s; ' "${PICK[@]}")"
  [ ${#PICK[@]} -gt 0 ] || { echo "nothing to run here -- chain ends"; exit 0; }
  AUTO=$BENCH_LOG/plan_${SLURM_JOB_ID}.txt; : > "$AUTO"
  for i in "${!PICK[@]}"; do echo "0 $i ${PICK[$i]}" >> "$AUTO"; done
  export PLAN=$AUTO PLAN_NODE=0
  CHAIN_LEFT=${CHAIN_LEFT:-12}
  if [ -f "$BENCH_LOG/STOP_CHAIN" ]; then echo "STOP_CHAIN present -- not queueing a successor"
  elif [ "$CHAIN_LEFT" -le 1 ]; then echo "CHAIN_LEFT exhausted -- not queueing a successor"
  else
    NEXT=$(env -u PLAN -u PLAN_NODE sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
             --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) "$W/jobs/debug_chunk.sbatch") \
      && echo "queued successor $NEXT (CHAIN_LEFT=$((CHAIN_LEFT - 1)))" || echo "!! could not queue a successor"
  fi
fi

echo "[$(date -Is)] job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST plan $PLAN"
srun --environment="$TOML" --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
     bash "$W/jobs/debug_node.sh" &
SRUN=$!
trap 'echo "[$(date -Is)] SIGTERM to the batch shell"; wait $SRUN' TERM
wait $SRUN; rc=$?
echo "[$(date -Is)] srun rc=$rc"
[ -s "$BENCH_LOG/FAILURES_${SLURM_JOB_ID}.txt" ] && { echo "failures:"; cat "$BENCH_LOG/FAILURES_${SLURM_JOB_ID}.txt"; }
echo "stamps:"; for d in "$BENCH_OUT"/{neurad,unisim,emernerf}/safe/*; do
  [ -d "$d" ] && echo "  ${d#$BENCH_OUT/}: $(cd "$d" && ls -1 *_DONE 2>/dev/null | tr '\n' ' ')"
done
exit $rc
