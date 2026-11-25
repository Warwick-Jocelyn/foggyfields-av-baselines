#!/bin/bash
#SBATCH --job-name=finalbase
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --signal=TERM@240
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_final/run_%x_%j.log
# Final-ablation external baselines + common evaluator (decision 109). 4 GPU chains from $PLAN.
# preemptable: a requeued job resumes (lane.sh / lane_emernerf.sh resume from the newest checkpoint,
# v13 lanes skip finished frames). Submitted on two accounts: node_final.sh's owner lock makes the loser exit.
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
[ "${FINAL_TAG:-}" = smoke ] || unset ITERS MAX_EVAL_FRAMES   # short budgets only for smoke runs
unset END_FRACTION SPLATAD_EXTRA SPLATAD_DP_EXTRA FORCE NEURAD_EXTRA NEURAD_DP_EXTRA VENV NS_REPO DATA_ROOT OUT_ROOT RES_ROOT VIZ_ROOT LOG_ROOT
: "${PLAN:?PLAN=<plan file> required}"
echo "[$(date -Is)] job $SLURM_JOB_ID acct $SLURM_JOB_ACCOUNT on $SLURM_JOB_NODELIST restart=${SLURM_RESTART_COUNT:-0} plan $PLAN"; cat "$PLAN"
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash "$W/jobs/final/node_final.sh" &
SRUN=$!
trap 'echo "[$(date -Is)] SIGTERM"; wait $SRUN' TERM
wait $SRUN; RC=$?; echo "[$(date -Is)] srun rc=$RC"
# Continuation (CHAIN=1): the lanes resume from their checkpoints, so a plan that did not finish inside this
# chunk is resubmitted with the same arguments (same account/partition/time); duplicates that exited on the
# owner lock and finished plans are not chained.
PN=$(basename "$PLAN" .txt)${FINAL_TAG:+_$FINAL_TAG}
if [ "${CHAIN:-0}" = 1 ] && [ "${CHAIN_N:-0}" -lt 8 ] && [ ! -f "$W/outputs_final${FINAL_TAG:+_$FINAL_TAG}/.plan_done_$PN" ] && [ ! -f "$W/outputs_final/.dup_exit_$SLURM_JOB_ID" ]; then
  TL=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | head -1); [ -n "$TL" ] || TL=03:00:00
  NEXT=$(sbatch --parsable -A "$SLURM_JOB_ACCOUNT" -p "${SLURM_JOB_PARTITION:-preemptable}" -t "$TL" --job-name="$SLURM_JOB_NAME" \
         --export=ALL,PLAN=$PLAN,CHAIN=1,CHAIN_N=$(( ${CHAIN_N:-0} + 1 ))${FINAL_TAG:+,FINAL_TAG=$FINAL_TAG} "$W/jobs/final/run_final.sbatch" 2>&1)
  echo "[$(date -Is)] plan $PN not finished -> continuation job $NEXT"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) job $NEXT $SLURM_JOB_ACCOUNT/${SLURM_JOB_PARTITION:-preemptable} 1 node <= $TL: continuation chunk of $PN after job $SLURM_JOB_ID [final-ablation]" >> /users/yitiwang/projects/foggyfields-v1/runs/registry_allocations.txt
fi
rm -f "$W/outputs_final/.dup_exit_$SLURM_JOB_ID"
