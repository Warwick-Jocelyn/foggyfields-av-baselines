#!/bin/bash
#SBATCH --job-name=fixbench
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --signal=TERM@240
#SBATCH --open-mode=append
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_fix/run_%x_%j.log
# Corrected bench (NeuRAD / UniSim / EmerNeRF after the 2026-10-25 audit): 4 lanes from $PLAN.
# (debug 1:30, or preemptable with --requeue: lane.sh resumes from the newest checkpoint).
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
unset END_FRACTION SPLATAD_EXTRA FORCE SAVE_EVERY NEURAD_EXTRA NEURAD_DP_EXTRA UNISIM_EXTRA UNISIM_DP_EXTRA
[ "${FIX_TAG:-fix}" = smoke ] || unset ITERS MAX_EVAL_FRAMES   # short budgets only for smoke runs
: "${PLAN:?PLAN=<plan file> required}"
echo "[$(date -Is)] job $SLURM_JOB_ID on $SLURM_JOB_NODELIST restart=${SLURM_RESTART_COUNT:-0} plan $PLAN"; cat "$PLAN"
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash "$W/jobs/fixbench/node.sh" &
SRUN=$!
trap 'echo "[$(date -Is)] SIGTERM"; wait $SRUN' TERM
wait $SRUN; echo "[$(date -Is)] srun rc=$?"
