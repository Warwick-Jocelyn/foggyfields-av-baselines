#!/bin/bash
#SBATCH --job-name=nffdiag
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --signal=TERM@240
#SBATCH --open-mode=append
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_nffdiag/run_%x_%j.log
# SplatAD fog diagnosis: 4 lanes (one per GPU) from $PLAN. Partition/time given at submit time
# (debug 1:30, or preemptable with --requeue: lane.sh resumes from the newest checkpoint).
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
unset ITERS END_FRACTION MAX_EVAL_FRAMES SPLATAD_EXTRA FORCE SAVE_EVERY
: "${PLAN:?PLAN=<plan file> required}"
echo "[$(date -Is)] job $SLURM_JOB_ID on $SLURM_JOB_NODELIST restart=${SLURM_RESTART_COUNT:-0} plan $PLAN"; cat "$PLAN"
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash "$W/jobs/nffdiag/node.sh" &
SRUN=$!
trap 'echo "[$(date -Is)] SIGTERM"; wait $SRUN' TERM
wait $SRUN; echo "[$(date -Is)] srun rc=$?"
