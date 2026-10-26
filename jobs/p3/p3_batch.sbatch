#!/bin/bash
#SBATCH --job-name=ff_p3
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --requeue
#SBATCH --output=/capstor/scratch/cscs/yitiwang/foggyfields/logs/p3_%j.log
# T-P3-6: V0 (B0' ref, medium off) / V1 (fog-ckpt init, staged 15k) / V2 (fresh, staged 30k)
# / V3 (negative control: V1 without the anchor). lane.sh resume-chain safe under preemption.
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash -c '
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
FF=/capstor/scratch/cscs/yitiwang/foggyfields
source $W/jobs/env.sh >/dev/null
cd "$(cd -P "$CODE/neurad-studio" && pwd)"
SEEDCKPT=$FF/p3/_seed/seeded_NFF-15.ckpt
[ -f $SEEDCKPT ] || CUDA_VISIBLE_DEVICES=0 python $FF/jobs/p3_seed_ckpt.py \
  --ckpt "$(find $W/outputs_fix/neurad/safe/NFF-15 -name step-000029999.ckpt | head -1)" \
  --head $FF/heads/NFF-15/rgb_head_NFF-15.pt --out $SEEDCKPT
sched(){ n=$1; for g in trajectory_opt cnn fields hashgrids camera_opt rgb_head medium; do
  printf -- "--optimizers.%s.scheduler.max-steps %s " $g $n; done; }
BASE="--pipeline.model.use-temporal-appearance False --load-optimizer False"
V1SCHED="--pipeline.model.medium.anchor-beta 0.0321 --pipeline.model.medium.anchor-until 6000 \
 --pipeline.model.medium.alternate-from 9000 --pipeline.model.medium.beta-drift-weight 0.01 \
 --pipeline.model.medium.trusted-mask True --pipeline.model.medium.lidar-mixture True \
 --pipeline.model.medium.mixture-mask-from 7000 --pipeline.model.medium.freeze-theta-at 6000 \
 --pipeline.model.medium.freeze-eta-at 9000 --pipeline.model.medium.beta0-init 0.0321"
V2SCHED="--pipeline.model.medium.anchor-beta 0.0321 --pipeline.model.medium.anchor-until 12000 \
 --pipeline.model.medium.alternate-from 18000 --pipeline.model.medium.beta-drift-weight 0.01 \
 --pipeline.model.medium.trusted-mask True --pipeline.model.medium.lidar-mixture True \
 --pipeline.model.medium.mixture-mask-from 13000 --pipeline.model.medium.freeze-theta-at 12000 \
 --pipeline.model.medium.freeze-eta-at 18000 --pipeline.model.medium.beta0-init 0.0321"
V3SCHED="--pipeline.model.medium.anchor-until 0 --pipeline.model.medium.alternate-from 9000 \
 --pipeline.model.medium.trusted-mask True --pipeline.model.medium.lidar-mixture True \
 --pipeline.model.medium.mixture-mask-from 7000 --pipeline.model.medium.freeze-theta-at 6000 \
 --pipeline.model.medium.freeze-eta-at 9000 --pipeline.model.medium.beta0-init 0.0321"
export PYTHONPATH=$FF/medium${PYTHONPATH:+:$PYTHONPATH}
lane(){ v=$1 g=$2 iters=$3 seed=$4 extra=$5
  FOGGY_EXTRA="$BASE $(sched $iters) $extra" FOGGY_ITERS=$iters SEED_CKPT=$seed \
  OUT_ROOT=$FF/p3/$v RES_ROOT=$FF/p3/${v}_res VIZ_ROOT=$FF/p3/${v}_viz LOG_ROOT=$FF/logs \
  TIMINGS=$FF/p3/${v}_res/timings.csv DATA_ROOT=$W/data_nff SAVE_EVERY=2000 \
  CUDA_VISIBLE_DEVICES=$g bash $W/jobs/lane.sh neurad-foggy safe NFF-15 $g \
    > $FF/logs/p3_${v}.log 2>&1
  echo "p3 $v rc=$?"; }
lane V0 0 30000 ""         "--pipeline.model.medium.enabled False" &
lane V1 1 15000 $SEEDCKPT  "$V1SCHED" &
lane V2 2 30000 ""         "$V2SCHED" &
lane V3 3 15000 $SEEDCKPT  "$V3SCHED" &
wait
echo ALL_DONE'
