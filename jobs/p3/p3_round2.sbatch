#!/bin/bash
#SBATCH --job-name=ff_p3r2
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=800G
#SBATCH --requeue
#SBATCH --output=/capstor/scratch/cscs/yitiwang/foggyfields/logs/p3r2_%j.log
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash -c '
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
FF=/capstor/scratch/cscs/yitiwang/foggyfields
source $W/jobs/env.sh >/dev/null
cd "$(cd -P "$CODE/neurad-studio" && pwd)"
sched(){ for g in trajectory_opt cnn fields hashgrids camera_opt rgb_head medium; do
  printf -- "--optimizers.%s.scheduler.max-steps 30000 " $g; done; }
BASE="--pipeline.model.use-temporal-appearance False --load-optimizer False \
 --pipeline.model.medium.anchor-beta 0.0321 --pipeline.model.medium.anchor-until 12000 \
 --pipeline.model.medium.alternate-from 18000 --pipeline.model.medium.beta-drift-weight 0.01 \
 --pipeline.model.medium.trusted-mask True --pipeline.model.medium.beta0-init 0.0321 \
 --pipeline.model.medium.alpha-bimodal-weight 0.001"
NOMIX="--pipeline.model.medium.lidar-mixture False"
PINMIX="--pipeline.model.medium.lidar-mixture True --pipeline.model.medium.mixture-mask-from 0 \
 --pipeline.model.medium.freeze-theta-at 1 --pipeline.model.medium.freeze-eta-at 1 \
 --pipeline.model.medium.eta-init 0.0154 --pipeline.model.medium.theta-init 1e-6"
export PYTHONPATH=$FF/medium${PYTHONPATH:+:$PYTHONPATH}
lane(){ v=$1 g=$2 seedval=$3 extra=$4
  FOGGY_EXTRA="$BASE $(sched) $extra" FOGGY_ITERS=30000 SEED=$seedval \
  OUT_ROOT=$FF/p3/$v RES_ROOT=$FF/p3/${v}_res VIZ_ROOT=$FF/p3/${v}_viz LOG_ROOT=$FF/logs \
  TIMINGS=$FF/p3/${v}_res/timings.csv DATA_ROOT=$W/data_nff SAVE_EVERY=2000 \
  CUDA_VISIBLE_DEVICES=$g bash $W/jobs/lane.sh neurad-foggy safe NFF-15 $g \
    > $FF/logs/p3_${v}.log 2>&1
  echo "p3r2 $v rc=$?"; }
lane V2b   0 42 "$NOMIX" &
lane V2c   1 42 "$PINMIX" &
lane V2b-s1 2 1042 "$NOMIX" &
lane V2b-s2 3 2042 "$NOMIX" &
wait
echo ALL_DONE'
