#!/bin/bash
#SBATCH --job-name=ff_p3smoke
#SBATCH --account=a0195
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --output=/capstor/scratch/cscs/yitiwang/foggyfields/logs/p3smoke_%j.log
# T-P3-5 smoke (post-validate v2): 500-iter V1-shaped run on NFF-15.
srun --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml bash -c '
set -o pipefail
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
FF=/capstor/scratch/cscs/yitiwang/foggyfields
source $W/jobs/env.sh >/dev/null
cd "$(cd -P "$CODE/neurad-studio" && pwd)"
SEEDCKPT=$FF/p3/_seed/seeded_NFF-15.ckpt
CUDA_VISIBLE_DEVICES=0 python $FF/jobs/p3_seed_ckpt.py \
  --ckpt "$(find $W/outputs_fix/neurad/safe/NFF-15 -name step-000029999.ckpt | head -1)" \
  --head $FF/heads/NFF-15/rgb_head_NFF-15.pt --out $SEEDCKPT || { echo "seed FAILED"; exit 1; }
export FOGGY_EXTRA="--pipeline.model.use-temporal-appearance False --load-optimizer False \
 --pipeline.model.medium.anchor-beta 0.0321 --pipeline.model.medium.anchor-until 200 \
 --pipeline.model.medium.alternate-from 350 --pipeline.model.medium.alternate-scene-steps 50 \
 --pipeline.model.medium.alternate-medium-steps 25 --pipeline.model.medium.beta-drift-weight 0.01 \
 --pipeline.model.medium.trusted-mask True --pipeline.model.medium.lidar-mixture True \
 --pipeline.model.medium.mixture-mask-from 250 --pipeline.model.medium.freeze-theta-at 200 \
 --pipeline.model.medium.freeze-eta-at 350 --pipeline.model.medium.beta0-init 0.0321"
export PYTHONPATH=$FF/medium${PYTHONPATH:+:$PYTHONPATH}
rm -rf $FF/p3/smoke $FF/p3/smoke_res
OUT_ROOT=$FF/p3/smoke RES_ROOT=$FF/p3/smoke_res VIZ_ROOT=$FF/p3/smoke_viz LOG_ROOT=$FF/logs \
  TIMINGS=$FF/p3/smoke_res/timings.csv DATA_ROOT=$W/data_nff SEED_CKPT=$SEEDCKPT \
  ITERS=500 MAX_EVAL_FRAMES=4 SAVE_EVERY=400 CUDA_VISIBLE_DEVICES=0 \
  bash $W/jobs/lane.sh neurad-foggy safe NFF-15 0 2>&1 | tail -45
echo "p3smoke done"'
