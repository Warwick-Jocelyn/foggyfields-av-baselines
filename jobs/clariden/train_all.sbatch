#!/bin/bash
#SBATCH --job-name=splatad_nff
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/splatad-nff-repro/logs/train_all_%j.log
# 4 lanes in parallel (one GPU each): faithful×2 clips, corrected×2 clips; then the 2 HF-checkpoint evals; then collect + mirror to store.
set -o pipefail
source /capstor/scratch/cscs/yitiwang/splatad-nff-repro/jobs/env.sh
echo "[$(date)] node=$(hostname)"; nvidia-smi -L
if python -c "import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then export VIS=wandb; else export VIS=tensorboard; fi
echo "VIS=$VIS"; mkdir -p $W/logs/lanes
L=$W/logs/lanes
# ---- pre-flight (fail fast before occupying 4 GPUs for an hour) ----
cd $CODE/neurad-studio
python $W/jobs/dataparser_check.py --data "$DATA_ROOT" --sequence 003 --lidar-points-frame sensor --use-valid-masks || { echo "!! pre-flight dataparser (003, corrected) failed"; exit 3; }
PF=$OUT_ROOT/_preflight; rm -rf "$PF"; mkdir -p "$PF"
CUDA_VISIBLE_DEVICES=0 timeout 900 ns-train splatad --output-dir "$PF" --vis tensorboard --max-num-iterations 5 --steps-per-eval-image 999999 \
  --steps-per-eval-all-images 999999 --steps-per-save 5 --machine.seed 42 --viewer.quit-on-train-completion True \
  --pipeline.model.strategy mcmc --pipeline.model.mcmc-cap-max 5000000 --pipeline.model.max-num-seed-points 2000000 \
  nvidia-av-fog-data --data "$DATA_ROOT" --sequence 003 --train-split-fraction 0.5 --use-valid-masks True --lidar-points-frame sensor \
  || { echo "!! pre-flight ns-train (corrected flags) failed"; exit 4; }
echo "[$(date)] pre-flight OK"
bash $W/jobs/lane.sh faithful 004 NFF-03 0 > $L/faithful_NFF-03.log 2>&1 &
bash $W/jobs/lane.sh faithful 003 NFF-02 1 > $L/faithful_NFF-02.log 2>&1 &
bash $W/jobs/lane.sh corrected 004 NFF-03 2 > $L/corrected_NFF-03.log 2>&1 &
bash $W/jobs/lane.sh corrected 003 NFF-02 3 > $L/corrected_NFF-02.log 2>&1 &
wait; echo "[$(date)] training lanes finished"
bash $W/jobs/lane.sh hf_ckpt 004 NFF-03 0 $W/ckpts/clip004_highway-heavy-fog     > $L/hf_ckpt_NFF-03.log 2>&1 &
bash $W/jobs/lane.sh hf_ckpt 003 NFF-02 1 $W/ckpts/clip003_residential-light-fog > $L/hf_ckpt_NFF-02.log 2>&1 &
wait; echo "[$(date)] hf_ckpt lanes finished"
python $W/jobs/collect_results.py
# ---- mirror keepers to durable store (checkpoints, configs, metrics, logs, videos; NOT renders) ----
STORE=/capstor/store/cscs/swissai/a0195/yitiwang/foggyfields/repro-2026-10; mkdir -p "$STORE"
rsync -a --prune-empty-dirs --include='*/' --include='*.ckpt' --include='config.yml' --include='*.json' --include='*.csv' --include='*.mp4' --include='LANE_DONE' --exclude='*' $W/outputs/ "$STORE/outputs/"
rsync -a $W/jobs $W/logs $W/RESULTS.md $W/results.json "$STORE/" 2>/dev/null
echo "[$(date)] ALL DONE — results: $W/RESULTS.md ; mirrored to $STORE"; echo "$(date -Is)" > $W/outputs/ALL_DONE
