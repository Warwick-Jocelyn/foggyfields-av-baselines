#!/bin/bash
# One SplatAD lane: VARIANT in {faithful, corrected, hf_ckpt}; SEQ in {003,004}; CLIP = NFF-02/NFF-03; GPU index.
#   faithful : original code+config (nvs_50_50 via linspace, no mask)   -> reproduce published numbers
#   corrected: valid-pixel masks in RGB loss + lidar sweeps read in their true (sensor) frame
#   hf_ckpt  : no training; evaluate the published HF checkpoint on NFF  -> dataset-equivalence check
set -eo pipefail
VARIANT=$1; SEQ=$2; CLIP=$3; GPU=${4:-0}; HFDIR=${5:-}
source /capstor/scratch/cscs/yitiwang/splatad-nff-repro/jobs/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
CLIPDIR=$(readlink -f "$DATA_ROOT"/processed/${SEQ}_*/processed_pinhole)
OUT=$OUT_ROOT/$VARIANT/$CLIP; mkdir -p "$OUT"; TIM=$OUT_ROOT/_timings/timings.csv; mkdir -p "$(dirname "$TIM")"
log(){ echo "[$(date '+%F %T')] [$VARIANT/$CLIP] $*"; }
cd $CODE/neurad-studio

if [ "$VARIANT" = "hf_ckpt" ]; then
  log "prepare published config from $HFDIR"
  python $W/jobs/prep_hf_config.py --src "$HFDIR" --dst "$OUT" --data "$DATA_ROOT"
  CFG="$OUT/config.yml"
else
  EXTRA=(); [ "$VARIANT" = "corrected" ] && EXTRA=(--use-valid-masks True --lidar-points-frame sensor)
  log "STEP 1 ns-train splatad 30k iters (vis=$VIS)"
  T0=$(date +%s); RC=0
  ns-train splatad --output-dir "$OUT" --experiment-name "splatad_${VARIANT}_${CLIP}" --vis "$VIS" \
    --max-num-iterations 30000 --machine.seed 42 --steps-per-eval-image 999999 --steps-per-eval-all-images 999999 \
    --steps-per-save 30000 --viewer.quit-on-train-completion True \
    --pipeline.model.strategy mcmc --pipeline.model.mcmc-cap-max 5000000 --pipeline.model.max-num-seed-points 2000000 \
    nvidia-av-fog-data --data "$DATA_ROOT" --sequence "$SEQ" --train-split-fraction 0.5 "${EXTRA[@]}" || RC=$?
  T1=$(date +%s); echo "splatad_$VARIANT,$CLIP,30000,train_sec,$((T1-T0)),rc=$RC" >> "$TIM"
  [ $RC -eq 0 ] || { log "!! ns-train failed rc=$RC"; exit $RC; }
  RUN_DIR=$(find "$OUT" -type d -name nerfstudio_models -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2-)
  CFG="$RUN_DIR/config.yml"
  ls "$RUN_DIR/nerfstudio_models"/step-*.ckpt >/dev/null || { log "!! no checkpoint written"; exit 2; }
fi
log "config: $CFG"

log "STEP 2 render val (rgb, gt-rgb, depth) + raw-depth"
T0=$(date +%s)
ns-render dataset --load-config "$CFG" --output-path "$OUT/renders" --image-format png --rendered-output-names rgb gt-rgb depth --pose-source val
ns-render dataset --load-config "$CFG" --output-path "$OUT/renders_rawdepth" --rendered-output-names raw-depth --pose-source val
T1=$(date +%s); echo "splatad_$VARIANT,$CLIP,render,sec,$((T1-T0))," >> "$TIM"

log "STEP 3 metrics"
python $W/jobs/rgb_metrics.py "$OUT/renders/val" --masks-root "$CLIPDIR/masks" --output "$OUT/rgb_metrics.json"
python $W/jobs/lidar_metrics.py --clip "$CLIPDIR" --rawdepth-root "$OUT/renders_rawdepth/val/raw-depth" --output "$OUT/lidar_metrics.json" || log "!! lidar metrics failed"

log "STEP 4 panorama video"
python $REPO/jobs/pngs_to_mp4_nvidia_fog.py "$OUT/renders" "$OUT/videos" --split val --fps 10 --panorama-only || log "!! video failed (non-fatal)"
echo "$(date -Is)" > "$OUT/LANE_DONE"; log "LANE DONE"
