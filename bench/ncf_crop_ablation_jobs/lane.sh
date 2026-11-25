#!/bin/bash
# One run of the crop/mask ablation:  lane.sh <method> <variant> <clip> <gpu>
#
#   method   splatad | neurad | unisim
#   variant  safe  = --apply-safe-crop True  --use-valid-masks False   (crop, no black borders)
#            mask  = --apply-safe-crop False --use-valid-masks True    (full canvas, corners out of the loss)
#
# Four stages, each with its own stamp file, each verified by the ARTIFACT it should have
# produced (not by the exit code alone). Re-running skips finished stages, so a job that
# hits the 12 h partition limit can simply be resubmitted.
set -o pipefail
METHOD=$1; VARIANT=$2; CLIP=$3; GPU=${4:-0}
[ -n "$CLIP" ] || { echo "usage: lane.sh <method> <variant> <clip> <gpu>"; exit 2; }
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/env.sh
export CUDA_VISIBLE_DEVICES=$GPU

ITERS_EFF=$(iters_for "$METHOD") || exit 2
RUN=$OUT_ROOT/$METHOD/$VARIANT/$CLIP
VIZ=$VIZ_ROOT/$METHOD/$VARIANT/$CLIP
MET=$RES_ROOT/metrics/metrics_${METHOD}__${VARIANT}__${CLIP}.json
mkdir -p "$RUN" "$VIZ" "$(dirname "$MET")"
TAG="$METHOD/$VARIANT/$CLIP"
log(){ echo "[$(date '+%F %T')] [$TAG] $*"; }
# One process per run dir at a time (a debug chunk and a normal-partition job may both reach it).
source "$JOBS/lanelock.sh"; lane_lock "$RUN"

# Run signature. A stamp is only valid for the SAME configuration that produced it.
# Without this a cheap smoke run (ITERS=500, END_FRACTION=0.15) leaves RUN_DONE behind and
# the real matrix SKIPS that clip, folding the smoke numbers into the published paired mean
# with nothing in the metrics JSON to reveal it.
SIG="iters=${ITERS_EFF};split=${SPLIT:-nvs_50_50};end_fraction=${END_FRACTION:-1.0};max_eval_frames=${MAX_EVAL_FRAMES:-all};variant=${VARIANT}"
sig_ok(){  # $1 = stamp file; true when the stamp carries this exact signature
  [ -f "$1" ] || return 1
  grep -qxF "sig:$SIG" "$1" 2>/dev/null
}
sig_stale(){ [ -f "$1" ] && ! sig_ok "$1"; }

# Clip root, for the split file (resolved the same way the dataparser does).
CLIP_ROOT=$(/users/yitiwang/miniforge3/bin/python3 -I "$JOBS/clip_paths.py" \
            --data "$DATA_ROOT" --sequence "$CLIP" --print-root 2>/dev/null || true)

# timings.csv is appended from up to NLANES processes at once -> take a lock.
tim(){  # stage seconds rc note
  [ -s "$TIMINGS" ] || { flock "$TIMINGS.lock" -c "[ -s '$TIMINGS' ] || echo 'ts,method,variant,clip,stage,seconds,rc,note' > '$TIMINGS'"; }
  flock "$TIMINGS.lock" -c "echo '$(date -Is),$METHOD,$VARIANT,$CLIP,$1,$2,$3,$4' >> '$TIMINGS'"
}
touch "$TIMINGS.lock"

case "$VARIANT" in
  safe) CROP_FLAGS=(--apply-safe-crop True  --use-valid-masks False) ;;
  mask) CROP_FLAGS=(--apply-safe-crop False --use-valid-masks True)  ;;
  *) log "!! unknown variant '$VARIANT' (expected safe|mask)"; exit 2 ;;
esac
DP_EXTRA=(); M_DP_EXTRA=()
[ -n "$MAX_EVAL_FRAMES" ] && DP_EXTRA+=(--max-eval-frames "$MAX_EVAL_FRAMES")
[ -n "$END_FRACTION" ]    && DP_EXTRA+=(--dataset-end-fraction "$END_FRACTION")
case "$METHOD" in
  splatad) M_EXTRA=(--pipeline.model.strategy mcmc --pipeline.model.mcmc-cap-max 5000000
                    --pipeline.model.max-num-seed-points 2000000 $SPLATAD_EXTRA)
           M_DP_EXTRA=($SPLATAD_DP_EXTRA) ;;
  neurad)  M_EXTRA=($NEURAD_EXTRA); M_DP_EXTRA=($NEURAD_DP_EXTRA) ;;
  neurad-foggy) M_EXTRA=($FOGGY_EXTRA); M_DP_EXTRA=($NEURAD_DP_EXTRA) ;;  # FoggyFields P3
  unisim)  M_EXTRA=($UNISIM_EXTRA); M_DP_EXTRA=($UNISIM_DP_EXTRA)
           # UniSim's rgb_loss ignores batch["mask"] (unlike neurad.py), so the mask variant would
           # silently train on the black lens-cone corners.
           [ "$VARIANT" = safe ] || { log "!! unisim supports only VARIANT=safe (its RGB loss ignores masks)"; exit 2; } ;;
  *) log "!! unknown method '$METHOD'"; exit 2 ;;
esac
# The method flags are part of the configuration too: a NeuRAD/UniSim stamp made with different
# model or dataparser flags must not be reused. (SplatAD keeps the old signature when SPLATAD_EXTRA is
# empty, so its finished stamps stay valid.)
case "$METHOD" in
  neurad|unisim|neurad-foggy) SIG="$SIG;extra=$(printf '%s ' "${M_EXTRA[@]}" '|' "${M_DP_EXTRA[@]}" | md5sum | cut -c1-12)" ;;
  splatad) [ -n "$SPLATAD_EXTRA" ] && SIG="$SIG;extra=$(printf '%s ' $SPLATAD_EXTRA | md5sum | cut -c1-12)"
           [ -n "$SPLATAD_DP_EXTRA" ] && SIG="$SIG;dpextra=$(printf '%s ' $SPLATAD_DP_EXTRA | md5sum | cut -c1-12)" ;;
esac

# Heavy stages run as a background child so a SIGTERM (wall-clock limit) is handled at once: the
# child is stopped and the PARTIAL segment goes into timings.csv with rc=143, so the segments of a
# resumed run add up to its true cost. (Same scheme as jobs/emernerf/lane_emernerf.sh.)
CHILD=; CUR_STAGE=; ST0=
run_child(){  # stage cmd...
  CUR_STAGE=$1; shift; ST0=$(date +%s)
  "$@" & CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  CHILD=; CUR_STAGE=; return $rc
}
on_term(){
  log "!! SIGTERM during stage '${CUR_STAGE:-none}'"
  if [ -n "$CHILD" ]; then
    kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null
    tim "$CUR_STAGE" $(( $(date +%s) - ST0 )) 143 "KILLED partial segment (SIGTERM)${RESUMED}; rerun resumes"
  fi
  exit 143
}
trap on_term TERM

log "start  gpu=$GPU iters=$ITERS_EFF split=$SPLIT data=$DATA_ROOT"
log "flags  ${CROP_FLAGS[*]} ${DP_EXTRA[*]} ${M_DP_EXTRA[*]} | model: ${M_EXTRA[*]}"
cd "$NS_REPO" || { log "!! $NS_REPO missing"; exit 2; }

# ---------------------------------------------------------------- 1. train
if sig_stale "$RUN/TRAIN_DONE"; then
  log "train REDO: stamp was made with a different configuration"
  log "    stamp: $(grep -m1 '^sig:' "$RUN/TRAIN_DONE" 2>/dev/null || echo '<no signature -- pre-signature run>')"
  log "    want : sig:$SIG"
  rm -f "$RUN/TRAIN_DONE" "$RUN/RENDER_DONE" "$RUN/RUN_DONE" "$MET"
fi
if [ "$FORCE" = 1 ] || [ ! -f "$RUN/TRAIN_DONE" ]; then
  # Resume. A train cut off by the wall clock (debug chunks are 1.5 h) leaves its newest checkpoint
  # behind (every SAVE_EVERY steps); continue from it, but only when TRAIN_SIG says it was written by
  # THIS configuration. nerfstudio's --max-num-iterations counts from the loaded step
  # (trainer.py: range(start, start + n)), so pass the steps LEFT: the last step is ITERS-1 either way.
  # Only checkpoints newer than TRAIN_START (touched when this configuration started from scratch)
  # count, so a leftover run of another configuration is never resumed, picked as final or pruned.
  # --load-checkpoint, not --load-dir: the resumed run's config.yml records it, and eval_setup
  # (ns-render, lidar_dump_ns.py) would load from a recorded load_dir -- the previous, pruned segment.
  NSTEPS=$ITERS_EFF; LOAD=(); RESUMED=; LAST=
  if [ "$FORCE" != 1 ] && sig_ok "$RUN/TRAIN_SIG" && [ -f "$RUN/TRAIN_START" ]; then
    LAST=$(find "$RUN" -path '*/nerfstudio_models/step-*.ckpt' -newer "$RUN/TRAIN_START" -printf '%f %p\n' 2>/dev/null | sort | tail -1 | cut -d' ' -f2-)
    if [ -n "$LAST" ]; then
      S=$(basename "$LAST" .ckpt); S=$((10#${S#step-}))
      NSTEPS=$((ITERS_EFF - 1 - S)); LOAD=(--load-checkpoint "$LAST"); RESUMED=" resumed@$S"
      log "train RESUME from $LAST (step $S, $NSTEPS steps left)"
    fi
  fi
  # FoggyFields P3: a surgically seeded checkpoint initialises a FRESH run (env SEED_CKPT).
  if [ -z "$LAST" ] && [ -n "${SEED_CKPT:-}" ] && [ -f "${SEED_CKPT:-}" ]; then
    NSTEPS=$((ITERS_EFF - 1)); LOAD=(--load-checkpoint "$SEED_CKPT"); RESUMED=" seeded"
    log "train SEEDED from $SEED_CKPT"
  fi
  [ -n "$LAST" ] || touch "$RUN/TRAIN_START"
  { echo "train-in-progress"; echo "sig:$SIG"; } > "$RUN/TRAIN_SIG"
  T0=$(date +%s); RC=0
  if [ "$NSTEPS" -gt 0 ]; then
    run_child train ns-train "$METHOD" \
      --output-dir "$RUN" --experiment-name "${METHOD}_${VARIANT}_${CLIP}" --vis "${VIS:-tensorboard}" \
      --max-num-iterations "$NSTEPS" --machine.seed "$SEED" "${LOAD[@]}" \
      --steps-per-eval-image 999999 --steps-per-eval-batch 1000 --steps-per-eval-all-images 999999 \
      --steps-per-save "${SAVE_EVERY:-2000}" --viewer.quit-on-train-completion True \
      "${M_EXTRA[@]}" \
      nvidia-av-fog-data --data "$DATA_ROOT" --sequence "$CLIP" \
        --rectification v2 --splits-json "$SPLIT" "${CROP_FLAGS[@]}" "${DP_EXTRA[@]}" "${M_DP_EXTRA[@]}" || RC=$?
  else
    log "train: final step already checkpointed (the stamp was not written), not retraining"
  fi
  T1=$(date +%s); tim train $((T1-T0)) "$RC" "iters=$ITERS_EFF$RESUMED"
  [ $RC -eq 0 ] || { log "!! ns-train rc=$RC"; exit $RC; }
  # The run is done only when the checkpoint of the LAST step exists (a resume chain must reach it).
  FINAL_CKPT=$(find "$RUN" -path "*/nerfstudio_models/$(printf 'step-%09d.ckpt' $((ITERS_EFF - 1)))" -newer "$RUN/TRAIN_START" 2>/dev/null | head -1)
  [ -n "$FINAL_CKPT" ] || { log "!! ns-train returned 0 but no step-$((ITERS_EFF - 1)) checkpoint under $RUN"; exit 3; }
  CFG=$(dirname "$(dirname "$FINAL_CKPT")")/config.yml
  [ -f "$CFG" ] || { log "!! no config.yml next to $FINAL_CKPT"; exit 3; }
  # Earlier segments of a resume chain each kept their own newest checkpoint (1.4 GB); drop them.
  find "$RUN" -path '*/nerfstudio_models/step-*' -newer "$RUN/TRAIN_START" ! -path "$FINAL_CKPT" -delete 2>/dev/null
  { echo "$CFG"; echo "sig:$SIG"; } > "$RUN/TRAIN_DONE"; log "train OK ($(( (T1-T0)/60 )) min$RESUMED) -> $CFG"
else
  log "train SKIP (TRAIN_DONE present)"
fi
CFG=$(head -1 "$RUN/TRAIN_DONE")
[ -f "$CFG" ] || { log "!! TRAIN_DONE points at a missing config: $CFG"; exit 3; }

# ---------------------------------------------------------------- 2. render held-out frames
if [ "$FORCE" = 1 ] || [ ! -f "$RUN/RENDER_DONE" ]; then
  T0=$(date +%s); RC=0
  run_child render ns-render dataset --load-config "$CFG" --output-path "$RUN/renders" --image-format png \
    --rendered-output-names rgb gt-rgb --pose-source val || RC=$?
  T1=$(date +%s); tim render $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] || { log "!! ns-render rc=$RC"; exit $RC; }
  NCAM=$(find "$RUN/renders/val/rgb" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  NIMG=$(find "$RUN/renders/val/rgb" -name '*.png' 2>/dev/null | wc -l)
  NGT=$(find "$RUN/renders/val/gt-rgb" -name '*.png' 2>/dev/null | wc -l)
  [ "$NCAM" -eq 7 ] || { log "!! expected 7 camera dirs in renders/val/rgb, got $NCAM"; exit 4; }
  [ "$NIMG" -gt 0 ] && [ "$NIMG" -eq "$NGT" ] \
    || { log "!! render/gt mismatch: $NIMG rendered vs $NGT ground-truth PNGs"; exit 4; }
  # Count against the split, not just against itself: a truncated ns-render that still exits 0
  # would otherwise stamp RENDER_DONE and the metrics would quietly average a subset.
  # Only enforced for an untruncated run -- with --dataset-end-fraction / --max-eval-frames the
  # expected count depends on dataparser-internal arithmetic, so there we only report.
  NEVAL=$(/users/yitiwang/miniforge3/bin/python3 -I -c "
import json,sys
d=json.load(open(sys.argv[1]))
n=d.get('num_eval')
if n is None:
    v=d.get('eval_indices') or d.get('eval') or d.get('held_out') or []
    n=len(v) if isinstance(v,list) else (v or 0)
print(int(n))
" "$CLIP_ROOT/splits/${SPLIT}.json" 2>/dev/null || echo 0)
  NEXP=$(( NEVAL * 7 ))
  if [ -z "$END_FRACTION" ] && [ -z "$MAX_EVAL_FRAMES" ] && [ "$NEVAL" -gt 0 ]; then
    [ "$NIMG" -eq "$NEXP" ] || { log "!! render count $NIMG != expected $NEXP ($NEVAL held-out timesteps x 7 cameras, split $SPLIT)"; exit 4; }
    log "render count check OK: $NIMG == $NEVAL x 7"
  else
    log "render count $NIMG (expected $NEXP untruncated; END_FRACTION=${END_FRACTION:-unset} MAX_EVAL_FRAMES=${MAX_EVAL_FRAMES:-unset} -> not enforced)"
  fi
  { echo "$NIMG"; echo "sig:$SIG"; } > "$RUN/RENDER_DONE"; log "render OK ($(( (T1-T0)/60 )) min, $NIMG frames over $NCAM cameras)"
else
  log "render SKIP (RENDER_DONE present, $(head -1 "$RUN/RENDER_DONE") frames)"
fi

# ---------------------------------------------------------------- 3. common-region metrics
if [ "$FORCE" = 1 ] || [ ! -s "$MET" ]; then
  T0=$(date +%s); RC=0
  run_child metrics python "$JOBS/eval_common_region.py" --renders "$RUN/renders/val" --data "$DATA_ROOT" \
    --sequence "$CLIP" --variant "$VARIANT" --method "$METHOD" -o "$MET" || RC=$?
  T1=$(date +%s); tim metrics $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] || { log "!! eval_common_region rc=$RC"; exit $RC; }
  [ -s "$MET" ] || { log "!! metrics file not written: $MET"; exit 5; }
  # Stamp the run signature INTO the metrics file. Without it a 500-iteration smoke metric is
  # byte-indistinguishable from a 30k one, and collect_results has no way to refuse it.
  /users/yitiwang/miniforge3/bin/python3 -I -c "
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d['run_signature']=sys.argv[2]
d['iters']=int(sys.argv[3])
d['split']=sys.argv[4]
d['end_fraction']=sys.argv[5]
d['max_eval_frames']=sys.argv[6]
json.dump(d,open(p,'w'),indent=2)
" "$MET" "$SIG" "$ITERS_EFF" "${SPLIT:-nvs_50_50}" "${END_FRACTION:-1.0}" "${MAX_EVAL_FRAMES:-all}" \
    || { log "!! could not stamp run signature into $MET"; exit 5; }
  log "metrics OK -> $MET"
else
  log "metrics SKIP ($MET present)"
fi

# ---------------------------------------------------------------- 4. videos + montage
if [ "$FORCE" = 1 ] || [ ! -f "$VIZ/VIDEO_MANIFEST.json" ]; then
  T0=$(date +%s); RC=0
  run_child videos python "$JOBS/make_videos.py" --renders "$RUN/renders/val" --data "$DATA_ROOT" --sequence "$CLIP" \
    --variant "$VARIANT" --method "$METHOD" --out "$VIZ" --panel-width "$PANEL_W" --fps "$VIZ_FPS" || RC=$?
  T1=$(date +%s); tim videos $((T1-T0)) "$RC" ""
  if [ $RC -ne 0 ]; then
    log "!! make_videos rc=$RC (metrics are already written; the run is usable, the videos are not)"
    exit $RC
  fi
  NV=$(find "$VIZ" -name '*.mp4' | wc -l)
  [ "$NV" -eq 7 ] || { log "!! expected 7 mp4s in $VIZ, got $NV"; exit 6; }
  log "videos OK ($(( (T1-T0)/60 )) min, $NV mp4)"
else
  log "videos SKIP (VIDEO_MANIFEST.json present)"
fi

{ date -Is; echo "sig:$SIG"; } > "$RUN/RUN_DONE"
log "RUN DONE"
