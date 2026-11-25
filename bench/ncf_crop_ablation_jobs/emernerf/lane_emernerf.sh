#!/bin/bash
# One EmerNeRF run of the harness:  lane_emernerf.sh <clip> <gpu>
#
#   1 train         train_emernerf.py (native-resolution safe_crop, nvs_50_50, LiDAR on)
#   2 render        held-out frames -> <run>/renders/val/{rgb,gt-rgb}/<cam>/frame_TTTT.png
#   3 metrics       jobs/eval_common_region.py   (the ONLY RGB metric; same script as lane.sh)
#   4 videos        jobs/make_videos.py
#   5 lidar_dump    one npz per held-out sweep (lidar_dump.py schema)
#   6 lidar_metrics jobs/emernerf/lidar_metrics_from_dump.py --eval-ray-set returns_only
#                   (cd_returns_m / cd2_returns_m2 / depth_median_l2_returns_m, lidar_dump_ns.py definition)
#   7 lidar_viz     jobs/lidar_viz.py
#
# Every stage is judged by the ARTIFACT it should produce, and stamps carry a run signature
# so a smoke run (ITERS=3000, MAX_EVAL_FRAMES=8) can never be mistaken for a formal one.
# Re-running skips finished stages; an interrupted training resumes from its newest
# checkpoint (saved every 2000 steps, see the config; intermediate ones are pruned after training).
set -o pipefail
CLIP=$1; GPU=${2:-0}
[ -n "$CLIP" ] || { echo "usage: lane_emernerf.sh <clip> <gpu>"; exit 2; }
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/emernerf/env_emernerf.sh
export CUDA_VISIBLE_DEVICES=$GPU

METHOD=emernerf; VARIANT=safe
# same layout as lane.sh / the cross-method bench: <root>/<method>/<variant>/<clip>
RUN=$OUT_ROOT/$METHOD/$VARIANT/$CLIP
RUNDIR=$RUN/train/run                 # EmerNeRF log_dir = output_root/project/run_name
VIZ=$VIZ_ROOT/$METHOD/$VARIANT/$CLIP
MET=$RES_ROOT/metrics/metrics_${METHOD}__${VARIANT}__${CLIP}.json
LMET=$RES_ROOT/lidar/lidar_${METHOD}_${VARIANT}__${CLIP}.json   # name the bench collector reads
mkdir -p "$RUN" "$VIZ"
TAG="$METHOD/$CLIP"
log(){ echo "[$(date '+%F %T')] [$TAG] $*"; }
# One process per run dir at a time (a debug chunk and a normal-partition job may both reach it).
source "$JOBS/lanelock.sh"; lane_lock "$RUN"

DATA=$(data_root_for "$CLIP") || { log "!! clip $CLIP not found under ${DATA_ROOT:-$DATA_ROOTS}"; exit 2; }
CLIP_ROOT=$("$HARNESS_PY" -I "$JOBS/clip_paths.py" --data "$DATA" --sequence "$CLIP" --print-root)
# The config CONTENT is part of the signature (not only its path): the 10-25 fixes changed
# configs/nvidia_av_fog_safecrop.yaml and default_config.yaml in place, and a run trained
# with the old content must not be taken as done.
case "$EMER_CONFIG" in /*) CFG_PATH=$EMER_CONFIG ;; *) CFG_PATH=$EMERNERF_ROOT/$EMER_CONFIG ;; esac
CFG_HASH=$(cat "$CFG_PATH" "$EMERNERF_ROOT/configs/default_config.yaml" | sha1sum | cut -c1-12) \
  || { echo "cannot hash $CFG_PATH"; exit 2; }
SIG="method=emernerf;iters=${ITERS};split=${SPLIT};max_eval_frames=${MAX_EVAL_FRAMES:-all};variant=safe;config=${EMER_CONFIG};cfg_sha1=${CFG_HASH};extra=${EMER_EXTRA}"
sig_ok(){ [ -f "$1" ] && grep -qxF "sig:$SIG" "$1" 2>/dev/null; }
sig_stale(){ [ -f "$1" ] && ! sig_ok "$1"; }
stamp(){ { echo "$2"; echo "sig:$SIG"; } > "$1"; }

tim(){  # stage seconds rc note
  [ -s "$TIMINGS" ] || { flock "$TIMINGS.lock" -c "[ -s '$TIMINGS' ] || echo 'ts,method,variant,clip,stage,seconds,rc,note' > '$TIMINGS'"; }
  flock "$TIMINGS.lock" -c "echo '$(date -Is),$METHOD,$VARIANT,$CLIP,$1,$2,$3,$4' >> '$TIMINGS'"
}
touch "$TIMINGS.lock"
NEVAL=$("$HARNESS_PY" -I -c "import json,sys;print(len(json.load(open(sys.argv[1]))['eval_indices']))" "$CLIP_ROOT/splits/${SPLIT}.json")
MEF=(); [ -n "$MAX_EVAL_FRAMES" ] && MEF=(--max-eval-frames "$MAX_EVAL_FRAMES")
want_stage(){ [ "$FORCE" = 1 ] || [ ! -f "$1" ]; }

# Heavy stages run as a background child so a SIGTERM (wall-clock limit, forwarded by
# run_emernerf.sbatch) is handled at once: the child is stopped and the PARTIAL segment is
# appended to timings.csv with rc=143, so a resumed run's segments add up to the true time.
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
    tim "$CUR_STAGE" $(( $(date +%s) - ST0 )) 143 "KILLED partial segment (SIGTERM)${RESUME:+ resumed}; rerun resumes"
  fi
  exit 143
}
trap on_term TERM

log "start gpu=$GPU iters=$ITERS split=$SPLIT data=$DATA clip_root=$CLIP_ROOT max_eval_frames=${MAX_EVAL_FRAMES:-all}"

# ---------------------------------------------------------------- 1. train
if sig_stale "$RUN/TRAIN_DONE" || { [ ! -f "$RUN/TRAIN_DONE" ] && [ -f "$RUN/TRAIN_SIG" ] && ! sig_ok "$RUN/TRAIN_SIG"; }; then
  OLD=$RUN.stale_$(date +%Y%m%d_%H%M%S)
  log "train REDO: existing run was made with a different configuration -> moved to $OLD"
  mv "$RUN" "$OLD"; mkdir -p "$RUN"
fi
FINAL=$RUNDIR/$(printf 'checkpoint_%05d.pth' "$ITERS")
if [ "$FORCE" = 1 ] || [ ! -f "$RUN/TRAIN_DONE" ]; then
  stamp "$RUN/TRAIN_SIG" "train-in-progress"
  RESUME=()
  LAST=$(ls "$RUNDIR"/checkpoint_*.pth 2>/dev/null | sort | tail -1)
  if [ "$FORCE" != 1 ] && [ -n "$LAST" ] && [ "$LAST" != "$FINAL" ]; then
    RESUME=("resume_from=$LAST"); log "train RESUME from $LAST"
  fi
  if [ "$FORCE" != 1 ] && [ -f "$FINAL" ]; then
    log "train: final checkpoint already present, not retraining"; RC=0; T0=$(date +%s); T1=$T0
  else
    T0=$(date +%s); RC=0
    run_child train bash -c 'cd "$1" && shift && exec "$@"' _ "$EMERNERF_ROOT" \
        "$EMER_PY" train_emernerf.py --config_file "$EMER_CONFIG" \
        --output_root "$RUN" --project train --run_name run \
        data.data_root="$DATA" data.scene_idx="$CLIP" data.split_json="$SPLIT" \
        optim.num_iters="$ITERS" "${RESUME[@]}" $EMER_EXTRA || RC=$?
    T1=$(date +%s); tim train $((T1-T0)) "$RC" "iters=$ITERS${RESUME:+ resumed}"
  fi
  [ $RC -eq 0 ] || { log "!! train rc=$RC"; exit $RC; }
  [ -f "$FINAL" ] || { log "!! training returned 0 but $FINAL is missing"; exit 3; }
  # Intermediate checkpoints (695 MB each) only served the resume chain.
  find "$RUNDIR" -maxdepth 1 -name 'checkpoint_*' ! -name "$(basename "$FINAL")" -delete
  stamp "$RUN/TRAIN_DONE" "$FINAL"; log "train OK ($(( (T1-T0)/60 )) min) -> $FINAL"
else
  log "train SKIP (TRAIN_DONE present)"
fi
FINAL=$(head -1 "$RUN/TRAIN_DONE")
[ -f "$FINAL" ] || { log "!! TRAIN_DONE points at a missing checkpoint: $FINAL"; exit 3; }
if [ "$(stat -c %Y "$FINAL")" -gt "$(stat -c %Y "$RUN/RENDER_DONE" 2>/dev/null || echo 0)" ] && [ -f "$RUN/RENDER_DONE" ]; then
  log "checkpoint is newer than the renders -> downstream stages redone"
  rm -f "$RUN/RENDER_DONE" "$RUN/LIDAR_DUMP_DONE" "$RUN/RUN_DONE" "$MET" "$LMET" "$VIZ/VIDEO_MANIFEST.json" "$VIZ/lidar/lidar_viz_stats.json"
fi
sig_stale "$RUN/RENDER_DONE" && rm -f "$RUN/RENDER_DONE" "$MET" "$VIZ/VIDEO_MANIFEST.json"
sig_stale "$RUN/LIDAR_DUMP_DONE" && rm -f "$RUN/LIDAR_DUMP_DONE" "$LMET" "$VIZ/lidar/lidar_viz_stats.json"

# ---------------------------------------------------------------- 2. render held-out frames
if want_stage "$RUN/RENDER_DONE"; then
  rm -rf "$RUN/renders"; rm -f "$MET" "$VIZ/VIDEO_MANIFEST.json"      # derived from the renders -> redo
  T0=$(date +%s); RC=0
  run_child render "$EMER_PY" "$EJOBS/render_heldout.py" --run-dir "$RUNDIR" --ckpt "$FINAL" --out "$RUN/renders/val" \
    --chunk "$RENDER_CHUNK" "${MEF[@]}" || RC=$?
  T1=$(date +%s); tim render $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] || { log "!! render rc=$RC"; exit $RC; }
  NCAM=$(find "$RUN/renders/val/rgb" -mindepth 1 -maxdepth 1 -type d | wc -l)
  NIMG=$(find "$RUN/renders/val/rgb" -name '*.png' | wc -l)
  NGT=$(find "$RUN/renders/val/gt-rgb" -name '*.png' | wc -l)
  [ "$NCAM" -eq 7 ] || { log "!! expected 7 camera dirs in renders/val/rgb, got $NCAM"; exit 4; }
  [ "$NIMG" -gt 0 ] && [ "$NIMG" -eq "$NGT" ] || { log "!! render/gt mismatch: $NIMG vs $NGT"; exit 4; }
  if [ -z "$MAX_EVAL_FRAMES" ]; then
    [ "$NIMG" -eq $((NEVAL * 7)) ] || { log "!! render count $NIMG != $NEVAL held-out x 7"; exit 4; }
    log "render count check OK: $NIMG == $NEVAL x 7"
  else
    log "render count $NIMG (truncated: MAX_EVAL_FRAMES=$MAX_EVAL_FRAMES of $NEVAL held-out timesteps)"
  fi
  stamp "$RUN/RENDER_DONE" "$NIMG"; log "render OK ($(( (T1-T0)/60 )) min, $NIMG frames)"
else
  log "render SKIP ($(head -1 "$RUN/RENDER_DONE") frames)"
fi

# ---------------------------------------------------------------- 3. common-region metrics
if [ "$FORCE" = 1 ] || [ ! -s "$MET" ]; then
  T0=$(date +%s); RC=0
  run_child metrics "$HARNESS_PY" "$JOBS/eval_common_region.py" --renders "$RUN/renders/val" --data "$DATA" \
    --sequence "$CLIP" --variant "$VARIANT" --method "$METHOD" -o "$MET" || RC=$?
  T1=$(date +%s); tim metrics $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] && [ -s "$MET" ] || { log "!! eval_common_region rc=$RC / no $MET"; rm -f "$MET"; exit 5; }
  "$HARNESS_PY" -I -c "
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d['run_signature']=sys.argv[2]; d['iters']=int(sys.argv[3]); d['split']=sys.argv[4]
d['end_fraction']='1.0'; d['max_eval_frames']=sys.argv[5]
m=json.load(open(sys.argv[6])); d['render_manifest']={k:m[k] for k in ('checkpoint','step','split_json','heldout_timesteps_split','rendered_timesteps','truncated','intrinsics_used','seconds_render','mrays_per_s')}
json.dump(d,open(p,'w'),indent=2)
" "$MET" "$SIG" "$ITERS" "$SPLIT" "${MAX_EVAL_FRAMES:-all}" "$RUN/renders/render_manifest.json" \
    || { log "!! could not stamp run signature into $MET -> unsigned file deleted"; rm -f "$MET"; exit 5; }
  log "metrics OK -> $MET"
else
  log "metrics SKIP ($MET present)"
fi

# ---------------------------------------------------------------- 4. videos
if [ "$FORCE" = 1 ] || [ ! -f "$VIZ/VIDEO_MANIFEST.json" ]; then
  T0=$(date +%s); RC=0
  run_child videos "$HARNESS_PY" "$JOBS/make_videos.py" --renders "$RUN/renders/val" --data "$DATA" --sequence "$CLIP" \
    --variant "$VARIANT" --method "$METHOD" --out "$VIZ" --panel-width "$PANEL_W" --fps "$VIZ_FPS" || RC=$?
  T1=$(date +%s); tim videos $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] || { log "!! make_videos rc=$RC"; exit $RC; }
  NV=$(find "$VIZ" -maxdepth 1 -name '*.mp4' | wc -l)
  [ "$NV" -eq 7 ] || { log "!! expected 7 mp4s in $VIZ, got $NV"; exit 6; }
  log "videos OK ($NV mp4)"
else
  log "videos SKIP"
fi

# ---------------------------------------------------------------- 5. LiDAR dump
if want_stage "$RUN/LIDAR_DUMP_DONE"; then
  rm -rf "$RUN/lidar_dump"
  T0=$(date +%s); RC=0
  run_child lidar_dump "$EMER_PY" "$EJOBS/lidar_dump_emernerf.py" --run-dir "$RUNDIR" --ckpt "$FINAL" --out "$RUN/lidar_dump" \
    --chunk "$LIDAR_CHUNK" "${MEF[@]}" || RC=$?
  T1=$(date +%s); tim lidar_dump $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] || { log "!! lidar dump rc=$RC"; exit $RC; }
  ND=$(find "$RUN/lidar_dump" -name 'frame_*.npz' | wc -l)
  [ "$ND" -gt 0 ] || { log "!! no npz written"; exit 7; }
  if [ -z "$MAX_EVAL_FRAMES" ]; then
    [ "$ND" -eq "$NEVAL" ] || { log "!! $ND npz != $NEVAL held-out sweeps"; exit 7; }
  fi
  stamp "$RUN/LIDAR_DUMP_DONE" "$ND"; log "lidar dump OK ($ND sweeps)"
  rm -f "$LMET" "$VIZ/lidar/lidar_viz_stats.json"     # derived from the dump -> redo
else
  log "lidar dump SKIP"
fi

# ---------------------------------------------------------------- 6. LiDAR metrics
if [ "$FORCE" = 1 ] || [ ! -s "$LMET" ]; then
  T0=$(date +%s); RC=0
  run_child lidar_metrics "$HARNESS_PY" "$EJOBS/lidar_metrics_from_dump.py" --dump "$RUN/lidar_dump" -o "$LMET" \
    --method "$METHOD" --clip "$CLIP" --eval-ray-set returns_only > "$LOG_ROOT/lanes/lidar_metrics__${CLIP}.log" 2>&1 || RC=$?
  T1=$(date +%s); tim lidar_metrics $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] && [ -s "$LMET" ] || { log "!! lidar metrics rc=$RC"; rm -f "$LMET"; exit 8; }
  "$HARNESS_PY" -I -c "
import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['run_signature']=sys.argv[2]; json.dump(d,open(p,'w'),indent=2)" "$LMET" "$SIG" \
    || { log "!! could not stamp run signature into $LMET -> unsigned file deleted"; rm -f "$LMET"; exit 8; }
  log "lidar metrics OK -> $LMET"
else
  log "lidar metrics SKIP"
fi

# ---------------------------------------------------------------- 7. LiDAR figures
if [ "$FORCE" = 1 ] || [ ! -s "$VIZ/lidar/lidar_viz_stats.json" ]; then
  T0=$(date +%s); RC=0
  run_child lidar_viz env MPLBACKEND=Agg "$HARNESS_PY" "$JOBS/lidar_viz.py" --dump "$RUN/lidar_dump" -o "$VIZ/lidar" || RC=$?
  T1=$(date +%s); tim lidar_viz $((T1-T0)) "$RC" ""
  [ $RC -eq 0 ] || { log "!! lidar_viz rc=$RC"; exit 9; }
  log "lidar viz OK -> $VIZ/lidar"
else
  log "lidar viz SKIP"
fi

stamp "$RUN/RUN_DONE" "$(date -Is)"
log "RUN DONE"
