#!/bin/bash
# Score one FoggyField V1.3 run with the bench evaluators:  v13_lane.sh <label> <clip> <config.yml> <gpu>
# 1 render+dump (GPU, v13_score.py; v13-uni worktree first, v13-mv if the config does not load there)
# 2 eval_common_region.py (variant safe)   3 lidar_metrics_from_dump.py returns_only   4 lidar_viz.py
set -o pipefail
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/final/config_final.sh
label=$1 clip=$2 cfg=$3 g=${4:-0}
[ -f "$cfg" ] || { echo "!! config $cfg missing"; exit 2; }
RUN=$BW/outputs_final/v13/${label}__${clip}; RES=$BW/results_final/v13; VIZ=$BW/viz_final/v13/${label}__${clip}
MET=$RES/metrics/metrics_v13__${label}__${clip}.json; LMET=$RES/lidar/lidar_v13__${label}__${clip}.json
mkdir -p "$RUN" "$RES/metrics" "$RES/lidar" "$VIZ"
log(){ echo "[$(date '+%F %T')] [v13 $label/$clip gpu $g] $*"; }
source "$BW/jobs/lanelock.sh"; lane_lock "$RUN"
TIM=$RES/timings_v13.csv; touch "$TIM.lock"
tim(){ flock "$TIM.lock" -c "[ -s '$TIM' ] || echo 'ts,label,clip,stage,seconds,rc' > '$TIM'; echo '$(date -Is),$label,$clip,$1,$2,$3' >> '$TIM'"; }
CHILD=; trap 'log "!! SIGTERM"; [ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; exit 143' TERM
run(){ "$@" & CHILD=$!; wait $CHILD; local rc=$?; CHILD=; return $rc; }
log "start cfg=$cfg"
PLAIN=0; grep -q "^method_name: neurad-fog-v13" "$cfg" || PLAIN=1     # 1 = plain nerfstudio model (NeuRAD A0): ns-render + lidar_dump_ns.py
if [ "$PLAIN" = 1 ] && [ ! -f "$RUN/SCORE_MANIFEST.json" ]; then
  T0=$(date +%s)
  run bash -c "source '$V13UNI/jobs/v13uni_env.sh' && cd '$V13UNI' && CUDA_VISIBLE_DEVICES=$g exec ns-render dataset --load-config '$cfg' --output-path '$RUN/renders' --image-format png --rendered-output-names rgb gt-rgb --pose-source val"; RC=$?
  [ $RC -eq 143 ] && exit 143
  tim render $(( $(date +%s) - T0 )) $RC; [ $RC -eq 0 ] || { log "!! ns-render rc=$RC"; exit 3; }
  T0=$(date +%s); rm -rf "$RUN/lidar_dump.tmp"
  run bash -c "source '$V13UNI/jobs/v13uni_env.sh' && cd '$V13UNI' && CUDA_VISIBLE_DEVICES=$g exec python '$BW/jobs/lidar_dump_ns.py' --config '$cfg' --output '$LMET.tmp' --dump '$RUN/lidar_dump.tmp' --n-cameras 0"; RC=$?
  [ $RC -eq 143 ] && exit 143
  tim lidar_dump_ns $(( $(date +%s) - T0 )) $RC; [ $RC -eq 0 ] && [ -s "$LMET.tmp" ] || { log "!! lidar_dump_ns rc=$RC"; exit 3; }
  rm -rf "$RUN/lidar_dump"; mv "$RUN/lidar_dump.tmp" "$RUN/lidar_dump"; mv "$LMET.tmp" "$LMET"
  step=$(ls "$(dirname "$cfg")"/nerfstudio_models/step-*.ckpt | tail -1 | sed -E 's/.*step-0*([0-9]+)\.ckpt/\1/')
  echo "{\"config\": \"$cfg\", \"step\": $step, \"path\": \"plain nerfstudio: ns-render dataset + jobs/lidar_dump_ns.py (ray adapter, drop-aware + returns_only metrics)\"}" > "$RUN/SCORE_MANIFEST.json"
  rm -f "$MET"
fi
if [ "$PLAIN" = 0 ] && [ ! -f "$RUN/SCORE_MANIFEST.json" ]; then
  T0=$(date +%s); RC=1
  for wt in "$V13UNI" "$V13MV"; do
    envsh=$wt/jobs/v13uni_env.sh; [ "$wt" = "$V13MV" ] && envsh=$wt/jobs/v13mv_env.sh
    [ -f "$envsh" ] || { log "no env script $envsh"; continue; }
    log "render with worktree $wt ($envsh)"
    run bash -c "source '$envsh' && cd '$wt' && CUDA_VISIBLE_DEVICES=$g exec python '$BW/jobs/final/v13_score.py' --config '$cfg' --out '$RUN'"; RC=$?
    [ $RC -eq 0 ] && break
    log "render rc=$RC with $wt"
    [ $RC -eq 143 ] && exit 143
  done
  tim render $(( $(date +%s) - T0 )) $RC
  [ $RC -eq 0 ] && [ -f "$RUN/SCORE_MANIFEST.json" ] || { log "!! render failed"; exit 3; }
  rm -f "$MET" "$LMET"
fi
NPNG=$(find "$RUN/renders/val/rgb" -name '*.png' | wc -l); NNPZ=$(find "$RUN/lidar_dump" -name 'frame_*.npz' | wc -l)
log "renders: $NPNG png, $NNPZ sweeps"
source "$BW/jobs/env.sh" >/dev/null 2>&1   # venv_neurad (VENV/NS_REPO exported by config_final.sh)
if [ ! -s "$MET" ]; then
  T0=$(date +%s); run "$HARNESS_PY" "$BW/jobs/eval_common_region.py" --renders "$RUN/renders/val" --data "$DATA_ROOT" \
     --sequence "$clip" --variant safe --method "v13-$label" -o "$MET"; RC=$?; tim metrics $(( $(date +%s) - T0 )) $RC
  [ $RC -eq 0 ] && [ -s "$MET" ] || { log "!! eval_common_region rc=$RC"; rm -f "$MET"; exit 4; }
  "$HARNESS_PY" -I -c "
import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['iters']=int(json.load(open(sys.argv[2]))['step'])+1; d['split']='nvs_50_50'
d['v13_config']=sys.argv[3]; d['note']='V1.3 renders are the safe_crop images of the dataparser (apply_safe_crop=True), 1 px trimmed; scored as variant safe'
json.dump(d,open(p,'w'),indent=2)" "$MET" "$RUN/SCORE_MANIFEST.json" "$cfg"
  log "metrics OK -> $MET"
fi
if [ ! -s "$LMET" ]; then
  T0=$(date +%s); run "$HARNESS_PY" "$BW/jobs/emernerf/lidar_metrics_from_dump.py" --dump "$RUN/lidar_dump" -o "$LMET" \
     --method "v13-$label" --clip "$clip" --eval-ray-set returns_only > "$BW/logs_final/v13/lidarmetrics_${label}__${clip}.log" 2>&1; RC=$?
  tim lidar_metrics $(( $(date +%s) - T0 )) $RC
  [ $RC -eq 0 ] && [ -s "$LMET" ] || { log "!! lidar metrics rc=$RC"; rm -f "$LMET"; exit 5; }
  log "lidar metrics OK -> $LMET"
fi
if [ ! -s "$VIZ/lidar/lidar_viz_stats.json" ]; then
  MPLBACKEND=Agg run "$HARNESS_PY" "$BW/jobs/lidar_viz.py" --dump "$RUN/lidar_dump" -o "$VIZ/lidar" > "$BW/logs_final/v13/lidarviz_${label}__${clip}.log" 2>&1 || log "lidar_viz failed (metrics unaffected)"
fi
log "DONE"
