#!/bin/bash
# Keep scoring V1.3 runs until $DEADLINE (epoch s):  v13_poll.sh <gpu>
# Candidates: $V13_QUEUE lines "<label> <clip> <config.yml>" and any config.yml / step-*.ckpt path in $LANES_MD
# (label = dir after 'train/', clip = NFF-XX in the path). One poller per run via a claim dir.
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/final/config_final.sh
g=${1:-0}; : "${DEADLINE:?}"
OV=/capstor/scratch/cscs/yitiwang/foggyfields-v1/v13_overnight
log(){ echo "[$(date '+%F %T')] [v13poll gpu $g] $*"; }
candidates(){
  [ -f "$V13_QUEUE" ] && grep -vE '^\s*(#|$)' "$V13_QUEUE"
  # P's final runs: $OV/20261123_<CLIP>_v13_final/train/V13-<lane>/ALL_DONE -> V13-<lane>/V13-<lane>/neurad-fog-v13-5k/run/config.yml
  for m in "$OV"/20261123_NFF-*_v13_final/train/*/ALL_DONE; do
    [ -f "$m" ] || continue
    lane=$(basename "$(dirname "$m")"); clip=$(echo "$m" | grep -oE 'NFF-[0-9]{2}' | head -1)
    cfg=$(ls "$(dirname "$m")/$lane"/*/run/config.yml 2>/dev/null | head -1)
    [ -n "$cfg" ] && [ -f "$cfg" ] && echo "$lane $clip $cfg"
  done
  # plain NeuRAD 5k controls (A0) of the same runs: complete when their last checkpoint exists
  for cfg in "$OV"/20261123_NFF-*_v13_final/train/A0/A0/neurad/run/config.yml; do
    [ -f "$cfg" ] && [ -f "$(dirname "$cfg")/nerfstudio_models/step-000004999.ckpt" ] && echo "A0 $(echo "$cfg" | grep -oE 'NFF-[0-9]{2}' | head -1) $cfg"
  done
  [ -f "$LANES_MD" ] && grep -oE '/[^ |)`"'"'"'<>]+(config\.yml|step-[0-9]+\.ckpt)' "$LANES_MD" | sort -u | while read -r p; do
    case "$p" in *.ckpt) cfg=$(dirname "$(dirname "$p")")/config.yml ;; *) cfg=$p ;; esac
    [ -f "$cfg" ] || continue
    [ -f "$(dirname "$cfg")/nerfstudio_models/step-000004999.ckpt" ] || ls "$(dirname "$cfg")"/nerfstudio_models/step-*.ckpt >/dev/null 2>&1 || continue
    label=$(echo "$cfg" | sed -nE 's#.*/train/([^/]+)/.*#\1#p'); [ -n "$label" ] || label=$(basename "$(dirname "$(dirname "$(dirname "$cfg")")")")
    clip=$(echo "$cfg" | grep -oE 'NFF-[0-9]{2}' | head -1); [ -n "$clip" ] || continue
    echo "$label $clip $cfg"
  done
}
CHILD=; trap 'log "SIGTERM"; [ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null; wait $CHILD 2>/dev/null; exit 143' TERM
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  did=0
  while read -r label clip cfg; do
    [ -n "$cfg" ] || continue
    [ -s "$BW/results_final/v13/lidar/lidar_v13__${label}__${clip}.json" ] && [ -s "$BW/results_final/v13/metrics/metrics_v13__${label}__${clip}.json" ] && continue
    claim=$BW/outputs_final/v13/.claim_${label}__${clip}; mkdir -p "$BW/outputs_final/v13"
    if ! mkdir "$claim" 2>/dev/null; then
      age=$(( $(date +%s) - $(stat -c %Y "$claim") )); [ "$age" -gt 7200 ] && { rm -rf "$claim"; mkdir "$claim" 2>/dev/null || continue; } || continue
    fi
    [ $(( DEADLINE - $(date +%s) )) -lt 2400 ] && { rmdir "$claim"; log "less than 40 min left: not starting $label/$clip"; break; }
    log "scoring $label $clip $cfg"
    mkdir -p "$BW/logs_final/v13"
    bash "$BW/jobs/final/v13_lane.sh" "$label" "$clip" "$cfg" "$g" >> "$BW/logs_final/v13/${label}__${clip}.log" 2>&1 & CHILD=$!; wait $CHILD; rc=$?; CHILD=
    rm -rf "$claim"; log "$label/$clip rc=$rc"; did=1
    [ $rc -eq 143 ] && exit 143
  done < <(candidates | sort -u)
  [ $did -eq 1 ] || sleep 300
done
log "deadline reached"
