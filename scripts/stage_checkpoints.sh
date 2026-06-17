#!/bin/bash
# Gather the best-PSNR checkpoint per (method x clip) + its sibling config into an
# HF upload tree. Existence-checked; prints what it copies. Does NOT upload.
#   bash scripts/stage_checkpoints.sh [HF_STAGE_DIR]
set -eo pipefail
OUT="${OUT:-/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output}"
PANDA="${PANDA:-/networkhome/WMGDS/wang3_y/ETH/Code/outputs}"
STAGE="${1:-$HOME/hf_ckpt_stage}"
mkdir -p "$STAGE"

copy() { # src dst
  if [ -f "$1" ]; then mkdir -p "$(dirname "$2")"; cp "$1" "$2"; echo "  OK  $(du -h "$1"|cut -f1)  $2"
  else echo "  !! MISSING: $1"; fi
}
# nerfstudio: copy ckpt + the config.yml that sits one level above nerfstudio_models/
copy_ns() { # ckpt_path dst_dir
  local ck="$1" dst="$2"
  copy "$ck" "$dst/$(basename "$ck")"
  copy "$(dirname "$(dirname "$ck")")/config.yml" "$dst/config.yml"
}

echo "== NVIDIA SplatAD (run 2026-05-29_230151) =="
for S in 002 003 004; do
  copy_ns "$OUT/splatad/$S/unnamed/splatad/2026-05-29_230151/nerfstudio_models/step-000029999.ckpt" "$STAGE/nvidia/splatad/$S"
done
echo "== NVIDIA NeuRAD =="
for S in 002 003 004; do
  CK=$(find "$OUT/neurad/$S" -name step-000029999.ckpt 2>/dev/null | sort | tail -1)
  copy_ns "$CK" "$STAGE/nvidia/neurad/$S"
done
echo "== NVIDIA EmerNeRF (PRE-split-fix, stale) =="
for S in 002 003 004; do
  D="$OUT/emernerf/$S/foggyfields_emernerf/nvidia_fog_${S}_7cam"
  copy "$D/checkpoint_25000.pth" "$STAGE/nvidia/emernerf/$S/checkpoint_25000.pth"
  copy "$D/config.yaml" "$STAGE/nvidia/emernerf/$S/config.yaml"
done

echo "== PandaSet SplatAD =="
copy_ns "$(find "$PANDA"/paper_011_split05_paperfaithful -name step-000030000.ckpt|tail -1)" "$STAGE/pandaset/splatad/011"
copy_ns "$(find "$PANDA"/paper_078_split05_paperfaithful -name step-000030000.ckpt|tail -1)" "$STAGE/pandaset/splatad/078"
echo "== PandaSet NeuRAD =="
for S in 011 078; do
  copy_ns "$(find "$PANDA"/neurad_paper_${S}_split05 -name 'step-0000299*.ckpt'|tail -1)" "$STAGE/pandaset/neurad/$S"
done
echo "== PandaSet EmerNeRF (rgb variant) =="
for S in 011 078; do
  D=$(find "$PANDA"/emernerf_${S}_split05 -type d -name 'scene_*_rgb'|head -1)
  copy "$D/checkpoint_30000.pth" "$STAGE/pandaset/emernerf/$S/checkpoint_30000.pth"
  copy "$D/config.yaml" "$STAGE/pandaset/emernerf/$S/config.yaml"
done

echo; echo "Staged at $STAGE  (total: $(du -sh "$STAGE"|cut -f1))"
echo "Next: bash scripts/upload_hf.sh $STAGE <hf-user>/nvidia-av-fog-baseline-checkpoints"
