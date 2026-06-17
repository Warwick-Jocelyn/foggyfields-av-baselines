#!/bin/bash
# Upload a staged checkpoint tree to a HuggingFace model repo.
# Prereq: `hf auth login` (the stored token on the origin box is invalid).
#   bash scripts/upload_hf.sh <stage_dir> <user>/nvidia-av-fog-baseline-checkpoints
set -eo pipefail
STAGE="${1:?usage: upload_hf.sh <stage_dir> <user>/<repo>}"
REPO="${2:?usage: upload_hf.sh <stage_dir> <user>/<repo>}"

command -v hf >/dev/null || { echo "install: pip install -U huggingface_hub"; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "run: hf auth login"; exit 1; }

hf repo create "$REPO" --repo-type model -y 2>/dev/null || true
echo "Uploading $(du -sh "$STAGE"|cut -f1) from $STAGE -> $REPO ..."
# single folder upload preserves the nvidia/ pandaset/ layout; lfs handled automatically
hf upload "$REPO" "$STAGE" . --repo-type model
echo "Done: https://huggingface.co/$REPO"

# --- dataset (run separately, AFTER confirming NVIDIA license + gating) ---
# DATA=/networkhome/WMGDS/wang3_y/ETH/Dataset/NVIDIA_AV_Fog/processed
# hf repo create <user>/nvidia-av-fog-processed --repo-type dataset -y   # then set 'gated' in repo settings
# for C in 002_* 003_* 004_*; do
#   hf upload <user>/nvidia-av-fog-processed "$DATA/$C/processed_pinhole" "processed/$C/processed_pinhole" --repo-type dataset
# done
