#!/bin/bash
# Push the final-ablation harness (+ results when MODE=final) to github.com/Warwick-Jocelyn/foggyfields-av-baselines (PUBLIC repo).
#   publish_github.sh code    -> bench/ snapshot only
#   publish_github.sh final   -> + results/final_ablation_2026-11/ (FINAL_TABLE, CONVERGENCE, RESULTS_BASELINES, metric JSONs)
#                                + one visual example in assets/ (GT-vs-methods montage, one 360-orbit contact sheet)
# Explicit pathspecs only; images are force-added (repo .gitignore excludes *.jpg on purpose); no mp4/ckpt/npz ever.
set -u
MODE=${1:-code}
BW=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
REPO=/users/yitiwang/projects/foggyfields-av-baselines
DST=/users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics
V3D=/users/yitiwang/FoggyfieldV1/RGB_branch_3D_viz/NFF-02
PY=/users/yitiwang/miniforge3/bin/python3
log(){ echo "[$(date '+%F %T')] [publish_github $MODE] $*"; }
cd "$REPO" || exit 1
rsync -a --delete --exclude '__pycache__' --exclude 'core*' --exclude '*.bak*' --exclude '*.log' --exclude '.plan_*' \
      --exclude '.viz3d_enabled' --exclude '*.new' "$BW/jobs/" bench/ncf_crop_ablation_jobs/
PATHS=(bench)
if [ "$MODE" = final ]; then
  R=results/final_ablation_2026-11; mkdir -p "$R/metrics"
  cp "$DST/FINAL_TABLE.md" "$R/README.md"
  cp "$DST/baselines/CONVERGENCE.md" "$DST/baselines/convergence.json" "$DST/baselines/RESULTS_BASELINES.md" "$R/" 2>/dev/null
  for tag in allfix opreg5e4 std v13; do for k in metrics lidar; do
    for f in "$BW"/results_final/$tag/$k/*.json; do [ -f "$f" ] && cp "$f" "$R/metrics/${tag}__$(basename "$f")"; done; done; done
  # visual example 1: GT vs every method, held-out frame 99, NFF-02 front wide (one JPEG, ~150 KB)
  [ -f "$DST/figures/NFF-02__frame_0099__front_wide_120fov.jpg" ] && \
    cp "$DST/figures/NFF-02__frame_0099__front_wide_120fov.jpg" assets/final_ablation_NFF-02_gt_vs_methods_frame99.jpg
  # visual example 2: one 360-degree orbit contact sheet (SplatAD, the strongest RGB row), downsized
  if [ -f "$V3D/SplatAD/contact_sheet_front_wide_120fov.jpg" ]; then
    "$PY" -I -c "from PIL import Image; im=Image.open('$V3D/SplatAD/contact_sheet_front_wide_120fov.jpg'); im.thumbnail((1600,1600)); im.save('assets/final_ablation_NFF-02_splatad_orbit360.jpg', quality=82)"
  fi
  cat >> "$R/README.md" <<'EOF'

## Visual example
![GT vs methods, NFF-02 held-out frame 99](../../assets/final_ablation_NFF-02_gt_vs_methods_frame99.jpg)

![SplatAD 360-degree orbit (RGB | expected depth), NFF-02](../../assets/final_ablation_NFF-02_splatad_orbit360.jpg)

Full 360-degree orbit videos of every method (mp4, not in git): `/users/yitiwang/FoggyfieldV1/RGB_branch_3D_viz/` on Clariden
(renderer `bench/ncf_crop_ablation_jobs/final/viz3d_orbit.py`). Harness: `bench/ncf_crop_ablation_jobs/final/` (see `bench/README.md`).
EOF
  PATHS+=("$R")
  for a in assets/final_ablation_NFF-02_gt_vs_methods_frame99.jpg assets/final_ablation_NFF-02_splatad_orbit360.jpg; do
    [ -f "$a" ] && { git add -f -- "$a"; PATHS+=("$a"); }
  done
fi
git add -- "${PATHS[@]}"
if git diff --cached --quiet; then log "nothing to commit"; else
  MSG="final ablation: $( [ "$MODE" = final ] && echo 'SOTA baseline results (NFF-02/08/10, one common evaluator, convergence-checked) + visual example' || echo 'bench harness snapshot (supervisor, 3D orbit viz, convergence check, EmerNeRF env self-repair)')"
  git commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PH1H2K9QFoGyai6uv3GGNm" -- "${PATHS[@]}" || { log "commit failed"; exit 1; }
  log "committed $(git log --oneline -1)"
fi
for try in 1 2 3; do
  timeout 120 git push origin main 2>&1 | tail -2 && [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main 2>/dev/null)" ] && { log "pushed $(git rev-parse --short HEAD)"; exit 0; }
  timeout 120 git pull --rebase -q origin main 2>&1 | tail -2; sleep 30
done
log "!! push failed"; exit 1
