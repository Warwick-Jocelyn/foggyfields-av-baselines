#!/bin/bash
# sync_to_store.sh — mirror the irreplaceable bytes from scratch to the backed-up
# /capstor/store tier, and push code to GitHub. Run after any run worth keeping,
# or as a daily cron. No GPU used.
#
#   bash scripts/infra/sync_to_store.sh
#
# Cron example (daily 02:00, from a login node that can reach store):
#   0 2 * * * cd /iopsstor/scratch/cscs/$USER/projects/foggyfields/code && \
#             bash scripts/infra/sync_to_store.sh >> $HOME/sync_to_store.log 2>&1
set -eo pipefail

WORK="${WORK:-/iopsstor/scratch/cscs/$USER/projects/foggyfields}"
OUT="${OUT:-$WORK/outputs}"                 # where training runs / checkpoints land
STORE="/capstor/store/cscs/swissai/a0195/$USER/foggyfields"
mkdir -p "$STORE"

echo "== sync $(date) =="

# 1. code repo working tree (incl. .git) -> store mirror --------------------
echo "[1/4] code -> $STORE/code_repo"
rsync -a --delete "$WORK/code/" "$STORE/code_repo/"

# 2. results (docs, metrics, videos) ----------------------------------------
if [ -d "$OUT" ] || [ -d "$WORK/code/results" ]; then
  echo "[2/4] results -> $STORE/results"
  [ -d "$WORK/code/results" ] && rsync -a "$WORK/code/results/" "$STORE/results/"
fi

# 3. FINAL checkpoints (the gap that caused the loss) -----------------------
# Grab the last checkpoint per run + its sibling config. Adjust patterns to taste.
echo "[3/4] final checkpoints -> $STORE/checkpoints_final"
mkdir -p "$STORE/checkpoints_final"
if [ -d "$OUT" ]; then
  # nerfstudio (.ckpt) + EmerNeRF (.pth), keep newest per parent dir
  find "$OUT" \( -name '*.ckpt' -o -name 'checkpoint_*.pth' \) 2>/dev/null | while read -r ck; do
    rel="${ck#$OUT/}"; dst="$STORE/checkpoints_final/$rel"
    mkdir -p "$(dirname "$dst")"; rsync -a "$ck" "$dst"
    cfg="$(dirname "$(dirname "$ck")")/config.yml"; [ -f "$cfg" ] && rsync -a "$cfg" "$(dirname "$dst")/"
    cfg2="$(dirname "$ck")/config.yaml";           [ -f "$cfg2" ] && rsync -a "$cfg2" "$(dirname "$dst")/"
  done
  echo "  staged: $(find "$STORE/checkpoints_final" -type f | wc -l) files, $(du -sh "$STORE/checkpoints_final" 2>/dev/null | cut -f1)"
else
  echo "  (no $OUT — nothing to stage)"
fi

# 4. git commit + push (best effort — off-cluster truth) --------------------
echo "[4/4] git push"
if git -C "$WORK/code" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$WORK/code" add -A
  git -C "$WORK/code" commit -m "sync: $(hostname) $(date -u +%Y-%m-%dT%H:%MZ)" 2>/dev/null \
    && git -C "$WORK/code" push 2>&1 | tail -3 \
    || echo "  (nothing to commit, or push needs auth: gh auth login / set a PAT)"
fi
echo "== done =="
