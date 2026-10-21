#!/usr/bin/env bash
# Regenerate patches/*.patch and refresh src/ from two upstream working trees.
#
# Usage:
#   scripts/regen_patches.sh <neurad-studio-worktree> <emernerf-worktree>
#
# Both worktrees must be checked out at the commits in BASE_COMMITS.txt with our
# changes applied as uncommitted edits / untracked files. The script writes:
#   patches/neurad-studio.patch   git diff (tracked) + untracked new files
#   patches/emernerf.patch
#   src/**                        plain-file copies of every new/modified file
set -euo pipefail
NS="${1:?neurad-studio worktree}"
EN="${2:?EmerNeRF worktree}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gen() {  # <worktree> <outfile>
  local wt="$1" out="$2"
  ( cd "$wt"
    # stage untracked files so `git diff --cached` includes them as new-file hunks
    git add -AN -- . >/dev/null
    git diff --binary HEAD -- . ) > "$out"
}

gen "$NS" "$REPO/patches/neurad-studio.patch"
gen "$EN" "$REPO/patches/emernerf.patch"

# --- plain-file fallback copies ------------------------------------------------
mkdir -p "$REPO/src/neurad-studio/nerfstudio/data/dataparsers" \
         "$REPO/src/neurad-studio/nerfstudio/data/datasets" \
         "$REPO/src/neurad-studio/nerfstudio/cameras" \
         "$REPO/src/emernerf/datasets" "$REPO/src/emernerf/configs"
cp "$NS/nerfstudio/data/dataparsers/nvidia_av_fog_dataparser.py" \
   "$REPO/src/neurad-studio/nerfstudio/data/dataparsers/"
for f in datasets/nvidia_av_fog.py datasets/_nvidia_av_fog_smoketest.py datasets/pandaset.py \
         configs/nvidia_av_fog.yaml configs/nvidia_av_fog_lidar.yaml \
         configs/pandaset.yaml configs/pandaset_lidar.yaml; do
  [ -f "$EN/$f" ] && cp "$EN/$f" "$REPO/src/emernerf/$f"
done
echo "regenerated patches + src/ from:"
echo "  neurad-studio: $NS"
echo "  EmerNeRF:      $EN"
