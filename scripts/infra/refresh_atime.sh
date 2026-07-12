#!/bin/bash
# refresh_atime.sh — OPTIONAL safety net. Resets last-access time on files you must
# keep on scratch mid-experiment so the 14/30-day purge doesn't eat them.
#
# ⚠️ This is a stopgap, NOT the strategy. The real protection is sync_to_store.sh +
# GitHub/HF. Use this only for large transient artifacts (datasets, live run dirs)
# that are painful to re-fetch but not yet worth a durable copy.
#
#   bash scripts/infra/refresh_atime.sh
#
# Weekly cron (Sundays 03:00) — well inside the 14-day iopsstor window:
#   0 3 * * 0 bash /iopsstor/scratch/cscs/$USER/projects/foggyfields/code/scripts/infra/refresh_atime.sh >> $HOME/atime.log 2>&1
set -eo pipefail

# Dirs to keep alive on scratch. Edit to match what you're actively using.
KEEP=(
  "/iopsstor/scratch/cscs/$USER/projects/foggyfields/code"
  "/iopsstor/scratch/cscs/$USER/projects/foggyfields/Dataset"
  "/iopsstor/scratch/cscs/$USER/projects/foggyfields/neurad-studio"
  "/iopsstor/scratch/cscs/$USER/projects/foggyfields/splatad"
  "/iopsstor/scratch/cscs/$USER/projects/foggyfields/EmerNeRF"
)

for d in "${KEEP[@]}"; do
  [ -e "$d" ] || { echo "skip (missing): $d"; continue; }
  n=$(find "$d" -type f -exec touch -a {} + -print 2>/dev/null | wc -l)
  echo "refreshed atime on $n files under $d"
done
echo "done $(date)"
