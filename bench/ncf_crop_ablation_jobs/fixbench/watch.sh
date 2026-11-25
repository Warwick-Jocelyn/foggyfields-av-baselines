#!/bin/bash
# Watch the fixbench jobs: release requeue-holds, stop on the first failed lane or when all jobs are gone.
W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
JOBS="$*"
while :; do
  live=$(squeue -h -j "${JOBS// /,}" -o "%i %T %r" 2>/dev/null)
  echo "$live" | awk '$2=="PENDING" && $3=="JobHeldUser"{print $1}' | while read -r j; do
    scontrol release "$j" && echo "[$(date '+%F %T')] released requeue-held $j"; done
  fails=$(grep -h "rc=[1-9]" $W/logs_fix/run_fix_*.log 2>/dev/null | grep -v "rc=143" | sort -u)
  [ -n "$fails" ] && { echo "[$(date '+%F %T')] LANE FAILURE:"; echo "$fails"; exit 1; }
  [ -z "$live" ] && { echo "[$(date '+%F %T')] all jobs finished"; grep -h "rc=" $W/logs_fix/run_fix_*.log | sort -u; exit 0; }
  sleep 300
done
