#!/bin/bash
# Login-node feeder (nohup): keeps ONE of my 1.5 h debug chunks in the queue at a time, only when the debug queue
# holds < 2 of my jobs (alternating with the orchestrator's driver). Plans in order; a plan advances when its
# .plan_done marker exists. Logs each submission to the registry and PLAN.md. Stops at DEADLINE or when all plans are done.
BW=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
REG=/users/yitiwang/projects/foggyfields-v1/runs/registry_allocations.txt
PL=/users/yitiwang/projects/foggyfields-v1/reports/v13_overnight/final_ablation_2026-11-23/PLAN.md
PLANS=(D2 D3); DEADLINE=$(( $(date +%s) + 30*3600 )); MIN_GAP=900; last=0
log(){ echo "[$(date '+%F %T')] [debug_feeder] $*"; }
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  cur=""; for p in "${PLANS[@]}"; do [ -f "$BW/outputs_final/.plan_done_plan_$p" ] || { cur=$p; break; }; done
  [ -n "$cur" ] || { log "all plans done"; exit 0; }
  mine=$(squeue -u yitiwang -h -p debug -o %j | grep -c '^finalbase_D')
  total=$(squeue -u yitiwang -h -p debug | wc -l)
  if [ "$mine" -eq 0 ] && [ "$total" -lt 2 ] && [ $(( $(date +%s) - last )) -ge $MIN_GAP ]; then
    jid=$(sbatch --parsable -A a0195 -p debug -t 01:30:00 --job-name=finalbase_$cur --export=ALL,PLAN=$BW/jobs/final/plan_$cur.txt "$BW/jobs/final/run_final.sbatch" 2>&1)
    case "$jid" in [0-9]*) last=$(date +%s)
      log "submitted plan_$cur -> $jid (debug total was $total)"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) job $jid a0195/debug 1 node <= 1.5 h: plan_$cur chunk (debug_feeder, lanes resume) [final-ablation]" >> "$REG"
      echo "- $(date '+%H:%M') B(feeder): debug chunk $jid = plan_$cur (debug queue had $total job(s))." >> "$PL" ;;
      *) log "sbatch failed: $jid"; last=$(date +%s) ;;
    esac
  fi
  sleep 120
done
log "deadline"
