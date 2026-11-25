#!/bin/bash
#SBATCH --job-name=abl_lidar
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs/lidar_%j.log
# Native-LiDAR evaluation of the already-trained crop/mask runs: Chamfer distance and
# friends, computed from SplatAD's own lidar rasteriser on the held-out sweeps.
# The camera variant is the ONLY difference between the two arms, so any CD gap answers
# "does the extra camera FOV help the point cloud?".
set -o pipefail
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/env.sh
OUT=${LIDAR_OUT:-$W/results/lidar}
DUMP=${LIDAR_DUMP:-$W/outputs/_lidar_dump}
mkdir -p "$OUT" "$DUMP" "$LOG_ROOT/lanes"
CLIPS=${CLIPS:-"NCF-02 NCF-07 NCF-10 NCF-15"}
VARIANTS=${VARIANTS:-"safe mask"}
DUMP_CLIPS=${DUMP_CLIPS:-"NCF-15"}        # point-cloud dumps are big: only these
echo "[$(date -Is)] node=$(hostname)"; nvidia-smi -L

queue=$(mktemp); for v in $VARIANTS; do for c in $CLIPS; do echo "$v $c"; done; done > "$queue"
cur=$(mktemp); echo 1 > "$cur"; touch "$cur.lock"

lane () {
  local gpu=$1
  export CUDA_VISIBLE_DEVICES=$gpu
  while :; do
    task=$(flock "$cur.lock" -c "n=\$(cat '$cur'); sed -n \"\${n}p\" '$queue'; echo \$((n+1)) > '$cur'")
    [ -n "$task" ] || break
    read -r v c <<< "$task"
    local run=$OUT_ROOT/splatad/$v/$c
    local cfg=$(find "$run" -name config.yml 2>/dev/null | head -1)
    local res=$OUT/lidar_${v}__${c}.json
    local lg=$LOG_ROOT/lanes/lidar__${v}__${c}.log
    if [ -z "$cfg" ]; then echo "[gpu $gpu] SKIP $v/$c (no config)"; continue; fi
    if [ -s "$res" ]; then echo "[gpu $gpu] SKIP $v/$c (done)"; continue; fi
    echo "[gpu $gpu] START lidar $v/$c"
    cd $CODE/neurad-studio
    if python "$JOBS/lidar_chamfer_eval.py" --config "$cfg" --output "$res" > "$lg" 2>&1; then
      echo "[gpu $gpu] DONE  $v/$c  $(python -c "import json;d=json.load(open('$res'));print({k:round(v,4) for k,v in d.items() if isinstance(v,(int,float))})" 2>/dev/null | cut -c1-150)"
    else
      echo "[gpu $gpu] FAIL  $v/$c -- see $lg"; tail -5 "$lg"
    fi
    # point-cloud dump for the clips we want to visualise
    for dc in $DUMP_CLIPS; do
      if [ "$c" = "$dc" ] && [ ! -d "$DUMP/${v}__${c}" ]; then
        echo "[gpu $gpu] dump point clouds $v/$c"
        python "$JOBS/lidar_dump.py" --config "$cfg" --out "$DUMP/${v}__${c}" >> "$lg" 2>&1 \
          || echo "[gpu $gpu] dump FAILED $v/$c"
      fi
    done
  done
}
for g in 0 1 2 3; do lane $g & sleep 5; done
wait
rm -f "$queue" "$cur" "$cur.lock"
echo "[$(date -Is)] LIDAR EVAL DONE -> $OUT"
ls -la "$OUT"
