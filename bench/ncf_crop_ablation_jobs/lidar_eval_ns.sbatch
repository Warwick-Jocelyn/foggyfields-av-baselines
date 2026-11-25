#!/bin/bash
#SBATCH --job-name=abl_lidar_ns
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:25:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs/lidar_ns_%j.log
# Native-LiDAR evaluation for any neurad-studio method (splatad | neurad | unisim) with
# jobs/lidar_dump_ns.py: one render pass per held-out sweep -> metrics JSON, plus the per-sweep npz
# dump for DUMP_CLIPS only. Eval ray set = returns + range-image non-returns (forced by lidar_dump_ns.py).
#
# One lane per GPU; tasks (method x variant x clip) are pulled from a shared queue, so with 4 GPUs and
# 4 clips each GPU gets one clip. Defaults above = debug, 1 GPU. Formal bench run on normal, 4 GPUs:
#
#   W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
#   sbatch --partition=normal --gres=gpu:4 --time=03:00:00 \
#     --export=ALL,METHODS="neurad",CLIPS="NCF-02 NCF-07 NCF-10 NCF-15 NFF-05",OUT_ROOT=$W/outputs_bench,DATA_ROOT=$W/data_all,RES_DIR=$W/results_bench/lidar,LOG_ROOT=$W/logs_bench \
#     $W/jobs/lidar_eval_ns.sbatch
#
# Knobs (all env-overridable):
#   METHODS     space-separated, e.g. "splatad neurad unisim"   (falls back to METHOD, default neurad)
#   VARIANTS    default "safe"
#   CLIPS       default "NCF-02 NCF-07 NCF-10 NCF-15"
#   OUT_ROOT    where <method>/<variant>/<clip>/.../config.yml live   (env.sh default $W/outputs)
#   DATA_ROOT   if set, passed as --set data=$DATA_ROOT (else the path stored in config.yml is used)
#   RES_DIR     metrics dir, default $RES_ROOT/lidar  -> lidar_<method>_<variant>__<clip>.json
#   LOG_ROOT    per-task logs go to $LOG_ROOT/lanes/lidar_ns__<method>__<variant>__<clip>.log
#   DUMP_CLIPS  clips that also get an npz dump (default none; 0.6-1.4 GB per clip)
#   DUMP_ROOT   default $OUT_ROOT/_lidar_dump  -> <method>_<variant>__<clip>/frame_XXXX.npz
#   N_CAMERAS   camera frames to time (default 70; 0 skips camera timing)
#   NLANES_LIDAR parallel lanes (default = GPUs allocated: SLURM_GPUS_ON_NODE, else 1)
#   LIDAR_ARGS  appended verbatim to lidar_dump_ns.py (e.g. "--crosscheck 1")
#   FORCE       1 = redo tasks whose JSON already exists
set -o pipefail
# capture the caller's values BEFORE env.sh fills in its own matrix defaults
DATA_ROOT_OVERRIDE=${DATA_ROOT:-}
_VARIANTS=${VARIANTS:-safe}
_CLIPS=${CLIPS:-"NCF-02 NCF-07 NCF-10 NCF-15"}
_METHODS=${METHODS:-${METHOD:-neurad}}
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/env.sh
METHODS=$_METHODS
VARIANTS=$_VARIANTS
CLIPS=$_CLIPS
RES_DIR=${RES_DIR:-$RES_ROOT/lidar}
DUMP_CLIPS=${DUMP_CLIPS:-}
DUMP_ROOT=${DUMP_ROOT:-$OUT_ROOT/_lidar_dump}
N_CAMERAS=${N_CAMERAS:-70}
LIDAR_ARGS=${LIDAR_ARGS:-}
FORCE=${FORCE:-0}
NLANES=${NLANES_LIDAR:-${SLURM_GPUS_ON_NODE:-1}}
mkdir -p "$RES_DIR" "$LOG_ROOT/lanes"
echo "[$(date -Is)] node=$(hostname) METHODS=$METHODS VARIANTS=$VARIANTS CLIPS=$CLIPS NLANES=$NLANES"
echo "  OUT_ROOT=$OUT_ROOT DATA_ROOT_OVERRIDE=${DATA_ROOT_OVERRIDE:-<from config.yml>} RES_DIR=$RES_DIR DUMP_CLIPS=$DUMP_CLIPS"
nvidia-smi -L
cd "$(cd -P "$CODE/neurad-studio" && pwd)"

queue=$(mktemp -p "$RES_DIR" .lidar_queue.XXXX); cur=$queue.cursor; fails=$queue.fails
for m in $METHODS; do for v in $VARIANTS; do for c in $CLIPS; do echo "$m $v $c"; done; done; done > "$queue"
echo 1 > "$cur"; : > "$fails"; touch "$cur.lock"

lane () {
  local gpu=$1
  export CUDA_VISIBLE_DEVICES=$gpu
  while :; do
    local task; task=$(flock "$cur.lock" -c "n=\$(cat '$cur'); sed -n \"\${n}p\" '$queue'; echo \$((n+1)) > '$cur'")
    [ -n "$task" ] || break
    local m v c; read -r m v c <<< "$task"
    local run=$OUT_ROOT/$m/$v/$c
    local cfg; cfg=$(find "$run/" -name config.yml 2>/dev/null | sort | tail -1)
    local res=$RES_DIR/lidar_${m}_${v}__${c}.json
    local lg=$LOG_ROOT/lanes/lidar_ns__${m}__${v}__${c}.log
    if [ -z "$cfg" ]; then echo "[gpu $gpu] SKIP $m/$v/$c (no config.yml under $run)"; continue; fi
    [ -f "$run/TRAIN_DONE" ] || echo "[gpu $gpu] WARN $m/$v/$c has no TRAIN_DONE stamp"
    if [ -s "$res" ] && [ "$FORCE" != 1 ]; then echo "[gpu $gpu] SKIP $m/$v/$c (done: $res)"; continue; fi
    local args=(--config "$cfg" --output "$res" --n-cameras "$N_CAMERAS")
    [ -n "$DATA_ROOT_OVERRIDE" ] && args+=(--set "data=$DATA_ROOT_OVERRIDE")
    for dc in $DUMP_CLIPS; do [ "$dc" = "$c" ] && args+=(--dump "$DUMP_ROOT/${m}_${v}__${c}"); done
    echo "[$(date -Is)] [gpu $gpu] START $m/$v/$c  cfg=$cfg"
    local t0; t0=$(date +%s)
    # shellcheck disable=SC2086
    if python "$JOBS/lidar_dump_ns.py" "${args[@]}" $LIDAR_ARGS > "$lg" 2>&1; then
      echo "[$(date -Is)] [gpu $gpu] DONE  $m/$v/$c ($(( $(date +%s) - t0 )) s) $(python -c "import json;d=json.load(open('$res'));print(d.get('eval_ray_set'), {k:round(d[k],4) for k in ('cd_m','cd2_m2','rcd','depth_median_l2_m','ray_drop_acc','cd_returns_m') if d.get(k) is not None})")"
    else
      echo "[$(date -Is)] [gpu $gpu] FAIL  $m/$v/$c -- see $lg"; tail -15 "$lg"
      flock "$cur.lock" -c "echo '$m $v $c' >> '$fails'"
    fi
  done
}
for ((g = 0; g < NLANES; g++)); do lane "$g" & sleep 5; done
wait
nfail=$(wc -l < "$fails")
[ "$nfail" -gt 0 ] && { echo "FAILED tasks:"; cat "$fails"; }
rm -f "$queue" "$cur" "$cur.lock" "$fails"
echo "[$(date -Is)] LIDAR NS EVAL DONE ($nfail failed) -> $RES_DIR"
ls -la "$RES_DIR"
exit "$nfail"
