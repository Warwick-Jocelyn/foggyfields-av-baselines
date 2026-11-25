#!/bin/bash
#SBATCH --job-name=emernerf_safe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --account=a0195
#SBATCH --signal=B:TERM@300
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs_emernerf/emernerf_%j.log
#
# EmerNeRF lane of the ncf-crop-ablation benchmark, SAFE_CROP protocol, split nvs_50_50.
# Per clip (jobs/emernerf/lane_emernerf.sh): train -> render held-out -> eval_common_region
# -> make_videos -> LiDAR npz dump -> LiDAR metrics -> lidar_viz. One clip per GPU.
#
# Formal runs (4 NCF clips + NFF-05, EmerNeRF's default 25000 iterations), into the shared
# cross-method bench tree next to SplatAD / NeuRAD / UniSim:
#   sbatch --export=ALL,DATA_ROOT=$W/data_all,OUT_ROOT=$W/outputs_bench,RES_ROOT=$W/results_bench,VIZ_ROOT=$W/viz_bench,LOG_ROOT=$W/logs_emernerf \
#     jobs/emernerf/run_emernerf.sbatch
# -> $OUT_ROOT/emernerf/safe/<clip>/, $RES_ROOT/metrics/metrics_emernerf__safe__<clip>.json,
#    $RES_ROOT/lidar/lidar_emernerf_safe__<clip>.json (eval_ray_set=returns_only), $VIZ_ROOT/emernerf/safe/<clip>/
# Without overrides it writes to $W/{outputs,results,viz}_emernerf with the same <method>/<variant>/<clip> layout.
# RESUMABLE: finished stages are skipped and an interrupted training resumes from its last
# checkpoint (every 5000 steps), so on a wall-clock kill just submit the same script again.
#
# Overrides (environment; see env_emernerf.sh): CLIPS, ITERS, SPLIT, DATA_ROOT/DATA_ROOTS,
# OUT_ROOT, RES_ROOT, VIZ_ROOT, LOG_ROOT, MAX_EVAL_FRAMES, NLANES, FORCE, EMER_EXTRA, ...
# Smoke example (debug partition, 1 GPU):
#   sbatch --partition=debug --time=01:20:00 --gres=gpu:1 \
#     --output=$W/logs_emernerf_smoke/emernerf_%j.log \
#     --export=ALL,CLIPS=NCF-02,ITERS=3000,MAX_EVAL_FRAMES=8,OUT_ROOT=$W/outputs_emernerf_smoke,RES_ROOT=$W/results_emernerf_smoke,VIZ_ROOT=$W/viz_emernerf_smoke,LOG_ROOT=$W/logs_emernerf_smoke \
#     jobs/emernerf/run_emernerf.sbatch
set -o pipefail
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/emernerf/env_emernerf.sh
JID=${SLURM_JOB_ID:-local}
FAILFILE=$LOG_ROOT/FAILURES_${JID}.txt; : > "$FAILFILE"
echo "[$(date -Is)] job $JID on $(hostname)"
nvidia-smi -L || { echo "!! no GPUs visible"; exit 1; }
# lane i runs on GPUS[i]: the GPUs Slurm exposed (CUDA_VISIBLE_DEVICES) if set, else all listed
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); else GPUS=($(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )))); fi
NGPU=${#GPUS[@]}
NLANES=${NLANES:-$NGPU}; [ "$NLANES" -gt "$NGPU" ] && NLANES=$NGPU
echo "clips='$CLIPS' iters=$ITERS split=$SPLIT max_eval_frames='${MAX_EVAL_FRAMES:-all}' lanes=$NLANES"
echo "out=$OUT_ROOT res=$RES_ROOT viz=$VIZ_ROOT"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 60 \
  > "$LOG_ROOT/gpumem_${JID}.csv" 2>/dev/null &
SMI=$!
LANEPIDS=$LOG_ROOT/_lanepids_${JID}.txt; STOP=$LOG_ROOT/_stop_${JID}; : > "$LANEPIDS"; rm -f "$STOP"
on_term(){
  echo "[$(date -Is)] !! SIGTERM (wall clock): stopping lanes; finished stages are stamped, resubmit to continue."
  touch "$STOP"                                      # workers take no new clip
  for p in $(cat "$LANEPIDS"); do kill -TERM "$p" 2>/dev/null; done   # lanes log the partial segment (rc=143)
  for _ in $(seq 1 100); do                          # up to ~200 s of the 300 s signal margin
    alive=0; for p in $(cat "$LANEPIDS"); do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 2
  done
  kill $SMI 2>/dev/null; exit 130
}
trap on_term TERM

# ---------------- pre-flight (CPU, seconds) ---------------------------------------------
[ -x "$EMER_PY" ] || { echo "!! $EMER_PY missing -- build it with jobs/emernerf/build_env.sh"; exit 3; }
"$EMER_PY" - <<'EOF' || { echo "!! EmerNeRF venv import check failed"; exit 3; }
import os, sys
sys.path.insert(0, os.environ["EMERNERF_ROOT"]); os.chdir(os.environ["EMERNERF_ROOT"])
import numpy, torch, nerfacc, omegaconf
from nerfacc import csrc  # AOT sm_90 build, not JIT
import third_party.tcnn_modules as t
assert int(numpy.__version__.split(".")[0]) < 2, numpy.__version__
print(f"  torch {torch.__version__} cuda {torch.version.cuda} | numpy {numpy.__version__} | nerfacc {nerfacc.__version__} "
      f"| tcnn binding {t._C.__name__} | cc {torch.cuda.get_device_capability(0)}")
EOF
for C in $CLIPS; do
  R=$(data_root_for "$C") || { echo "!! clip $C not found under ${DATA_ROOT:-$DATA_ROOTS}"; exit 3; }
  CR=$("$HARNESS_PY" -I "$JOBS/clip_paths.py" --data "$R" --sequence "$C" --print-root)
  "$HARNESS_PY" -I - "$CR" "$SPLIT" <<'EOF' || exit 3
import json, sys
from pathlib import Path
cr, split = Path(sys.argv[1]), sys.argv[2]
s = json.loads((cr / "splits" / f"{split}.json").read_text())
tr, ev = set(s["train_indices"]), set(s["eval_indices"])
assert not (tr & ev), "train/eval overlap"
prov = json.loads((cr / "provenance.json").read_text()).get("lidar_frame")
assert prov in ("ego", "sensor"), f"provenance lidar_frame={prov!r}"
k2 = json.loads((cr / "calib" / "K_rectified_v2.json").read_text())
assert all("safe_crop" in v for v in k2.values()) and len(k2) == 7
print(f"  {cr.name:22s} split {split}: {len(tr)} train / {len(ev)} held-out | lidar_frame={prov} | 7 cams with safe_crop")
EOF
done

# ---------------- work queue -------------------------------------------------------------
QUEUE=$LOG_ROOT/_queue_${JID}.txt; CUR=$QUEUE.cursor
: > "$QUEUE"; echo 1 > "$CUR"; touch "$CUR.lock"
for C in $CLIPS; do echo "$C" >> "$QUEUE"; done
worker() {
  local lane=$1 gpu=$2 c lg
  sleep $((lane * 20))                      # stagger the image loads (host RAM / Lustre)
  while :; do
    [ -f "$STOP" ] && break
    c=$(flock "$CUR.lock" -c "n=\$(cat '$CUR'); sed -n \"\${n}p\" '$QUEUE'; echo \$((n+1)) > '$CUR'")
    [ -n "$c" ] || break
    lg=$LOG_ROOT/lanes/emernerf__${c}.log
    echo "[lane $lane gpu $gpu] START $c -> $lg"
    { echo; echo "######## $(date -Is) job $JID lane $lane gpu $gpu ########"; } >> "$lg"
    bash "$EJOBS/lane_emernerf.sh" "$c" "$gpu" >> "$lg" 2>&1 & local lp=$!
    echo "$lp" >> "$LANEPIDS"
    if wait "$lp"; then
      echo "[lane $lane gpu $gpu] DONE  $c"
    else
      rc=$?; [ $rc -eq 75 ] && { echo "[lane $lane gpu $gpu] BUSY  $c (another job holds it)"; continue; }
      echo "[lane $lane gpu $gpu] FAIL  $c rc=$rc -- see $lg"; echo "$(date -Is) $c rc=$rc" >> "$FAILFILE"
    fi
  done
}
PIDS=()
for ((i = 0; i < NLANES; i++)); do worker "$i" "${GPUS[$i]}" & PIDS+=($!); done
wait "${PIDS[@]}"
rm -f "$LANEPIDS"
kill $SMI 2>/dev/null
echo "[$(date -Is)] all lanes finished"
[ -s "$FAILFILE" ] && { echo "!! failures:"; cat "$FAILFILE"; }

# ---------------- summary ----------------------------------------------------------------
"$HARNESS_PY" -I - "$RES_ROOT" $CLIPS <<'EOF'
import json, sys
from pathlib import Path
res = Path(sys.argv[1]); rows = []
for c in sys.argv[2:]:
    m = res / "metrics" / f"metrics_emernerf__safe__{c}.json"; l = res / "lidar" / f"lidar_emernerf_safe__{c}.json"
    r = {"clip": c}
    if m.is_file():
        d = json.loads(m.read_text()); r.update({k: d["primary"][k] for k in ("psnr", "ssim", "lpips")}, n=d["n_images"], iters=d.get("iters"))
    if l.is_file():
        d = json.loads(l.read_text()); r.update(eval_ray_set=d.get("eval_ray_set"), cd_returns=d["cd_returns_m"], cd2_returns=d["cd2_returns_m2"],
                 depth_med_returns=d["depth_median_l2_returns_m"])
    rows.append(r)
print(json.dumps(rows, indent=1))
(res / "summary_emernerf.json").write_text(json.dumps(rows, indent=2))
EOF
if [ -s "$FAILFILE" ]; then echo "[$(date -Is)] EMERNERF FINISHED WITH FAILURES (see $FAILFILE)"; exit 1; fi
echo "[$(date -Is)] EMERNERF DONE"
