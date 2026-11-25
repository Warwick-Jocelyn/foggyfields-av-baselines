#!/bin/bash
#SBATCH --job-name=ncf_crop_abl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --account=a0195
#SBATCH --signal=B:TERM@300
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/logs/matrix_%j.log
#
# NCF-mini crop (safe_crop) vs mask (valid_mask) ablation.
#   full matrix = CLIPS x VARIANTS x METHODS = 5 x 2 x 2 = 20 runs over the node's 4 GPUs.
#
# Everything is overridable from the environment (see jobs/env.sh), e.g. a smoke subset:
#   sbatch --export=ALL,CLIPS=NCF-02,ITERS=500,END_FRACTION=0.15 jobs/run_matrix.sbatch
#
# RESUMABLE: a run whose stages already produced their artifacts is skipped, so if the
# 12 h partition limit is hit, just submit the same script again.
set -o pipefail
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/env.sh

FAILFILE=$LOG_ROOT/FAILURES_${SLURM_JOB_ID:-local}.txt
: > "$FAILFILE"
echo "[$(date -Is)] job ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi -L || { echo "!! no GPUs visible"; exit 1; }
echo "matrix: METHODS='$METHODS' VARIANTS='$VARIANTS' CLIPS='$CLIPS'"
echo "iters : splatad=$(iters_for splatad)  neurad=$(iters_for neurad)  (ITERS override='$ITERS')"
echo "split : $SPLIT   MAX_EVAL_FRAMES='$MAX_EVAL_FRAMES'  END_FRACTION='$END_FRACTION'"

finish() {
  echo "[$(date -Is)] per-clip comparison montages"
  for C in $CLIPS; do
    python "$JOBS/make_clip_montage.py" --outputs "$OUT_ROOT" --data "$DATA_ROOT" --sequence "$C" \
      --methods "$METHODS" --variants "$VARIANTS" --out "$VIZ_ROOT/_per_clip/$C.jpg" \
      || echo "!! per-clip montage failed for $C (non-fatal)"
  done
  echo "[$(date -Is)] collecting whatever has finished"
  python "$JOBS/collect_results.py" --results "$RES_ROOT" --clips "$CLIPS" --methods "$METHODS" \
    > "$LOG_ROOT/collect_${SLURM_JOB_ID:-local}.log" 2>&1
  echo "summary -> $RES_ROOT/SUMMARY.md"
}
on_term() { echo "[$(date -Is)] !! SIGTERM (wall-clock limit). Finished runs are stamped; resubmit the same script to continue."; finish; exit 130; }
trap on_term TERM

# ====================== pre-flight: CPU, then ONE GPU, before occupying four ==========
if [[ " $METHODS " == *" unisim "* ]]; then
  python -c "import unisim" 2>/dev/null || pip install --no-deps --no-build-isolation -e "$CODE/_src/unisim" \
    || { echo "!! could not install the unisim plugin"; exit 3; }
  # Capture first: under pipefail, `ns-train --help | grep -q` fails on SIGPIPE when grep exits early.
  NS_HELP=$(ns-train --help 2>&1 || true)
  grep -qw unisim <<< "$NS_HELP" || { echo "!! ns-train does not list the unisim method"; exit 3; }
  python -c "from unisim.unisim_config import unisim" || { echo "!! unisim plugin does not import"; exit 3; }
fi
echo "---------------- pre-flight (CPU) ----------------"
python "$JOBS/preflight.py" --data "$DATA_ROOT" --clips "$CLIPS" --methods "$METHODS" \
  --variants "$VARIANTS" --split "$SPLIT" --nlanes "$NLANES" || { echo "!! pre-flight failed"; exit 3; }

FIRST_CLIP=$(echo "$CLIPS" | awk '{print $1}')
echo "---------------- pre-flight: metric self-test (CPU) ----------------"
python "$JOBS/selftest_eval.py" --data "$DATA_ROOT" --sequence "$FIRST_CLIP" \
  || { echo "!! common-region metric self-test failed -- the numbers would not be trustworthy"; exit 3; }

echo "---------------- pre-flight: 10-iteration train on ONE GPU ----------------"
PF=$OUT_ROOT/_preflight; rm -rf "$PF"; mkdir -p "$PF"
cd "$NS_REPO" || exit 3
for M in $METHODS; do
  case "$M" in
    splatad) PFX=(--pipeline.model.strategy mcmc --pipeline.model.mcmc-cap-max 5000000 --pipeline.model.max-num-seed-points 2000000); PDX=() ;;
    # Same model AND dataparser flags as lane.sh. The pre-flight used to pass none for NeuRAD, so a
    # flag that broke the CLI (add-missing-points placed before the subcommand) sailed through it.
    neurad)  PFX=($NEURAD_EXTRA); PDX=($NEURAD_DP_EXTRA) ;;
    # steps-per-stage 3 3 4: the 10 pre-flight steps cross UniSim's phase switches (patch sampler,
    # CNN decoder + VGG loss, trajectory optimisation), which otherwise start only at 2500 / 10000.
    unisim)  PFX=($UNISIM_EXTRA --pipeline.steps-per-stage 3 3 4); PDX=($UNISIM_DP_EXTRA) ;;
    *) echo "!! unknown method '$M'"; exit 3 ;;
  esac
  for V in $VARIANTS; do
    case "$V" in
      safe) CF=(--apply-safe-crop True  --use-valid-masks False) ;;
      mask) CF=(--apply-safe-crop False --use-valid-masks True)  ;;
      *) echo "!! unknown variant '$V'"; exit 3 ;;
    esac
    if [ "$M" = unisim ] && [ "$V" != safe ]; then echo "  [preflight] skip unisim/$V (safe only)"; continue; fi
    echo "  [preflight] $M / $V"
    CUDA_VISIBLE_DEVICES=0 timeout 1800 ns-train "$M" "${PFX[@]}" --output-dir "$PF/$M-$V" --vis tensorboard \
      --max-num-iterations 10 --steps-per-save 10 --machine.seed "$SEED" \
      --steps-per-eval-image 999999 --steps-per-eval-batch 999999 --steps-per-eval-all-images 999999 \
      --viewer.quit-on-train-completion True \
      nvidia-av-fog-data --data "$DATA_ROOT" --sequence "$FIRST_CLIP" --rectification v2 \
        --splits-json "$SPLIT" "${CF[@]}" "${PDX[@]}" --dataset-end-fraction "${PREFLIGHT_END_FRACTION:-0.1}" \
        --cameras camera_front_wide_120fov camera_rear_left_70fov camera_front_tele_30fov \
      || { if [ "$M" = unisim ] && [ "$(echo $METHODS | wc -w)" -gt 1 ]; then
             echo "!! pre-flight failed for unisim/$V -- DROPPING unisim from this job, the other methods continue"
             METHODS=$(echo " $METHODS " | sed 's/ unisim / /; s/^ *//; s/ *$//'); continue; fi
           echo "!! pre-flight ns-train failed for $M/$V -- not occupying 4 GPUs. Fix this first."; \
           echo "   (if it failed only because of the reduced --cameras/--dataset-end-fraction, rerun with PREFLIGHT_END_FRACTION=1.0)"; exit 4; }
  done
done
rm -rf "$PF"
echo "[$(date -Is)] pre-flight OK"

# ====================== work queue ====================================================
QUEUE=$OUT_ROOT/_queue_${SLURM_JOB_ID:-local}.txt
CUR=$QUEUE.cursor
: > "$QUEUE"; echo 1 > "$CUR"; touch "$CUR.lock"
for M in $METHODS; do for V in $VARIANTS; do for C in $CLIPS; do echo "$M $V $C" >> "$QUEUE"; done; done; done
NTASK=$(wc -l < "$QUEUE")
echo "queued $NTASK runs across $NLANES lanes"

worker() {
  local lane=$1 gpu=$2 task m v c
  sleep $((lane * 30))                     # stagger the image-cache builds
  while :; do
    task=$(flock "$CUR.lock" -c "n=\$(cat '$CUR'); sed -n \"\${n}p\" '$QUEUE'; echo \$((n+1)) > '$CUR'")
    [ -n "$task" ] || break
    read -r m v c <<< "$task"
    local run=$OUT_ROOT/$m/$v/$c
    local lg=$LOG_ROOT/lanes/${m}__${v}__${c}.log
    # A stamp only counts when it was produced by THIS configuration. lane.sh writes a
    # "sig:" line into every stamp; a cheap smoke run (ITERS=500, END_FRACTION=0.15) must not
    # make the real matrix skip that clip and publish the smoke numbers.
    local want_sig="iters=$(iters_for "$m");split=${SPLIT:-nvs_50_50};end_fraction=${END_FRACTION:-1.0};max_eval_frames=${MAX_EVAL_FRAMES:-all};variant=${v}"
    if [ "$FORCE" != 1 ] && [ -f "$run/RUN_DONE" ]; then
      if grep -qxF "sig:$want_sig" "$run/RUN_DONE" 2>/dev/null; then
        echo "[lane $lane gpu $gpu] SKIP $m/$v/$c (RUN_DONE $(head -1 "$run/RUN_DONE"), signature matches)"
        continue
      fi
      echo "[lane $lane gpu $gpu] REDO $m/$v/$c -- stamp signature differs"
      echo "    stamp: $(grep -m1 '^sig:' "$run/RUN_DONE" 2>/dev/null || echo '<none>')"
      echo "    want : sig:$want_sig"
    fi
    echo "[lane $lane gpu $gpu] START $m/$v/$c -> $lg"
    { echo; echo "######## $(date -Is) job ${SLURM_JOB_ID:-local} lane $lane gpu $gpu ########"; } >> "$lg"
    if bash "$JOBS/lane.sh" "$m" "$v" "$c" "$gpu" >> "$lg" 2>&1; then
      echo "[lane $lane gpu $gpu] DONE  $m/$v/$c"
    else
      rc=$?
      if [ $rc -eq 75 ]; then echo "[lane $lane gpu $gpu] BUSY  $m/$v/$c (another job holds it)"; continue; fi
      echo "[lane $lane gpu $gpu] FAIL  $m/$v/$c rc=$rc -- see $lg"
      echo "$(date -Is) $m $v $c rc=$rc" >> "$FAILFILE"
    fi
  done
  echo "[lane $lane gpu $gpu] queue empty"
}

for ((i = 0; i < NLANES; i++)); do
  worker "$i" "$i" &
done
wait
echo "[$(date -Is)] all lanes finished"

if [ -s "$FAILFILE" ]; then
  echo "!! some runs failed:"; cat "$FAILFILE"
fi
finish
echo "[$(date -Is)] MATRIX DONE"
