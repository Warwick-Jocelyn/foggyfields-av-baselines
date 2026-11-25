#!/bin/bash
# Bench copy of neurad-studio-v13-final-run2/jobs/v13final.sbatch (2026-11-25) for the 30k fair-budget lanes, run INSIDE the
# final bench node job (node_final.sh kind v13train): RUN_SUFFIX (separate root), GPU0, SAVE_EVERY, SKIP_POST, one token, rc.
#!/bin/bash
#SBATCH --job-name=v13final
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --partition=debug
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/foggyfields-v1/logs/v13final_%j.log
# Decision 109 final ablation: up to 4 lane tokens CLIP:LANE (one per GPU), 5k stage schedule (METHOD neurad-fog-v13-5k, B_END 2500,
# C_END 3500), seed 42, nvs_50_50; one run dir per clip (20261123_<CLIP>_v13_final, lanes V13-<lane>); resumable (checkpoint every
# 100 steps, each lane restarts from its latest checkpoint); in-job eval + renders + counterfactuals + defog outputs per lane.
# Token CLIP:A0 = plain NeuRAD 5k baseline (ns-train neurad in the V1.2 checkout, scored by the same V1.3 suite via adapter_v12).
# Code = the FROZEN snapshot $V13 (jobs/v13final_env.sh), never the editable worktree.
#   TOKENS="NFF-02:UcnnR NFF-08:U0 NFF-08:UcnnR NFF-10:A0" sbatch -A <acct> jobs/v13final.sbatch
set -o pipefail
ulimit -c 0
source ${V13:-/capstor/scratch/cscs/yitiwang/splatad-nff-repro/code/neurad-studio-v13-final-run2}/jobs/v13final_env.sh || exit 9
cd $V13
TOKENS=${TOKENS:?"set TOKENS='CLIP:LANE ...'"}
STEPS=${STEPS:-5000}
METHOD=${METHOD:-neurad-fog-v13-5k}
B_END=${B_END:-2500}
C_END=${C_END:-3500}
OV=/capstor/scratch/cscs/yitiwang/foggyfields-v1/v13_overnight
COS=/capstor/scratch/cscs/yitiwang/foggyfields-v1/matprior/out/cosmos_diffusion_renderer
V12NS=/capstor/scratch/cscs/yitiwang/splatad-nff-repro/code/neurad-studio
source $V13/jobs/v13final_lanes.sh
# fail fast (2026-11-25): the 30k method must have the scaled stages and EVERY scheduler at the budget before any GPU time is spent
if [ "$METHOD" = neurad-fog-v13-30k ]; then
  python - <<'PYCHK' || { echo "!! $METHOD config precheck FAILED"; exit 3; }
from nerfstudio.configs.method_configs import method_configs as M
c = M["neurad-fog-v13-30k"]; f = c.pipeline.model.fog_v13
assert (f.stage_a_end, f.stage_b_end, f.stage_c_end) == (6000, 15000, 21000), (f.stage_a_end, f.stage_b_end, f.stage_c_end)
bad = {k: o["scheduler"].max_steps for k, o in c.optimizers.items() if o.get("scheduler") is not None and o["scheduler"].max_steps != 30000}
assert not bad and c.max_num_iterations == 30000, bad
print("[precheck] neurad-fog-v13-30k OK: stages 6000/15000/21000, %d schedulers at 30000" % len(c.optimizers))
PYCHK
fi
grep -q "hard_overlap_full_m" $V13/nerfstudio/models/neurad_fog_v13.py && export LIDAR_FLAGS_PRESENT=1 || export LIDAR_FLAGS_PRESENT=0
echo "[$(date)] node=$(hostname) V13=$V13 git=$(git rev-parse HEAD) dirty=$(git status --porcelain | grep -v core_ | wc -l) tokens=$TOKENS steps=$STEPS method=$METHOD lidar_flags=$LIDAR_FLAGS_PRESENT"
for tok in $TOKENS; do   # validator M3: abort on an unknown token before anything trains
  clip=${tok%%:*}; L=${tok##*:}
  clip_setup $clip || { echo "!! unknown clip in '$tok'"; exit 3; }
  [ "$L" = A0 ] && continue
  case "$(lane_args $L)" in *UNKNOWN_LANE*) echo "!! unknown lane '$L' in '$tok': abort"; exit 3;; esac
done
g=${GPU0:-0}
for tok in $TOKENS; do
  ( export CUDA_VISIBLE_DEVICES=$g
    clip=${tok%%:*}; L=${tok##*:}
    clip_setup $clip
    RUN_ID=20261123_${clip}_v13_final${RUN_SUFFIX:-}
    ROOT=$OV/$RUN_ID
    ANCH=/users/yitiwang/projects/foggyfields-v1/outputs/v12/anchors/${clip}_anchors.npz
    HELD=$ROOT/heldout_ids.json
    [ -f $ANCH ] || { echo "!! $tok: no anchors $ANCH"; exit 2; }
    [ -f $HELD ] || { mkdir -p $ROOT; cp $OV/20261123_${clip}_v13_final/heldout_ids.json $HELD 2>/dev/null; }   # same held-out ids as the 5k lanes
    [ -f $HELD ] || { echo "!! $tok: no heldout_ids.json in $ROOT"; exit 2; }
    mkdir -p $ROOT/train $ROOT/logs $ROOT/config
    # run-dir owner lock per lane dir (user rule: one live job per lane directory)
    out=$ROOT/train/V13-$L; [ "$L" = A0 ] && out=$ROOT/train/A0
    mkdir -p $out
    if ! mkdir $out/.owner_lock 2>/dev/null; then
      own=$(cat $out/.owner_lock/jobid 2>/dev/null)
      # only a RUNNING owner blocks: a preempted/requeued owner (PENDING) or a dead one hands the lane over (resumable lanes)
      if [ -n "$own" ] && [ "$own" != "$SLURM_JOB_ID" ] && squeue -j $own -h -t RUNNING 2>/dev/null | grep -q .; then
        echo "!! $tok: lane dir $out owned by live job $own: skipping"; exit 0; fi
    fi
    echo $SLURM_JOB_ID > $out/.owner_lock/jobid
    git rev-parse HEAD > $out/commit_sha.txt; git diff > $out/uncommitted.diff
    echo "[$(date)] job $SLURM_JOB_ID gpu $g token $tok code $(git rev-parse --short HEAD) prior_state=$(prior_state)" >> $ROOT/logs/lanes.txt
    TL=$ROOT/logs/timings.txt
    DP=(nvidia-av-fog-data --data /users/yitiwang/FoggyfieldV1/data --sequence $clip --train-split-fraction 0.5 --use-valid-masks True)
    if [ "$L" = A0 ]; then
      # ---- plain NeuRAD 5k baseline: exactly the V1.2 smoke recipe (jobs/v12_smoke_clip.sbatch lane A0), V1.2 checkout + venv ----
      if [ ! -f $out/TRAIN_DONE ]; then
        T0=$(date +%s)
        ( export PYTHONPATH=$V12NS:${PYTHONPATH#$V13:}; cd $V12NS
          python -c "import nerfstudio,sys; sys.exit(0 if nerfstudio.__file__.startswith('$V12NS/') else 9)" || { echo "!! A0: wrong nerfstudio for the V1.2 checkout"; exit 9; }
          echo "[A0] nerfstudio = $(python -c 'import nerfstudio; print(nerfstudio.__file__)') git=$(git rev-parse --short HEAD)" >> $out/train.log
          ns-train neurad --output-dir $out --experiment-name A0 --timestamp run --max-num-iterations $STEPS --vis tensorboard --machine.seed 42 \
            --steps-per-eval-image 999999 --steps-per-eval-all-images 999999 --steps-per-save $STEPS --viewer.quit-on-train-completion True \
            --logging.steps-per-log 50 "${DP[@]}" >> $out/train.log 2>&1 ); rc=$?
        echo "A0 train rc=$rc sec=$(( $(date +%s) - T0 ))" >> $TL
        ck=$(ls $out/A0/neurad/run/nerfstudio_models/step-$(printf %09d $((STEPS-1))).ckpt 2>/dev/null | head -1)
        [ $rc -eq 0 ] && [ -n "$ck" ] && date -Is > $out/TRAIN_DONE || { echo "A0 FAILED (no final checkpoint)" >> $TL; exit 1; }
      fi
      cfg=$out/A0/neurad/run/config.yml
      T1=$(date +%s)
      timeout 3000 python tools/v13/eval_foggyfield_v13.py --backend v12 --config $cfg --mode A0 --out $ROOT/eval/A0 \
        --heldout-json $HELD --anchor-flags $ANCH --cosmos-root $COS --run-id $RUN_ID > $ROOT/logs/eval_A0.log 2>&1 || echo "A0 eval FAILED" >> $TL
      timeout 1800 python tools/v13/render_foggyfield_v13.py --backend v12 --config $cfg --mode A0 --out $ROOT/renders/A0 \
        --heldout-json $HELD --anchor-flags $ANCH > $ROOT/logs/render_A0.log 2>&1 || echo "A0 render FAILED" >> $TL
      echo "A0 eval+render sec=$(( $(date +%s) - T1 ))" >> $TL
      exit 0
    fi
    # ---- V1.3 lane ----
    args_for() { echo "$(lane_args $1) ${CLIPFLAGS[*]}"; }
    echo "$(args_for $L)" > $out/lane_args.txt; echo "$(expect_for $L)" > $out/lane_expect.txt; prior_state > $out/prior_state.txt
    DPA=("${DP[@]}" --anchor-flags $ANCH)
    train_to() {   # $1 = target number of completed steps; resumes from the latest checkpoint of this lane
      local tgt=$1; shift
      local ck=$(ls $out/*/*/*/nerfstudio_models/step-*.ckpt 2>/dev/null | sort | tail -1)
      local done=0 load=()
      if [ -n "$ck" ]; then done=$(( 10#$(basename $ck .ckpt | sed 's/step-//') + 1 )); load=(--load-dir $(dirname $ck)); fi
      [ $done -ge $tgt ] && return 0
      local T0=$(date +%s)
      ns-train $METHOD --output-dir $out --experiment-name V13-$L --timestamp run --max-num-iterations $(( tgt - done )) --machine.seed 42 \
        --vis tensorboard --steps-per-eval-image 999999 --steps-per-eval-all-images 999999 --steps-per-save ${SAVE_EVERY:-100} \
        --save-only-latest-checkpoint False --viewer.quit-on-train-completion True --logging.steps-per-log 10 \
        $(args_for $L) "$@" "${load[@]}" "${DPA[@]}" >> $out/train.log 2>&1; local rc=$?
      echo "V13-$L train_to=$tgt from=$done rc=$rc sec=$(( $(date +%s) - T0 ))" >> $TL
      ck=$(ls $out/*/*/*/nerfstudio_models/step-$(printf %09d $((tgt-1))).ckpt 2>/dev/null | head -1)
      [ $rc -eq 0 ] && [ -n "$ck" ]
    }
    if [ ! -f $out/TRAIN_DONE ]; then
      train_to $B_END || { echo "V13-$L phase 1a FAILED" >> $TL; exit 1; }
      cfg=$(ls $out/*/*/*/config.yml | head -1)
      # validator K2: the resolved config must carry this lane's intended settings (prior, renderer, R_cam, LiDAR ingredients, clip)
      python jobs/v13colrho_cfgcheck.py $cfg $(expect_for $L) > $out/cfgcheck.txt 2>&1 \
        || { echo "V13-$L CONFIG CHECK FAILED (see $out/cfgcheck.txt): lane stopped after phase 1a" >> $TL; date -Is > $out/STOPPED_CFG; exit 1; }
      echo "V13-$L config check OK" >> $TL
      # decision 92: calibrate lambda ONCE on the stage-B checkpoint (eta at init), then freeze it
      tauflag=()
      [ -f $out/gate_B.json ] || timeout 3000 python jobs/v13_stage_gate.py --config $cfg --heldout $HELD \
          --anchor-flags $ANCH --stage B --n-sweeps 9 --out $out/gate_B.json > $ROOT/logs/gate_B_V13-$L.log 2>&1
      lam=$(python -c "import json; t=json.load(open('$out/gate_B.json'))['overlabel'].get('detect_lambda'); print(t if t else '')" 2>/dev/null)
      tau=$(python -c "import json; t=json.load(open('$out/gate_B.json'))['overlabel'].get('detect_tau'); print(t if t else '')" 2>/dev/null)
      sdet=$(python -c "import json; t=json.load(open('$out/gate_B.json'))['overlabel'].get('detect_s'); print(t if t else '')" 2>/dev/null)
      calok=$(python -c "import json; print(int(bool(json.load(open('$out/gate_B.json'))['overlabel'].get('calibration_ok'))))" 2>/dev/null)
      if [ "$calok" != 1 ]; then
        st=$(python -c "import json; print(json.load(open('$out/gate_B.json'))['overlabel'].get('detect_tau_status'))" 2>/dev/null)
        echo "V13-$L gate B calibration FAILED ($st): stage C NOT run for this lane" >> $TL
        date -Is > $out/STOPPED_AT_B; exit 1
      fi
      if [ -n "$lam" ]; then tauflag=(--pipeline.model.fog-v13.detect-lambda $lam)
        echo "V13-$L lambda=$lam (CALIBRATED_ON_ECHO_RATE; Poisson detection likelihood ON in C/D)" >> $TL
      elif [ -n "$tau" ] && [ -n "$sdet" ]; then tauflag=(--pipeline.model.fog-v13.detect-tau $tau --pipeline.model.fog-v13.detect-s $sdet)
        echo "V13-$L tau=$tau s=$sdet (CALIBRATED_ON_ECHO_RATE; detection likelihood ON in C/D)" >> $TL
      else echo "V13-$L tau NOT calibrated: drift gate will STOP (soft term off in D)" >> $TL; fi
      echo "${tauflag[@]}" > $out/tau_flags.txt
      train_to $C_END "${tauflag[@]}" || { echo "V13-$L phase 1b FAILED" >> $TL; exit 1; }
      softflag=()
      [ -f $out/gate_C.json ] || timeout 2400 python jobs/v13_stage_gate.py --config $cfg --heldout $HELD --anchor-flags $ANCH --stage C \
          --out $out/gate_C.json > $ROOT/logs/gate_C_V13-$L.log 2>&1 || echo "V13-$L gate C FAILED to run" >> $TL
      if python -c "import json,sys; sys.exit(0 if json.load(open('$out/gate_C.json'))['overlabel']['STOP'] else 1)" 2>/dev/null; then
        softflag=(--pipeline.model.fog-v13.lidar-soft-in-d False); echo "V13-$L drift gate tripped at end of C (rate outside [1/3,3] x observed): stage D with the soft term OFF" >> $TL
      elif [ ! -f $out/gate_C.json ]; then
        softflag=(--pipeline.model.fog-v13.lidar-soft-in-d False); echo "V13-$L gate C missing: stage D with the soft term OFF (conservative)" >> $TL
      fi
      echo "${softflag[@]}" > $out/stage_D_flags.txt
      tauflag=($(cat $out/tau_flags.txt 2>/dev/null))
      train_to $STEPS "${tauflag[@]}" "${softflag[@]}" || { echo "V13-$L phase 2 FAILED (stage-C checkpoint kept)" >> $TL; exit 1; }
      date -Is > $out/TRAIN_DONE
    fi
    [ -f $out/TRAIN_DONE ] || { echo "V13-$L NOT DONE (no final checkpoint)" >> $TL; exit 1; }
    [ "${SKIP_POST:-0}" = 1 ] && { echo "V13-$L TRAIN_DONE; post stages skipped (scored by the common evaluator)" >> $TL; exit 0; }
    cfg=$(ls $out/*/*/*/config.yml | head -1)
    T1=$(date +%s)
    [ -f $ROOT/eval/V13-$L/metrics/summary.json ] || timeout 3600 python tools/v13/eval_foggyfield_v13.py --backend v13 --config $cfg --mode V13-$L --out $ROOT/eval/V13-$L \
      --heldout-json $HELD --anchor-flags $ANCH --run-id $RUN_ID > $ROOT/logs/eval_V13-$L.log 2>&1 || echo "V13-$L eval FAILED" >> $TL
    # completion markers (not bare directories): a step killed by the wall clock is redone by the next resumable job
    [ -f $out/RENDER_DONE ] || { timeout 1800 python tools/v13/render_foggyfield_v13.py --backend v13 --config $cfg --mode V13-$L --out $ROOT/renders/V13-$L \
      --heldout-json $HELD --anchor-flags $ANCH > $ROOT/logs/render_V13-$L.log 2>&1 && date -Is > $out/RENDER_DONE || echo "V13-$L render FAILED" >> $TL; }
    [ -f $out/CF_DONE ] || { timeout 1500 python tools/v13/counterfactuals.py --backend v13 --config $cfg --out $ROOT/counterfactuals/V13-$L \
      --heldout-json $HELD --anchor-flags $ANCH > $ROOT/logs/cf_V13-$L.log 2>&1 && date -Is > $out/CF_DONE || echo "V13-$L counterfactuals FAILED" >> $TL; }
    [ -f $out/DEFOG_DONE ] || { timeout 1500 python tools/v13/defog_outputs.py render --config $cfg --out $ROOT/defog/V13-$L --heldout-json $HELD \
      --anchor-flags $ANCH > $ROOT/logs/defog_V13-$L.log 2>&1 && date -Is > $out/DEFOG_DONE || echo "V13-$L defog render FAILED" >> $TL; }
    echo "V13-$L eval+render+cf+defog sec=$(( $(date +%s) - T1 ))" >> $TL
    # agent V's 3D RGB-LiDAR visualisation (GPU, ~30 min incl. 25 min image-cache load under contention); failure or timeout never blocks the lane, markers make it resumable
    VIZ=/users/yitiwang/FoggyfieldV1/RGB-LiDAR-Viz/_pipeline/viz_model.sh
    # 2026-11-24 02:20 user: visualisation tasks dropped -> viz runs only when RUN_VIZ=1 is exported
    if [ "${RUN_VIZ:-0}" = 1 ] && [ -f $ROOT/eval/V13-$L/metrics/summary.json ] && [ -f $out/RENDER_DONE ] && [ ! -f $out/VIZ_DONE ] && [ -x $VIZ -o -f $VIZ ]; then
      T2=$(date +%s); timeout 3600 bash $VIZ $ROOT $L $clip > $ROOT/logs/viz_V13-$L.log 2>&1 && date -Is > $out/VIZ_DONE || echo "V13-$L viz FAILED/timeout (non-blocking)" >> $TL
      echo "V13-$L viz sec=$(( $(date +%s) - T2 ))" >> $TL
    fi
    [ -f $ROOT/eval/V13-$L/metrics/summary.json ] && [ -f $out/RENDER_DONE ] && [ -f $out/CF_DONE ] && [ -f $out/DEFOG_DONE ] && date -Is > $out/ALL_DONE
  )
  RC=$?
  g=$(( g + 1 ))
done
echo "[$(date)] ALL DONE ($TOKENS)"
for tok in $TOKENS; do clip=${tok%%:*}; echo "--- $clip"; cat $OV/20261123_${clip}_v13_final/logs/timings.txt 2>/dev/null; done
exit ${RC:-1}
