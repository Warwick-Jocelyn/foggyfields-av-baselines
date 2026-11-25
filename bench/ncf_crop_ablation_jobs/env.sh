# Shared environment for the NCF crop/mask ablation. Source INSIDE the container.
#
#   source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/env.sh
#
# Caches deliberately point at the splatad-nff-repro workspace: that is where the
# sm_90 venv and the already-JIT-compiled torch extensions live, and it is what the
# EDF toml (--environment=.../splatad-nff-repro.toml) sets too. Pointing them here
# instead would silently trigger a ~1 h rebuild of gsplat/tcnn on first use.

export W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
export REPRO=/capstor/scratch/cscs/yitiwang/splatad-nff-repro
export VENV=${VENV:-$REPRO/venv}             # 2026-11-23: $REPRO/venv was gutted by the 11-12 purge; final bench passes venv_neurad
export CODE=$W/code
export NS_REPO=${NS_REPO:-$CODE/neurad-studio}   # symlink -> $REPRO/code/neurad-studio (final bench: the a2f5084f worktree, see jobs/final/)
export JOBS=$W/jobs

# ---- where things go --------------------------------------------------------------
export DATA_ROOT=${DATA_ROOT:-$W/data}        # holds clips/NCF-XX__<hash>/
export OUT_ROOT=${OUT_ROOT:-$W/outputs}       # bulk / disposable (ckpts + PNG renders)
export RES_ROOT=${RES_ROOT:-$W/results}       # KEEPER: metrics json + SUMMARY.md
export VIZ_ROOT=${VIZ_ROOT:-$W/viz}           # KEEPER: mp4 + montages
export LOG_ROOT=${LOG_ROOT:-$W/logs}
export TIMINGS=${TIMINGS:-$RES_ROOT/timings.csv}
mkdir -p "$OUT_ROOT" "$RES_ROOT/metrics" "$VIZ_ROOT" "$LOG_ROOT/lanes"

# ---- GH200 / sm_90 ----------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="9.0"
export TCNN_CUDA_ARCHITECTURES="90"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export MAX_JOBS=32

# ---- caches (shared with the repro workspace on purpose; see header) --------------
export XDG_CACHE_HOME=$REPRO/cache
export TORCH_HOME=$REPRO/cache/torch
export TORCH_EXTENSIONS_DIR=$REPRO/cache/torch_extensions
export PIP_CACHE_DIR=$REPRO/cache/pip
export HF_HOME=$REPRO/cache/huggingface
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
export PIP_NO_INPUT=1

# ---- experiment matrix (everything overridable from the environment) --------------
export CLIPS=${CLIPS:-"NCF-02 NCF-07 NCF-08 NCF-10 NCF-15"}
export VARIANTS=${VARIANTS:-"safe mask"}
export METHODS=${METHODS:-"splatad neurad"}       # also: unisim
export SPLIT=${SPLIT:-nvs_50_50}              # fixed split file -> identical frames for both variants
export SPLATAD_ITERS=${SPLATAD_ITERS:-30000}
export NEURAD_ITERS=${NEURAD_ITERS:-30000}   # match the reference protocol (REPRODUCE.md: NeuRAD 30k)
export UNISIM_ITERS=${UNISIM_ITERS:-20001}   # the plugin's paper config: stages 2500 + 7500 + 10001
export ITERS=${ITERS:-}                       # if set, overrides BOTH of the above (smoke runs)
export MAX_EVAL_FRAMES=${MAX_EVAL_FRAMES:-}   # cap held-out timesteps (seed 123, identical across variants)
export END_FRACTION=${END_FRACTION:-}         # use only the first X of each clip (smoke)
export NLANES=${NLANES:-4}
export SEED=${SEED:-42}
export FORCE=${FORCE:-0}                      # 1 = ignore stage stamps and redo everything
export SPLATAD_EXTRA=${SPLATAD_EXTRA:-}       # extra ns-train flags, appended verbatim
# NeuRAD flags. Two deliberate choices, each with a reason:
#   use-temporal-appearance False
#       Every published NeuRAD number came from `ns-train neurad-paper`, whose ONLY effective
#       difference from `ns-train neurad` is this flag (method_configs.py also sets
#       f.flip_prob=0.0 on the field configs, but those have no such attribute -- it is a dead
#       assignment, so actor flip stays at 0.25/0.5 in both). With it on, the model gets
#       8 sensors x 20 one-second bins = 160 free appearance codes, an extra degree of freedom
#       the published runs did not have.
#   add-missing-points True                                    (NEURAD_DP_EXTRA, see below)
#       Keeps the ray-drop loss alive (ad_pipeline.py:72 calls disable_ray_drop() when this is
#       False). On this dataparser the flag does nothing else -- it does NOT synthesise points.
#       Only meaningful now that the range-image loader actually reads the published .npz files.
#   (checkpoint frequency is SAVE_EVERY below, not a model flag: it does not change the result.)
export NEURAD_EXTRA=${NEURAD_EXTRA:-"--pipeline.model.use-temporal-appearance False"}
# *_DP_EXTRA are DATAPARSER options. They must come AFTER the `nvidia-av-fog-data` subcommand:
# written before it as --pipeline.datamanager.dataparser.<x>, tyro rejects them ("Unrecognized
# options") and ns-train exits 2 before step 0. lane.sh and the pre-flight both append them there.
export NEURAD_DP_EXTRA=${NEURAD_DP_EXTRA:-"--add-missing-points True"}
# UniSim (pip plugin carlinds/unisim, source in $CODE/_src/unisim), paper config as shipped.
#   use-range-image-raydrop False
#       The paper's UniSim does not model ray drop (ray_drop_loss_mult=0, add_missing_points=False
#       in its config). Its depth/intensity losses already mask by did_return, so the ~35% range-image
#       non-return rays would only eat the 8192-ray LiDAR budget without training anything.
export UNISIM_EXTRA=${UNISIM_EXTRA:-}
# Checkpoint every SAVE_EVERY steps (nerfstudio keeps only the newest). lane.sh resumes an
# interrupted train from it, so a run cut off by the 1.5 h debug wall clock loses < SAVE_EVERY steps.
export SAVE_EVERY=${SAVE_EVERY:-2000}
export UNISIM_DP_EXTRA=${UNISIM_DP_EXTRA:-"--use-range-image-raydrop False"}
export PANEL_W=${PANEL_W:-960}                # per-panel width in the side-by-side videos
export VIZ_FPS=${VIZ_FPS:-10}
export MEM_BUDGET_FRAC=${MEM_BUDGET_FRAC:-0.75}   # of node MemTotal, for the image-cache check
export ALLOW_BIG_CACHE=${ALLOW_BIG_CACHE:-0}

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
# When NS_REPO is not the checkout the venv's editable install points at, put it first on PYTHONPATH
# (the editable finder sits after sys.path, so PYTHONPATH wins; same scheme as the v13 jobs).
case "$NS_REPO" in "$CODE/neurad-studio") ;; *) export PYTHONPATH=$NS_REPO${PYTHONPATH:+:$PYTHONPATH} ;; esac
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Effective iteration count for a method.
iters_for () {
  case "$1" in
    splatad) echo "${ITERS:-$SPLATAD_ITERS}" ;;
    neurad)  echo "${ITERS:-$NEURAD_ITERS}" ;;
    neurad-foggy) echo "${ITERS:-${FOGGY_ITERS:-30000}}" ;;
    unisim)  echo "${ITERS:-$UNISIM_ITERS}" ;;
    *) echo "iters_for: unknown method '$1'" >&2; return 1 ;;
  esac
}
export -f iters_for
