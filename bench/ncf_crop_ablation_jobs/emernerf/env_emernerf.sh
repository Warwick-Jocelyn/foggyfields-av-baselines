# Environment for the EmerNeRF lane of the ncf-crop-ablation harness. Source INSIDE the
# container (EDF /users/yitiwang/infra/projects/splatad-nff-repro.toml):
#
#   source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/emernerf/env_emernerf.sh
#
# Two interpreters, on purpose:
#   EMER_PY     $W/venv-emernerf  -- EmerNeRF train / render / LiDAR dump. Own venv
#               (--system-site-packages) + a .pth onto the splatad-nff-repro venv's
#               site-packages for its sm_90 tiny-cuda-nn; nerfacc (EmerNeRF's pinned
#               commit) is built for sm_90 into THIS venv. Nothing is installed into the
#               splatad venv.
#   HARNESS_PY  the splatad-nff-repro venv's own python, i.e. the exact interpreter lane.sh
#               uses for eval_common_region.py / make_videos.py / lidar_viz.py, so the
#               metrics come from the same code AND the same package versions.

export W=/capstor/scratch/cscs/yitiwang/ncf-crop-ablation
export REPRO=/capstor/scratch/cscs/yitiwang/splatad-nff-repro
export EVENV=$W/venv-emernerf
export EMER_PY=$EVENV/bin/python
export HARNESS_PY=${HARNESS_PY:-$REPRO/venv/bin/python}   # 2026-11-23: $REPRO/venv gutted by the purge; final bench passes venv_neurad
export EMERNERF_ROOT=$W/code/emernerf
export JOBS=$W/jobs
export EJOBS=$W/jobs/emernerf

# ---- where things go ----------------------------------------------------------------
# NCF clips live in $W/data, NFF-05 in $W/data_nff; each clip is resolved against
# DATA_ROOTS in order (DATA_ROOT, if set, wins and is the only root searched).
export DATA_ROOTS=${DATA_ROOTS:-"$W/data $W/data_nff"}
export DATA_ROOT=${DATA_ROOT:-}
export OUT_ROOT=${OUT_ROOT:-$W/outputs_emernerf}     # bulk: run dirs, PNG renders, npz dumps
export RES_ROOT=${RES_ROOT:-$W/results_emernerf}     # KEEPER: metrics json, timings.csv
export VIZ_ROOT=${VIZ_ROOT:-$W/viz_emernerf}         # KEEPER: mp4 + LiDAR figures
export LOG_ROOT=${LOG_ROOT:-$W/logs_emernerf}
export TIMINGS=${TIMINGS:-$RES_ROOT/timings.csv}
mkdir -p "$OUT_ROOT" "$RES_ROOT/metrics" "$RES_ROOT/lidar" "$VIZ_ROOT" "$LOG_ROOT/lanes"

# ---- GH200 / sm_90 ------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="9.0"
export TCNN_CUDA_ARCHITECTURES="90"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export MAX_JOBS=${MAX_JOBS:-32}

# ---- caches: our own, except TORCH_HOME (read-only use of the already-downloaded LPIPS /
# AlexNet weights, the same file lane.sh's metrics load) -------------------------------
export XDG_CACHE_HOME=$EVENV/cache
export PIP_CACHE_DIR=$EVENV/cache/pip
export TORCH_EXTENSIONS_DIR=$EVENV/cache/torch_extensions
export MPLCONFIGDIR=$EVENV/cache/matplotlib
export TORCH_HOME=$REPRO/cache/torch
export WANDB_MODE=disabled
export PIP_NO_INPUT=1

# ---- run knobs (all overridable) -------------------------------------------------------
export CLIPS=${CLIPS:-"NCF-02 NCF-07 NCF-10 NCF-15 NFF-05"}
export SPLIT=${SPLIT:-nvs_50_50}
# EmerNeRF's own full-length default: configs/default_config.yaml optim.num_iters = 25000
# (configs/nvidia_av_fog*.yaml do not override it; REPRODUCE.md used 25k as well).
export ITERS=${ITERS:-25000}
export EMER_CONFIG=${EMER_CONFIG:-configs/nvidia_av_fog_safecrop.yaml}
export EMER_EXTRA=${EMER_EXTRA:-}                 # extra OmegaConf CLI overrides, verbatim
export MAX_EVAL_FRAMES=${MAX_EVAL_FRAMES:-}       # smoke only: N evenly spaced held-out timesteps
export RENDER_CHUNK=${RENDER_CHUNK:-32768}
export LIDAR_CHUNK=${LIDAR_CHUNK:-65536}
export NLANES=${NLANES:-}                         # default: one lane per visible GPU
export FORCE=${FORCE:-0}
export PANEL_W=${PANEL_W:-960}
export VIZ_FPS=${VIZ_FPS:-10}

ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# clip -> data root that contains it (first match in DATA_ROOTS)
data_root_for () {
  local c=$1 r
  for r in ${DATA_ROOT:-$DATA_ROOTS}; do
    if "$HARNESS_PY" -I "$JOBS/clip_paths.py" --data "$r" --sequence "$c" --print-root >/dev/null 2>&1; then
      echo "$r"; return 0
    fi
  done
  return 1
}
export -f data_root_for
