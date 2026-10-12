#!/bin/bash
#SBATCH --job-name=splatad_build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --account=a0195
#SBATCH --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml
#SBATCH --output=/capstor/scratch/cscs/yitiwang/splatad-nff-repro/logs/build_%j.log
# Build the SplatAD venv for GH200 (sm_90) on top of the NGC container torch, then smoke-test.
set -eo pipefail
source /capstor/scratch/cscs/yitiwang/splatad-nff-repro/jobs/env.sh
echo "[$(date)] node=$(hostname)"; nvidia-smi -L; nvidia-smi --query-gpu=compute_cap,driver_version --format=csv,noheader
python -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,'py',sys.version.split()[0],'cap',torch.cuda.get_device_capability())"
which nvcc && nvcc --version | tail -1

# ---- 1. venv on scratch (python -m venv lacks ensurepip in this image -> virtualenv) ----
if [ ! -f "$VENV/bin/activate" ]; then
  pip install -q virtualenv
  python -m virtualenv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -q -U pip "setuptools<70" wheel ninja

# ---- 2. neurad-studio (no-deps) + explicit runtime deps ----
cd $CODE/neurad-studio
pip install -q --no-deps -e .
pip install -q "appdirs" "av>=9.2.0" "comet_ml" "cryptography>=38" "tyro>=0.6.6" "gdown>=4.6.0" "ninja>=1.10" \
  "h5py>=2.9.0" "imageio>=2.21.1" "importlib-metadata>=6.0.0" "ipywidgets>=7.6" "jaxtyping>=0.2.15" \
  "matplotlib>=3.6.0" "mediapy>=1.1.0" "msgpack>=1.0.4" "msgpack_numpy>=0.4.8" "Pillow>=10.3.0" "plotly>=5.7.0" \
  "protobuf<=3.20.3,!=3.20.0" "pyngrok>=5.1.0" "python-socketio>=5.7.1" "pyquaternion>=0.9.9" "requests" "rich>=12.5.1" \
  "scikit-image>=0.19.3" "splines==0.3.0" "tensorboard>=2.13.0" "torchmetrics[image]>=1.0.1" "typing_extensions>=4.4.0" \
  "wandb>=0.13.3" "xatlas" "trimesh>=3.20.2" "timm==0.6.7" "pytorch-msssim" "pathos" "packaging" "zod>=0.1.7" "av2==0.2.1" \
  "nuscenes-devkit>=1.1.1" "fastapi[all]==0.110" "pyarrow" "pandas" "imageio-ffmpeg" \
  "git+https://github.com/atonderski/viser.git@57142e42df8edd4de33fd60a08d6bb6c35970aa1" \
  "git+https://github.com/scaleapi/pandaset-devkit.git#egg=pandaset&subdirectory=python" || echo "!! some deps failed, continuing"

# ---- 3. CUDA extensions for sm_90 (torch visible at build -> no build isolation) ----
pip install -q --no-build-isolation "nerfacc==0.5.2" || pip install -q --no-build-isolation "git+https://github.com/nerfstudio-project/nerfacc.git@v0.5.2"
pip install -q --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"
(cd $CODE/splatad && git submodule update --init --recursive && pip install -q --no-build-isolation --no-deps -e .)

# ---- 4. headless cv2, numpy<2 LAST (container torch needs it) ----
pip uninstall -q -y opencv-python opencv-contrib-python 2>/dev/null || true
pip install -q "opencv-python-headless<4.11"
pip install -q "dataclass-wizard==0.22.3" "protobuf<=3.20.3,!=3.20.0"
pip install -q "numpy==1.26.4"

# ---- 5. import smoke with auto-fix of missing modules ----
python $W/jobs/import_fix.py

# ---- 6. gsplat JIT compile + dataparser CPU check + 60-iter training smoke on NFF-03 ----
cd $CODE/neurad-studio
python - <<'PY'
import torch, gsplat, tinycudann, nerfacc
print("gsplat", gsplat.__version__, "tcnn ok", "nerfacc", nerfacc.__version__)
from gsplat.cuda._backend import _C  # triggers JIT build for sm_90
print("gsplat CUDA backend built")
PY
python $W/jobs/dataparser_check.py --data "$DATA_ROOT" --sequence 004
SMOKE=$OUT_ROOT/_smoke; rm -rf "$SMOKE"; mkdir -p "$SMOKE"
timeout 1500 ns-train splatad --output-dir "$SMOKE" --vis tensorboard --max-num-iterations 60 \
  --steps-per-eval-image 999999 --steps-per-eval-all-images 999999 --steps-per-save 60 --machine.seed 42 \
  --pipeline.model.strategy mcmc --pipeline.model.mcmc-cap-max 5000000 --pipeline.model.max-num-seed-points 2000000 \
  --viewer.quit-on-train-completion True \
  nvidia-av-fog-data --data "$DATA_ROOT" --sequence 004 --train-split-fraction 0.5
find "$SMOKE" -name 'step-*.ckpt' -print
echo "[$(date)] BUILD+SMOKE OK"
