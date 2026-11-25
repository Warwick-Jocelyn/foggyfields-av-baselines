#!/bin/bash
# Build $W/venv-emernerf. Run INSIDE the container on a compute node (it compiles nerfacc's
# CUDA kernels for sm_90; not for a login node):
#
#   srun -A a0195 -p debug -t 00:40:00 --gres=gpu:1 \
#     --environment=/users/yitiwang/infra/projects/splatad-nff-repro.toml \
#     bash /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/emernerf/build_env.sh
#
# Recipe (idempotent):
#   1. virtualenv --system-site-packages on the container python (NGC torch 2.5, CUDA 12.6)
#   2. a .pth onto the splatad-nff-repro venv's site-packages -> its sm_90 tiny-cuda-nn
#      (+ wandb, plotly, scikit-image, ...). Read-only: nothing is installed there.
#   3. omegaconf 2.3.0 (EmerNeRF pin) if the stack lacks it
#   4. nerfacc @ 8340e19 (EmerNeRF README pin, v0.5.0) built AOT for sm_90 into this venv,
#      so it shadows the splatad venv's JIT-only nerfacc 0.5.2
#   5. numpy<2 re-pinned last if anything dragged numpy 2 in
#   6. import + CUDA smoke check
set -eo pipefail
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/emernerf/env_emernerf.sh
SP_REPRO=$REPRO/venv/lib/python3.10/site-packages
echo "[$(date -Is)] build $EVENV on $(hostname)"; nvidia-smi -L || true

if [ ! -f "$EVENV/.venv_ok" ]; then
  # the container python has no ensurepip (python -m venv fails) and no virtualenv module:
  # create the venv without pip and use the container's system pip, which installs into
  # the venv prefix because the venv has --system-site-packages
  rm -rf "$EVENV/bin" "$EVENV/lib" "$EVENV/lib64" "$EVENV/include" "$EVENV/pyvenv.cfg"
  python -m venv --without-pip --system-site-packages "$EVENV"
  "$EVENV/bin/python" -m pip --version && touch "$EVENV/.venv_ok"
fi
SP=$("$EVENV/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
echo "$SP_REPRO" > "$SP/zz_splatad_nff_repro_venv.pth"     # zz_: processed last -> lowest priority
source "$EVENV/bin/activate"
python -m pip --version

python -c "import omegaconf" 2>/dev/null || python -m pip install "omegaconf==2.3.0"
if ! python -c "from nerfacc import csrc" 2>/dev/null; then
  export BUILD_NO_CUDA=0
  python -m pip install --no-build-isolation --no-deps -v "$EVENV/src/nerfacc" 2>&1 | tail -20
fi
python -c "import numpy,sys; sys.exit(int(numpy.__version__.split('.')[0]) >= 2)" || python -m pip install "numpy<2"

cd "$EMERNERF_ROOT"
python - <<'EOF'
import sys, torch, numpy, nerfacc, omegaconf
sys.path.insert(0, ".")
from nerfacc import csrc
import third_party.tcnn_modules as tcnn
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda,
      "| numpy", numpy.__version__, "| nerfacc", nerfacc.__version__, nerfacc.__file__)
print("tcnn binding:", tcnn._C.__file__, "| device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2,
                        "log2_hashmap_size": 15, "base_resolution": 16, "per_level_scale": 1.5})
x = torch.rand(4096, 3, device="cuda", requires_grad=True)
y = enc(x); y.float().sum().backward()
from nerfacc.pdf import searchsorted, importance_sampling
from nerfacc.data_specs import RayIntervals
t = torch.sort(torch.rand(8, 33, device="cuda"), -1)[0]
w = torch.rand(8, 32, device="cuda")
iv, sm = importance_sampling(RayIntervals(vals=t), torch.cumsum(torch.cat([torch.zeros(8,1,device="cuda"), w], -1), -1) / w.sum(-1, keepdim=True), 16)
import datasets, builders, radiance_fields, loss, utils.misc  # noqa: F401  full EmerNeRF import graph
print("hash-grid fwd/bwd OK", tuple(y.shape), "| nerfacc importance_sampling OK", tuple(sm.vals.shape))
print("BUILD_OK")
EOF
