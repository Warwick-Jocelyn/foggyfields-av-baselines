#!/bin/bash
# Self-repair of venv-emernerf after the 2026-11-12 scratch purge (omegaconf / nerfacc kept only __pycache__, src/nerfacc lost
# its sources). Idempotent; run INSIDE the container on a GPU node before the EmerNeRF lane. Exit 0 = BUILD_OK.
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/emernerf/env_emernerf.sh
SP=$EVENV/lib/python3.10/site-packages
export PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= PIP_NO_INPUT=1 PIP_CACHE_DIR=$EVENV/cache/pip
log(){ echo "[$(date '+%F %T')] [emer_fix_env] $*"; }
ok_omega(){ "$EMER_PY" -c "from omegaconf import OmegaConf; import antlr4" 2>/dev/null; }
ok_nerfacc(){ "$EMER_PY" -c "from nerfacc import csrc; from nerfacc.pdf import importance_sampling; from nerfacc.data_specs import RayIntervals" 2>/dev/null; }
if ! ok_omega; then
  log "omegaconf broken -> reinstall 2.3.0"; rm -rf "$SP"/omegaconf "$SP"/omegaconf-*.dist-info "$SP"/antlr4 "$SP"/antlr4_python3_runtime-*.dist-info
  # 2026-11-25: pip's isolated sdist build of antlr4 fails on the compute nodes -> both are pure python: unpack directly.
  TMPX=$(mktemp -d); tar xzf "$EVENV/wheels/antlr4-python3-runtime-4.9.3.tar.gz" -C "$TMPX" \
    && cp -r "$TMPX/antlr4-python3-runtime-4.9.3/src/antlr4" "$SP/" \
    && "$EMER_PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$EVENV/wheels/omegaconf-2.3.0-py3-none-any.whl" "$SP"
  rm -rf "$TMPX"
  ok_omega || { log "!! omegaconf still broken"; exit 1; }
fi
if ! ok_nerfacc; then
  log "nerfacc .py sources missing? -> restore them from the sdist next to the existing csrc.so"
  TMPX=$(mktemp -d); tar xzf "$EVENV/wheels/nerfacc-0.5.0.tar.gz" -C "$TMPX" && (cd "$TMPX/nerfacc-0.5.0" && find nerfacc -name "*.py" -exec cp --parents {} "$SP/" \;); rm -rf "$TMPX"
fi
if ! ok_nerfacc; then
  log "nerfacc broken -> rebuild 0.5.0 for sm_90 (PyPI sdist, ~10 min)"; rm -rf "$SP"/nerfacc "$SP"/nerfacc-*.dist-info
  export BUILD_NO_CUDA=0 TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32
  "$EMER_PY" -m pip install --no-build-isolation --no-deps -v "$EVENV/wheels/nerfacc-0.5.0.tar.gz" 2>&1 | tail -5
  ok_nerfacc || { log "!! nerfacc still broken"; exit 1; }
fi
"$EMER_PY" -c "import numpy,sys; sys.exit(int(numpy.__version__.split('.')[0]) >= 2)" || "$EMER_PY" -m pip install "numpy<2" 2>&1 | tail -1
cd "$EMERNERF_ROOT" && "$EMER_PY" - <<'PY' || { log "!! smoke import/CUDA check failed"; exit 1; }
import sys, torch, numpy, nerfacc, omegaconf
sys.path.insert(0, ".")
from nerfacc import csrc
import third_party.tcnn_modules as tcnn
enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 15, "base_resolution": 16, "per_level_scale": 1.5})
x = torch.rand(4096, 3, device="cuda", requires_grad=True); y = enc(x); y.float().sum().backward()
from nerfacc.pdf import importance_sampling
from nerfacc.data_specs import RayIntervals
t = torch.sort(torch.rand(8, 33, device="cuda"), -1)[0]; w = torch.rand(8, 32, device="cuda")
iv, sm = importance_sampling(RayIntervals(vals=t), torch.cumsum(torch.cat([torch.zeros(8,1,device="cuda"), w], -1), -1) / w.sum(-1, keepdim=True), 16)
import datasets, builders, radiance_fields, loss, utils.misc  # noqa: F401
print("torch", torch.__version__, "numpy", numpy.__version__, "nerfacc", nerfacc.__version__, nerfacc.__file__, "| tcnn+nerfacc CUDA OK")
PY
log "BUILD_OK"
