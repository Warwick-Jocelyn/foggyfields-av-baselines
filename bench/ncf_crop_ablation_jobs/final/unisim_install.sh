#!/bin/bash
# Register the UniSim plugin (code/_src/unisim) in the active venv (venv_neurad) if ns-train does not know it yet.
# Editable install, --no-deps, no network. Idempotent. Inside the container.
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/final/config_final.sh
source "$BW/jobs/env.sh" >/dev/null 2>&1
log(){ echo "[$(date '+%F %T')] [unisim_install] $*"; }
have(){ python - <<'PY' 2>/dev/null
from importlib.metadata import entry_points
import sys
eps = [e for e in entry_points(group="nerfstudio.method_configs") if e.name == "unisim"]
sys.exit(0 if eps else 1)
PY
}
if have; then log "unisim entry point present"; exit 0; fi
log "installing plugin editable into $VENV"
python -m pip install --no-deps --no-build-isolation --no-index -e "$BW/code/_src/unisim" 2>&1 | tail -3
have && { log "OK"; exit 0; } || { log "!! entry point still missing"; exit 1; }
