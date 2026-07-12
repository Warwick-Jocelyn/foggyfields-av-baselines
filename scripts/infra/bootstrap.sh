#!/bin/bash
# bootstrap.sh — rebuild a purged /iopsstor scratch working tree from the durable
# sources of truth (GitHub + HuggingFace + /capstor/store). No GPU used.
#
#   bash scripts/infra/bootstrap.sh [WORKDIR]
#
# WORKDIR default: /iopsstor/scratch/cscs/$USER/projects/foggyfields
# Idempotent: safe to re-run; skips what already exists.
set -eo pipefail

WORK="${1:-/iopsstor/scratch/cscs/$USER/projects/foggyfields}"
CODE_URL="https://github.com/Warwick-Jocelyn/foggyfields-av-baselines.git"
HF_DATASET="JocelynW/NFF"
STORE="/capstor/store/cscs/swissai/a0195/$USER"
export GIT_TERMINAL_PROMPT=0

echo "== bootstrap into $WORK =="
mkdir -p "$WORK"; cd "$WORK"

# 1. code repo (this repo) --------------------------------------------------
if [ -d code/.git ]; then
  echo "[1/4] code/: pulling latest"; git -C code pull --ff-only || echo "  (pull skipped)"
else
  echo "[1/4] code/: cloning"; rm -rf code; git clone "$CODE_URL" code
fi

# 2. upstreams + patches (delegates to the repo's own setup) ----------------
echo "[2/4] upstreams (neurad-studio, splatad, EmerNeRF at pinned commits + patches)"
bash code/scripts/setup_new_machine.sh "$WORK"

# 3. dataset from HuggingFace (gated -> needs your HF token) -----------------
echo "[3/4] dataset $HF_DATASET -> Dataset/NFF"
if command -v hf >/dev/null; then
  hf download "$HF_DATASET" --repo-type dataset --local-dir Dataset/NFF
else
  echo "  !! 'hf' CLI missing: pip install -U huggingface_hub, then 'hf auth login'"
fi

# 4. containers + configs from durable store --------------------------------
echo "[4/4] containers + configs from $STORE"
if [ -d "$STORE/containers" ]; then
  mkdir -p containers && rsync -a "$STORE/containers/" containers/
else echo "  (no $STORE/containers — skipping)"; fi
if [ -d "$STORE/configs_backup" ]; then
  rsync -a "$STORE/configs_backup/" code/configs_backup/ 2>/dev/null || true
fi

cat <<EOF

== done ==
Next (needs a GPU node — see code/REPRODUCE.md §2,§4):
  - build env + REBUILD CUDA ext for THIS GPU arch (sm_90 on GH200)
  - checkpoints were NOT restored (none survived); retrain or pull from the HF
    checkpoints repo if you have uploaded one.
EOF
