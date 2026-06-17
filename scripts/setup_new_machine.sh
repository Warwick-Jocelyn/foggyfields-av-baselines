#!/bin/bash
# Clone the three upstream repos at their pinned commits and apply our patches.
# Usage:  bash scripts/setup_new_machine.sh /path/to/workdir
set -eo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:?usage: setup_new_machine.sh <workdir>}"
mkdir -p "$WORK"; cd "$WORK"

clone() { # url commit dir
  [ -d "$3" ] || git clone "$1" "$3"
  git -C "$3" fetch --depth 1 origin "$2" 2>/dev/null || true
  git -C "$3" checkout "$2"
}
clone https://github.com/georghess/neurad-studio.git e6f7e4e509b828a952d8584b7165f7844711ecb2 neurad-studio
clone https://github.com/carlinds/splatad.git        6e31ad766d39e0c33f9034a2ed772d51364b2343 splatad
clone https://github.com/NVlabs/EmerNeRF.git         8c051d7cccbad3b52c7b11a519c971b8ead97e1a EmerNeRF

apply() { # repo patch
  if git -C "$1" apply --check "$2" 2>/dev/null; then
    git -C "$1" apply "$2"; echo "applied $2 -> $1"
  else
    echo "!! patch $2 did not apply cleanly to $1 — falling back to src/ plain files"
    return 1
  fi
}
apply neurad-studio "$REPO/patches/neurad-studio.patch" || \
  cp "$REPO/src/neurad-studio/nerfstudio/data/dataparsers/nvidia_av_fog_dataparser.py" \
     neurad-studio/nerfstudio/data/dataparsers/
apply EmerNeRF "$REPO/patches/emernerf.patch" || {
  cp "$REPO"/src/emernerf/datasets/*.py EmerNeRF/datasets/
  cp "$REPO"/src/emernerf/configs/*.yaml EmerNeRF/configs/
}
echo "Done. Next: build the env + rebuild CUDA ext for your GPU arch (REPRODUCE.md §2,§4)."
