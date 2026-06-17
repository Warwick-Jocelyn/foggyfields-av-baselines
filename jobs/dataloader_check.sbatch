#!/bin/bash
#SBATCH --job-name=dl_check
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/dl_check_%j.log

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""   # force CPU-only: torch.cuda.is_available()==False

echo "[$(date)] node=$(hostname)  CPU-only dataloader sanity check"
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

cd ~/ETH/Code/SplatAD/neurad-studio

python -u - <<'PY'
"""
CPU-only sanity check for PandaSet test subset.
Loads scene 011 via the official dataparser and prints shapes — no CUDA.
"""
from pathlib import Path
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig

TEST_ROOT = Path("/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset")
for seq in ("011", "078"):
    print(f"\n=== sequence {seq} ===")
    cfg = PandaSetDataParserConfig(data=TEST_ROOT, sequence=seq)
    parser = cfg.setup()
    out = parser._generate_dataparser_outputs(split="train")

    cams = out.cameras
    print(f"cameras: N={len(cams)}  W={int(cams.width[0])}  H_range=[{int(cams.height.min())},{int(cams.height.max())}]")
    print(f"  fx[0]={float(cams.fx[0]):.2f}  cx[0]={float(cams.cx[0]):.2f}  cy[0]={float(cams.cy[0]):.2f}")
    print(f"  t-range=[{float(cams.times.min()):.3f},{float(cams.times.max()):.3f}]  span={float(cams.times.max()-cams.times.min()):.3f}s")

    md = out.metadata
    print(f"metadata keys: {sorted(md.keys())}")
    if "lidars" in md:
        lid = md["lidars"]
        print(f"lidars: N={len(lid)}")
    if "trajectories" in md:
        trajs = md["trajectories"]
        print(f"dynamic actor trajectories: {len(trajs)}")
        if trajs:
            labels = {}
            for t in trajs:
                labels[t['label']] = labels.get(t['label'], 0) + 1
            print(f"  label histogram: {labels}")
            t0 = trajs[0]
            print(f"  sample traj[0]: label={t0['label']!r}  T={len(t0['poses'])}  dims={t0['dims'].tolist()}")

    print(f"image filenames: {len(out.image_filenames)} (first: {out.image_filenames[0]})")
print("\n[OK] dataloader self-check passed for both scenes")
PY

echo "[$(date)] DONE"
