#!/bin/bash
#SBATCH --job-name=pandaset_dl
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/jobs/pandaset_dl_%j.log
#SBATCH --requeue

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate splatad
export PYTHONNOUSERSITE=1

cd ~/ETH/Dataset    # full pandaset lives here now (historical script, kept for reproducibility)

echo "[$(date)] starting hf_hub_download (resume mode)"
python -u -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='georghess/pandaset',
    filename='pandaset.zip',
    repo_type='dataset',
    local_dir='.',
)
print('downloaded:', p)
"
echo "[$(date)] download finished. unzipping..."

mkdir -p pandaset
unzip -q -o pandaset.zip -d pandaset
echo "[$(date)] unzip finished. top-level:"
ls pandaset | head -20
du -sh pandaset
