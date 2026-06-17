# SplatAD reproduction — environment & setup log

Source of truth for the **`splatad`** conda env and how to run things on this cluster.
Update this file whenever you change the env, install a new dep, or change a SLURM script.

---

## 0. Layout (relocated 2026-05-17)

```
~/ETH/Code/SplatAD/            # ← project root (was ~/code/SplatAD/)
├── neurad-studio/             # cloned from github.com/georghess/neurad-studio
├── splatad/                   # cloned from github.com/carlinds/splatad (gsplat fork)
├── env/                       # frozen snapshots of the conda env (date-stamped)
├── jobs/                      # sbatch scripts + slurm logs
└── SETUP.md                   # this file

~/ETH/Dataset/
└── pandaset/                  # full PandaSet, 103 scenes (was ~/code/SplatAD/data/pandaset/)

~/ETH/Code/SplatAD/dataset/    # SMALL test subset — what the dataparser points at first
├── 011/                       # scene 011 (renamed from 011-1) — full 6 cameras + lidar (366 MB)
└── 078/                       # scene 078 (renamed from 078_night_1) — full 6 cameras + lidar (395 MB)
```

**Why this split:**
- `ETH/Code/SplatAD/` — long-term project home
- `ETH/Dataset/pandaset/` — full 43 GB dataset, kept out of the project
- `~/ETH/Code/SplatAD/dataset/` — tiny test bed (≈760 MB total) for fast iteration. Pass `--data /networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset --sequence 011` (or `078`) to point any training at the subset.

---

## 1. Conda environment: `splatad`

- **Name:** `splatad`
- **Python:** 3.10
- **CUDA toolkit:** 11.8 (installed *inside* the env via `nvidia/label/cuda-11.8.0`)
- **PyTorch:** `2.0.1+cu118`
- **torchvision:** `0.15.2+cu118`

### Required env var
```bash
PYTHONNOUSERSITE=1
```
We pinned this with `conda env config vars set PYTHONNOUSERSITE=1` inside the env.
**Reason:** there is a stale `torch` and a stale `huggingface_hub` in
`~/.local/lib/python3.10/site-packages/` that shadow the env. Without this var
imports break (e.g. `libtorch_global_deps.so: cannot open shared object`).

### Exact create + install commands actually run
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n splatad python=3.10
conda activate splatad
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate && conda activate splatad      # re-activate so var takes effect

# Upgrade pip / pin setuptools<70 (tiny-cuda-nn setup.py uses pkg_resources)
python -m pip install --upgrade pip "setuptools<70.0"

# PyTorch + CUDA 11.8 wheels
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Things torch's wheel doesn't drag in (because we have user-site disabled)
python -m pip install typing_extensions networkx sympy filelock numpy pillow dill

# CUDA toolkit (nvcc, cuda libs) — needed to compile tiny-cuda-nn & the gsplat fork
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# HuggingFace client (for PandaSet download)
python -m pip install huggingface_hub requests

# Build helper
python -m pip install ninja
```

### Still to install (run on a GPU node — see §3)
```bash
# tiny-cuda-nn  -> submitted as sbatch job (see jobs/build_tcnn.sbatch)
pip install --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# neurad-studio (editable)
cd ~/ETH/Code/SplatAD/neurad-studio && pip install -e .

# carlinds/splatad gsplat fork  (provides lidar rasterization, rolling shutter)
cd ~/ETH/Code/SplatAD/splatad && pip install -e .

# Optional: Waymo Open Dataset v2 support
# pip install waymo-open-dataset-tf-2-11-0==1.6.1
```

### Snapshots
- `env/requirements_snapshot_<date>.txt` — `pip freeze` at that date
- `env/conda_env_<date>.yaml`           — `conda env export` at that date

To restore the env from a snapshot:
```bash
conda env create -n splatad -f env/conda_env_<date>.yaml
```

---

## 2. Cluster gotchas (read before debugging weird crashes)

### 2.1 Login node kills heavy processes with SIGKILL
- The login node has a per-user cgroup memory cap.
- `nvcc` compiling tiny-cuda-nn or `hf_hub_download` of a 44 GB file both got OOM-killed (`[code=137]`) here.
- **Rule:** any compile, any download >1 GB, any training → SLURM. Never on login.
- Compute nodes (`gpu-01/02/03`) have **351 GB RAM** and internet access (verified `curl -I huggingface.co` returns 200).

### 2.2 GPU partitions
| Partition | Time | GPU | Notes |
|-----------|------|-----|-------|
| `debug`   | 12h  | L40 (48GB) ×4 per node | default |
| `short`   | 1h   | L40                     | quick tests |
| `medium`  | 6h   | L40                     | tcnn build, smoke training |
| `long`    | 2d   | L40                     | full training, downloads |
| `xlong`   | 14d  | L40                     | long runs |
| `test`    | 3d   | **RTX PRO 6000 Blackwell (96GB) ×8** on `gpu-04` | requires CUDA ≥ 12.8 — **NOT compatible with our CUDA 11.8 PyTorch** |

We use **L40 (sm_89)** nodes. `TCNN_CUDA_ARCHITECTURES=89` is set in the tcnn build script for this reason.

### 2.3 User-site shadowing (`~/.local/lib/python3.10/site-packages/`)
Old installs there leak into any python3.10 process. Always run with `PYTHONNOUSERSITE=1` (the env has this pinned). If a tool ignores env vars, prepend `PYTHONNOUSERSITE=1 ` explicitly.

---

## 3. SLURM scripts in `jobs/`

### `download_pandaset.sbatch`
- Partition `long`, 4 CPU, 16 GB RAM, 24 h, `--requeue`.
- Resumes `pandaset.zip` (~44.5 GB) from HF (`georghess/pandaset`) into `data/pandaset.zip`, then unzips to `data/pandaset/`.

### `build_tcnn.sbatch`
- Partition `medium`, 1× L40, 8 CPU, 64 GB RAM, 2 h.
- Sets `CUDA_HOME=$CONDA_PREFIX`, `TCNN_CUDA_ARCHITECTURES=89`, `MAX_JOBS=2` (avoid OOM during parallel nvcc).
- Verifies `import tinycudann as tcnn` at the end.

### Submit
```bash
cd ~/ETH/Code/SplatAD/jobs
sbatch download_pandaset.sbatch
sbatch build_tcnn.sbatch
squeue -u $USER
```

### Logs land at
```
jobs/pandaset_dl_<jobid>.log
jobs/tcnn_build_<jobid>.log
```

---

## 4. Dataset (PandaSet)

- 1 file on HF: `pandaset.zip` (~44.5 GB, Git LFS).
- 103 scenes, 8 s each, 6 cameras + 64-beam lidar.
- Expected location per neurad-studio README: `data/pandaset/` (a directory tree, not a zip).
- Disk: home has ~570 GB free; zip + extracted ≈ 100 GB peak — fine.

---

## 5. Training (after env + dataset are ready)

From `~/ETH/Code/SplatAD/neurad-studio/`:
```bash
# Point at the small test subset first (only 011 & 078 are present there)
DATA=/networkhome/WMGDS/wang3_y/ETH/Code/SplatAD/dataset

# NeuRAD baseline
python nerfstudio/scripts/train.py neurad pandaset-data --data $DATA --sequence 011
# SplatAD (3DGS) on scene 078
python nerfstudio/scripts/train.py splatad pandaset-data --data $DATA --sequence 078
```

When you want to use the full set, swap `--data` to `/networkhome/WMGDS/wang3_y/ETH/Dataset/pandaset`.

For SLURM, wrap the same line in an sbatch script with `--gres=gpu:L40:1` and at least `--mem=64G`.

---

## 6. Current state (update as we go)

- [x] conda env `splatad` created and snapshotted
- [x] PyTorch 2.0.1 + cu118 working
- [x] CUDA 11.8 toolkit in env (`nvcc` works)
- [x] neurad-studio + splatad gsplat fork cloned
- [x] **PandaSet downloaded + extracted + flattened** (zip deleted; 103 scenes at `data/pandaset/<id>/`)
- [x] tiny-cuda-nn installed  (sbatch job 10433, CPU-only, 4 min)
- [x] Project relocated to `~/ETH/Code/SplatAD/`, full dataset to `~/ETH/Dataset/pandaset/`, test subset at `~/ETH/Code/SplatAD/dataset/{011,078}`
- [ ] neurad-studio editable install (job 10436 cancelled mid-flight before relocation; will resubmit from new path)
- [ ] SplatAD smoke training on scene 011 (test subset)

### Notes on the pandaset extraction
- Path layout inside `data/pandaset/`: scenes are `001`, `002`, ..., plus some with suffixes (`011-1`, `057_night`, `149_night`, ...). Total 103.
- The unzip log shows warnings about `pandaset/pandaset/149/` (no such file) — this is a stale ghost entry in the zip; the real scene is `149_night`. Harmless.
- The original double-nested layout (`data/pandaset/pandaset/<id>/`) was flattened with `shopt -s dotglob && mv -- * ../ && rmdir pandaset`.
- `pandaset.zip` was deleted after verification (freed 42 GB).

### tcnn build note
- First attempt (job 10432) requested 1 L40 + waited indefinitely in PENDING (Resources). Cancelled.
- Job 10433 is **CPU-only**: tcnn can compile without a GPU as long as `TCNN_CUDA_ARCHITECTURES=89` is set. Schedules instantly. Final `import tinycudann` test was moved to a separate step (needs GPU).
