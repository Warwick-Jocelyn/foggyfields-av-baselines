# FoggyFields — AV Neural-Field Baselines (SplatAD · NeuRAD · EmerNeRF)

Reproduction of three autonomous-driving neural-field methods on a custom **foggy**
multi-camera + LiDAR benchmark, plus PandaSet reference runs. Part of the
**FoggyFields** project (physically-consistent RGB + LiDAR foggy neural field).

| Method | Type | Repo (upstream) | Pinned commit |
|---|---|---|---|
| **SplatAD** (CVPR'25) | 3D Gaussian Splatting for AD | `carlinds/splatad` (gsplat) + `georghess/neurad-studio` | `6e31ad7` / `e6f7e4e` |
| **NeuRAD** (CVPR'24) | NeRF for AD | `georghess/neurad-studio` | `e6f7e4e` |
| **EmerNeRF** (NeurIPS'23) | self-supervised dynamic NeRF | `NVlabs/EmerNeRF` | `8c051d7` |

Datasets:
- **NVIDIA_AV_Fog** — 3 foggy clips (`002` rural/medium · `003` residential/light · `004` highway/heavy), 7 pinhole cameras + 1×128-line LiDAR, derived from NVIDIA PhysicalAI-AV / NCore.
- **PandaSet** — reference clips `011`, `078` (6 cameras + LiDAR).

> **This repo is code-only.** Checkpoints (~18 GB) and the processed dataset (~4.4 GB)
> live on HuggingFace — see [`CHECKPOINTS.md`](CHECKPOINTS.md). Renders/videos are
> **not** stored anywhere: they regenerate from a checkpoint in minutes (see REPRODUCE).

---

## Why this repo exists / how to use it on a new machine

We do **not** vendor the three upstream forks (they are ~100 MB + 4 GB). Instead we ship
our changes as **git patches against pinned upstream commits** (`patches/`), with
**plain-file copies** of every new/modified file as a fallback (`src/`). A fresh machine
(or a fresh agent) reproduces the full environment by:

1. cloning the three upstreams at the pinned commits (`BASE_COMMITS.txt`),
2. applying our patches,
3. building the env (conda or Apptainer) and **rebuilding the CUDA extensions for the
   target GPU arch** (gsplat + tiny-cuda-nn are arch-specific — this is the one real
   reproducibility gotcha; see [`REPRODUCE.md`](REPRODUCE.md) §4),
4. downloading data + checkpoints from HuggingFace,
5. training or just rendering/evaluating.

**Start here → [`REPRODUCE.md`](REPRODUCE.md)** (step-by-step, copy-pasteable).

---

## Repo layout

```
foggyfields-av-baselines/
├── README.md                ← you are here
├── REPRODUCE.md             ← master step-by-step reproduce guide (READ THIS)
├── RESULTS.md               ← results report: PSNR/SSIM/LPIPS/LiDAR, time, iterations, best ckpt
├── CHECKPOINTS.md           ← HuggingFace model/dataset manifest (what to download)
├── BASE_COMMITS.txt         ← exact upstream commits to clone
├── patches/
│   ├── neurad-studio.patch  ← our SplatAD+NeuRAD changes (dataparser + 4 files)
│   └── emernerf.patch       ← our EmerNeRF changes (loader + configs, incl. split fix)
├── src/                     ← plain-file fallback of every new/modified file
│   ├── neurad-studio/.../nvidia_av_fog_dataparser.py
│   └── emernerf/{datasets/*, configs/*}
├── jobs/                    ← SLURM sbatch + render/metric/viz Python (38 scripts)
│   ├── nvidia_fog_{splatad,neurad,emernerf}.sbatch   ← train + render + metrics drivers
│   ├── pngs_to_mp4_nvidia_fog.py   ← ★ the 7-camera panorama video renderer
│   ├── compute_{rgb,lidar}_metrics.py, compute_lpips_emernerf.py
│   └── ... (bbox overlay, lidar BEV, env-build scripts)
├── env/                     ← conda yaml + pip freeze + SETUP.md (Blackwell build notes)
├── results/metrics/nvidia/  ← authoritative per-clip metric JSONs (committed)
└── scripts/
    ├── setup_new_machine.sh ← clone upstreams + apply patches (one command)
    ├── stage_checkpoints.sh ← gather best checkpoints into an HF upload tree
    └── upload_hf.sh         ← push checkpoints to HuggingFace (after `hf auth login`)
```

---

## Headline results (held-out novel-view, nvs_50_50)

NVIDIA_AV_Fog, mean over clips 002/003/004 — full table in [`RESULTS.md`](RESULTS.md):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | LiDAR depth median↓ | Best at |
|---|---|---|---|---|---|
| EmerNeRF† | ~33 | ~0.95 | ~0.16 | — | RGB (all clips) |
| NeuRAD | 28.9 | 0.897 | 0.164 | **7.4 m** | LiDAR depth |
| SplatAD | 23.4 | 0.868 | 0.363 | 10.7 m | speed |

† EmerNeRF NVIDIA numbers are **pending re-train** — see the split-fix note below.

---

## ⚠️ Important: EmerNeRF eval-split fix (must re-train EmerNeRF)

A bug was found and fixed: EmerNeRF previously held out the **complementary** 50/50
frame set (even frames) vs SplatAD/NeuRAD (odd frames), so the old EmerNeRF numbers were
measured on the *opposite* views and are not strictly comparable. The loader now reads the
official `splits/nvs_50_50.json` directly (`src/emernerf/datasets/nvidia_av_fog.py` →
`split_train_test`), so **all three methods hold out the identical odd frames**.

→ The EmerNeRF NVIDIA checkpoints on HuggingFace are the **pre-fix** ones (flagged stale);
**re-train EmerNeRF on the new machine** with the fixed loader to refresh those numbers.
SplatAD/NeuRAD are unaffected and final.

---

## Conventions / what was hard (so you don't rediscover it)

- **Poses:** `T_camera_rig.json` / `T_lidar_rig.json` are **sensor→rig**, used *directly*
  (no inverse). World frame is **already ISO 8855** — no extra rotation. (Inverting either,
  or adding the ISO rotation, silently drops PSNR to ~20 and renders garbage — verified.)
- **LiDAR sweeps** `[N,6]=(x,y,z,intensity,t_offset_us,ring_id)` are in the **ego** frame.
- **CUDA arch:** gsplat + tiny-cuda-nn JIT/compile against `TORCH_CUDA_ARCH_LIST`
  (hard-coded Blackwell `12.0` in the sbatch files). **Set it from `nvidia-smi
  --query-gpu=compute_cap` on the new GPU and force-rebuild** — see REPRODUCE §4.
- Always `export PYTHONNOUSERSITE=1` (user-site numpy shadows the env otherwise).

See [`RESULTS.md`](RESULTS.md) for the full reproduction notes and per-clip numbers.
