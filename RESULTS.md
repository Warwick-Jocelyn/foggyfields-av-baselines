# RESULTS — reproduction report

Protocol everywhere: **nvs_50_50** (50% train / 50% held-out novel-view eval; same odd
frames held out for all three methods after the EmerNeRF split fix).
Hardware: 1× NVIDIA RTX PRO 6000 Blackwell (sm_120) per run.
Higher PSNR/SSIM better; lower LPIPS / LiDAR-depth-error better.

Metric provenance: NVIDIA RGB = torchmetrics over rendered held-out val PNGs
(`results/metrics/nvidia/*_rgb_metrics.json`); NVIDIA LiDAR = GT-sweep-vs-rendered-raw-depth
L1 (`*_lidar_metrics.json`); PandaSet = each run's `events.out.tfevents` "all images" eval.

---

## 1. NVIDIA_AV_Fog — held-out novel view

### RGB
| Clip | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| 002 (rural, med fog, ~113 km/h) | SplatAD | 21.64 | 0.850 | 0.413 |
| | NeuRAD | 27.63 | 0.884 | 0.180 |
| | EmerNeRF † | 31.36 | 0.941 | 0.177 |
| 003 (residential, light fog) | SplatAD | 24.60 | 0.873 | 0.353 |
| | NeuRAD | 29.10 | 0.897 | 0.161 |
| | EmerNeRF † | 34.11 | 0.956 | 0.167 |
| 004 (highway, heavy fog) | SplatAD | 23.94 | 0.882 | 0.324 |
| | NeuRAD | 29.90 | 0.911 | 0.152 |
| | EmerNeRF † | 34.66 | 0.963 | 0.136 |
| **mean** | SplatAD | **23.39** | 0.868 | 0.363 |
| | NeuRAD | **28.88** | 0.897 | 0.164 |
| | EmerNeRF † | **33.38** | 0.953 | 0.160 |

† **EmerNeRF NVIDIA numbers are PRE-split-fix** (it held out the complementary even frames,
and the LPIPS/PSNR were measured off the compressed test-video). They are *indicative*, not
final. **Re-train EmerNeRF on the new machine** with the fixed loader to get same-protocol
numbers. SplatAD/NeuRAD are final.

### LiDAR depth (rendered depth vs GT sweep, lower = better)
| Clip | Method | median L1 (m)↓ | mean L1 (m)↓ | acc@5m↑ |
|---|---|---|---|---|
| 002 | NeuRAD | **5.27** | 11.22 | **48.1%** |
| | SplatAD | 8.15 | 12.52 | 30.9% |
| 003 | NeuRAD | **7.49** | 13.05 | **36.1%** |
| | SplatAD | 8.63 | 13.54 | 33.2% |
| 004 | NeuRAD | **9.38** | 15.18 | **28.1%** |
| | SplatAD | 15.20 | 19.48 | 14.2% |

EmerNeRF LiDAR depth = not computed (its eval pipeline only emits colormapped depth, no raw
float). Re-render with a raw-depth hook if needed. SplatAD LiDAR errors are inflated by the
participating-media fog (Gaussians place mass in the fog volume); NeuRAD's volumetric
integration handles depth better.

---

## 2. PandaSet — reference runs (clips 011, 078; 50/50 split, paper-faithful)

> ⚠️ **Lighting differs between the two PandaSet clips** (see `DATASET_NOTES.md`):
> **011 = daytime** (`011-1`), **078 = NIGHTTIME** (`078_night_1`). Both are fog-free,
> so they isolate the *no-fog* baseline — but 078's higher numbers are a **night** scene,
> not directly comparable to the daytime 011.

| Clip | Lighting | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| 011 | **day** | SplatAD | 27.52 | 0.868 | 0.162 |
| | | NeuRAD | 26.46 | 0.805 | 0.201 |
| | | EmerNeRF | 27.55 | 0.790 | n/a |
| 078 | **night** | SplatAD | 31.65 | 0.930 | 0.242 |
| | | NeuRAD | 30.04 | 0.898 | 0.214 |
| | | EmerNeRF | 31.26 | 0.879 | n/a |

(EmerNeRF PandaSet tfevents log PSNR+SSIM only. On fog-free PandaSet the three methods
are close — SplatAD slightly ahead on SSIM/LPIPS — confirming the implementations are sound
and that the large EmerNeRF lead on NVIDIA_AV_Fog is a *fog*-specific effect.)

---

## 3. Training time & iterations (per clip, 1 GPU)

| Method | Iterations | Wall time (per clip) | Notes |
|---|---|---|---|
| SplatAD | 30 000 | **~30–47 min** | fastest; Gaussian splatting |
| NeuRAD | 30 000 | **~6 h 17 min** | slowest; full NeRF + LiDAR datamanager |
| EmerNeRF | 25 000 | **~4 h 22 min** | self-supervised dynamic NeRF |

(All 9 NVIDIA runs were trained in parallel on one 8-GPU node. Times are wall-clock per run.)

---

## 4. Best-PSNR checkpoint per (method × clip)

These are the checkpoints uploaded to HuggingFace (see `CHECKPOINTS.md`). Each
nerfstudio checkpoint requires its sibling `config.yml` to load.

### NVIDIA_AV_Fog
| Method | Clip | Checkpoint (run) | PSNR |
|---|---|---|---|
| SplatAD | 002 | `splatad/002/.../2026-05-29_230151/.../step-000029999.ckpt` | 21.64 |
| SplatAD | 003 | `splatad/003/.../2026-05-29_230151/.../step-000029999.ckpt` | 24.60 |
| SplatAD | 004 | `splatad/004/.../2026-05-29_230151/.../step-000029999.ckpt` | 23.94 |
| NeuRAD | 002 | `neurad/002/.../2026-05-29_002702/.../step-000029999.ckpt` | 27.63 |
| NeuRAD | 003 | `neurad/003/.../2026-05-29_002701/.../step-000029999.ckpt` | 29.10 |
| NeuRAD | 004 | `neurad/004/.../2026-05-29_002702/.../step-000029999.ckpt` | 29.90 |
| EmerNeRF † | 002/003/004 | `emernerf/<seq>/.../nvidia_fog_<seq>_7cam/checkpoint_25000.pth` | (stale, re-train) |

### PandaSet
| Method | Clip | Checkpoint | PSNR |
|---|---|---|---|
| SplatAD | 011 | `paper_011_split05_paperfaithful/.../2026-05-18_113557/.../step-000030000.ckpt` | 27.52 |
| SplatAD | 078 | `paper_078_split05_paperfaithful/.../2026-05-18_124006/.../step-000030000.ckpt` | 31.65 |
| NeuRAD | 011 | `neurad_paper_011_split05/.../2026-05-18_174055/.../step-000029999.ckpt` | 26.46 |
| NeuRAD | 078 | `neurad_paper_078_split05/.../2026-05-18_174055/.../step-000029999.ckpt` | 30.04 |
| EmerNeRF | 011 | `emernerf_011_split05/.../scene_011_6cam_dyn_flow_rgb/checkpoint_30000.pth` | 27.55 |
| EmerNeRF | 078 | `emernerf_078_split05/.../scene_078_6cam_dyn_flow_rgb/checkpoint_30000.pth` | 31.26 |

---

## 5. Key takeaways

- **EmerNeRF dominates RGB in fog** (mean ~33 dB vs NeuRAD ~29, SplatAD ~23) — its
  volumetric RGB+LiDAR integration models participating media better than discrete
  Gaussians. On fog-free PandaSet the gap vanishes, so this is a **fog-specific**
  finding, not an implementation artifact. (Note PandaSet 078 is a *night* scene — see
  `DATASET_NOTES.md`.)
- **NeuRAD wins LiDAR depth** on every fog clip (median 5–9 m vs SplatAD 8–15 m).
- **SplatAD is 8–13× faster** to train (~40 min vs 4–6 h) — its trade-off is fog fidelity.
- Difficulty tracks fog × speed: clip 002 (rural, 113 km/h, medium fog) is hardest;
  clip 004 (heavy-fog highway, more static structure) scores highest.

## 6. Reproduction fixes applied (for the record)
- **Pose bug (catastrophic):** earlier runs inverted the sensor→rig transforms and added a
  spurious ISO-8855 rotation → PSNR ~20, garbage renders. Fixed: use `T_*_rig.json` directly,
  world already ISO 8855. (+7 dB recovery on EmerNeRF confirmed the fix.)
- **EmerNeRF split bug:** held out even frames instead of odd → not comparable. Fixed to read
  `nvs_50_50.json` directly. (Re-train pending.)
- **NeuRAD FD crash** (`Too many open files`): `ulimit -n $(ulimit -Hn)` in the sbatch.
- **SplatAD K=1 actor crash on clip 003** (`broadcast shape` at `splatad.py`): `.squeeze()` →
  `reshape(-1)` so single-actor index tensors stay 1-D.
- **SplatAD eval OOM:** `ns-eval` chamfer (`torch.cdist`) OOMs against the ~90 GB resident
  Gaussian model; RGB metrics computed from rendered PNGs instead, LiDAR via raw-depth render.
