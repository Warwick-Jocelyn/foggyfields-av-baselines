# Clariden (CSCS Alps, GH200 / sm_90) reproduction of SplatAD on NFF

Everything here runs inside the NGC PyTorch 24.08 container via the CSCS Container Engine
(`splatad-nff-repro.toml`, copy to `~/infra/projects/`). Workspace on 30-day scratch:
`/capstor/scratch/cscs/yitiwang/splatad-nff-repro/` (`W`).

## Data: NFF HDF5 -> the layout this repo's dataparser expects
The durable NFF copy is 15 per-clip HDF5 files on store. Extract one clip, then expose it under the
old `processed/<seq>_<uuid8>_.../processed_pinhole` layout with a symlink — no loader change needed:
```bash
python ~/infra/scripts/nff_extract.py ~/store/datasets/NFF/NFF-03__5c5006b6.hdf5 --all -o $W/data/NFF/clips/NFF-03__5c5006b6
mkdir -p $W/data/NVIDIA_AV_Fog/processed/004_5c5006b6_NFF-03_fog_day_highway_with_cars
ln -s ../../../NFF/clips/NFF-03__5c5006b6 $W/data/NVIDIA_AV_Fog/processed/004_5c5006b6_NFF-03_fog_day_highway_with_cars/processed_pinhole
```
Clip-ID map (the HF checkpoint names use the OLD ids): old 002 = NFF-01, **old 003 = NFF-02**, **old 004 = NFF-03**, old 006 = NFF-04.

## Scripts
| file | what |
|---|---|
| `env.sh` | shared env: sm_90 arch flags, caches on scratch, venv activation, pip index override |
| `build_env.sbatch` | debug partition: virtualenv on scratch, neurad-studio + deps, nerfacc/tcnn/gsplat built for sm_90, import auto-fix, dataparser CPU check, 60-iter smoke |
| `import_fix.py` / `dataparser_check.py` | helpers used by the build |
| `train_all.sbatch` | normal partition, 1 node: 4 lanes (faithful × 2 clips, corrected × 2 clips) then 2 HF-checkpoint evals, collect, mirror to store |
| `lane.sh` | one run: ns-train -> ns-render (rgb/gt/depth + raw-depth) -> metrics -> panorama video |
| `prep_hf_config.py` | rewrite a published `config.yml` (paths, no wandb) so `ns-render/ns-eval` can load the HF checkpoint here |
| `rgb_metrics.py` | PSNR/SSIM/LPIPS, unmasked (= original script) and masked by `masks/<cam>/valid_mask.png` |
| `lidar_metrics.py` | LiDAR depth error with the validated conventions (`correct`) and the original script's math (`legacy`, wrong) side by side |
| `collect_results.py` | `RESULTS.md` with published / June-2026 / new numbers side by side |

## Dataparser changes (both default to the ORIGINAL behaviour; the `corrected` variant turns them on)
- `use_valid_masks` — attach the per-camera `valid_mask.png` as `mask_filenames`, so the black ftheta->pinhole
  corners (18–25 % of pixels) are excluded from the RGB loss (SplatAD already honours `batch["mask"]`).
- `lidar_points_frame` — `"ego"` (original: points mapped with `inv(T_lidar_rig)`) or `"sensor"` (use as-is).
  Validated 2026-10-12 on the packaged NFF release: sweeps are stored in the LIDAR SENSOR frame (raw ground plane
  at z≈−2.0 m = minus the 1.94 m mount height; 35k vs 34 points land inside the 3D boxes). The original setting
  therefore offsets all LiDAR geometry by (−1.14, −0.01, −1.94) m relative to the cameras.

## Metric-script findings (validator, 2026-10-12)
- `calib/T_camera_rig.json` is the camera pose in the rig (rig←cam). The dataparser uses it correctly
  (`c2w = T_rig_world @ T_camera_rig`). The original `jobs/compute_lidar_metrics.py` used it the other way round
  AND treated sweeps as ego-frame, so **its published LiDAR depth numbers are not meaningful**. `lidar_metrics.py`
  replaces it.
- `splits_json: auto` (per-sensor linspace, fraction 0.5) is exactly `nvs_50_50`: even timesteps train, odd eval.

## Gotchas that cost time
- `python -m venv` has no ensurepip in the image -> `virtualenv --system-site-packages`.
- The image's pip extra index `pypi.ngc.nvidia.com` does not resolve from compute nodes -> `PIP_EXTRA_INDEX_URL=https://pypi.org/simple`.
- numpy must stay `<2` (container torch); install `numpy==1.26.4` last. CUDA extensions need `--no-build-isolation`.
- `sbatch --environment=` works but warns; keep `--account=a0195`, `--gres=gpu:N`.
- The gsplat fork JIT-compiles and needs its `glm` git submodule (`git submodule update --init --recursive`), else `fatal error: third_party/glm/glm/glm.hpp`.
- `zod` pulls a too-new `dataclass-wizard`; pin `dataclass-wizard==0.22.3`.
