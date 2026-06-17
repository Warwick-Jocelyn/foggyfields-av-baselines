# REPRODUCE — step by step

Target: a SLURM cluster (or single box) with an NVIDIA GPU and CUDA ≥ 12.1.
Everything below assumes `bash`. Replace `$DATA_ROOT` / `$OUT` / arch with your values.

```bash
# pick your locations
export DATA_ROOT=/path/to/NVIDIA_AV_Fog        # processed dataset root (has processed/<clip>/processed_pinhole)
export OUT=/path/to/NVIDIA-Fog-Output          # where runs/renders/metrics go
export PYTHONNOUSERSITE=1                       # ALWAYS — user-site pkgs shadow the env
```

---

## 1. Get the code (upstreams + our patches)

`scripts/setup_new_machine.sh` does this in one shot, or manually:

```bash
# commits are in BASE_COMMITS.txt
git clone https://github.com/georghess/neurad-studio.git && \
  git -C neurad-studio checkout e6f7e4e509b828a952d8584b7165f7844711ecb2
git clone https://github.com/carlinds/splatad.git && \
  git -C splatad checkout 6e31ad766d39e0c33f9034a2ed772d51364b2343
git clone https://github.com/NVlabs/EmerNeRF.git && \
  git -C EmerNeRF checkout 8c051d7cccbad3b52c7b11a519c971b8ead97e1a

# apply our changes
git -C neurad-studio apply /path/to/this-repo/patches/neurad-studio.patch
git -C EmerNeRF       apply /path/to/this-repo/patches/emernerf.patch
```

If a patch fails to apply (e.g. upstream re-tagged), fall back to the plain files in
`src/` — copy them over the cloned tree at the same relative paths. The patch and the
`src/` copies are byte-identical; either reproduces our state.

`splatad` (the gsplat CUDA backend) has **no** local diff — it is pristine upstream at the
pinned commit; just clone it.

---

## 2. Build the environment

Two supported paths. **Apptainer is recommended for SLURM** (no root/daemon needed);
conda works fine on a single box you control.

### 2a. conda
```bash
conda env create -f env/blackwell_conda_20260517_2215.yaml   # or conda_env_20260517.yaml
conda activate splatad            # (env was named splatad-blackwell on the origin box; rename as you like)
pip install -r env/blackwell_requirements_20260517_2215.txt   # exact pins
# then install the three repos editable:
pip install -e ./neurad-studio    # provides ns-train / ns-render / ns-eval (SplatAD + NeuRAD)
pip install -e ./splatad          # gsplat CUDA backend
# EmerNeRF runs from its own dir (python train_emernerf.py), no install needed
```
See `env/SETUP.md` for the Blackwell-specific build notes (the `bin/nvcc` wrapper trick for
the cu128 toolchain).

### 2b. Apptainer (recommended on a cluster)
Build an image whose `%post` runs the conda create above, then do the arch rebuild (§4)
at first run. (`env/SETUP.md` documents the toolchain; turn it into an `apptainer.def`
`%post` and expose `TORCH_CUDA_ARCH_LIST` as a build arg.)

---

## 3. Get data + checkpoints

```bash
# dataset (gated HF dataset — accept the license first; see CHECKPOINTS.md)
hf download <user>/nvidia-av-fog-processed --repo-type dataset --local-dir "$DATA_ROOT"

# checkpoints (only if you want to skip training)
hf download <user>/nvidia-av-fog-baseline-checkpoints --repo-type model --local-dir "$OUT/_ckpts"
```
PandaSet: get it from the official source and point the pandaset dataparser at it.

---

## 4. ★ Rebuild CUDA extensions for YOUR GPU arch (the one real gotcha)

The sbatch files hard-code Blackwell `TORCH_CUDA_ARCH_LIST="12.0"`. gsplat and
tiny-cuda-nn compile kernels for that arch; on a different GPU they will fail or silently
mis-run. Fix it **before** training/eval:

```bash
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)   # e.g. 8.9 (Ada), 8.6 (Ampere), 12.0 (Blackwell)
export TORCH_CUDA_ARCH_LIST="$ARCH"
export TCNN_CUDA_ARCHITECTURES="${ARCH/./}"        # same, no dot (e.g. 89)
export CUDA_HOME=$CONDA_PREFIX
export PYTHONNOUSERSITE=1

rm -rf ~/.cache/torch_extensions/*/gsplat_cuda     # gsplat JIT-compiles on first run
pip install --no-build-isolation --force-reinstall \
  "git+https://github.com/NVlabs/tiny-cuda-nn/@749dd70#subdirectory=bindings/torch"
```
Then **edit the `TORCH_CUDA_ARCH_LIST="12.0"` line in each `jobs/*.sbatch` to `$ARCH`**
(or `export` it before the script — the sbatch `export` will otherwise override it).

If the target only has an older CUDA driver (e.g. 11.8), also swap the torch wheel
(`+cu118`) and rebuild the env from the yaml rather than reusing a Blackwell image.

---

## 5. Parameterize machine-specific values

Every `jobs/*.sbatch` hard-codes origin-box values. Before running elsewhere, set/replace:

| Variable | Origin value | Where |
|---|---|---|
| `#SBATCH --partition` | `test` | every sbatch line ~3 |
| `#SBATCH --gres` | `gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1` | train sbatch line ~8 |
| `#SBATCH --output` log path | `/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output/_logs/...` | line ~10 |
| conda init | `source ~/miniconda3/etc/profile.d/conda.sh` | ~line 16-21 |
| conda env | `splatad-blackwell` | ~line 17-22 |
| `TORCH_CUDA_ARCH_LIST` | `"12.0"` | ~line 21-26 |
| output root | `/networkhome/.../NVIDIA-Fog-Output/<method>/<seq>` | ~line 29-33 |
| EmerNeRF `data_root` | `/networkhome/.../NVIDIA_AV_Fog` | `src/emernerf/configs/nvidia_av_fog_lidar.yaml:5` |
| wandb `--entity` | `haonan_zhao` | `nvidia_fog_emernerf.sbatch` (make optional) |

Tip: a quick `sed -i "s#/networkhome/WMGDS/wang3_y/ETH#$ROOT#g; s/partition=test/partition=$PART/g" jobs/*.sbatch`
handles the bulk.

---

## 6. Train (full reproduce)

Protocol: **nvs_50_50** (50% train / 50% held-out). Iterations: SplatAD 30k · NeuRAD 30k ·
EmerNeRF 25k. Seed: nerfstudio default (pin with `--machine.seed 42`). 1 GPU per run.

```bash
# --- SplatAD (neurad-studio) ---
for SEQ in 002 003 004; do
  ns-train splatad --output-dir "$OUT/splatad/$SEQ" --max-num-iterations 30000 --machine.seed 42 \
    --steps-per-eval-image 999999 --steps-per-eval-all-images 999999 \
    nvidia-av-fog-data --data "$DATA_ROOT" --sequence "$SEQ" --splits-json nvs_50_50
done

# --- NeuRAD (same repo) ---  (ulimit raise avoids the 'too many open files' FD crash)
ulimit -n "$(ulimit -Hn)"
for SEQ in 002 003 004; do
  ns-train neurad-paper --output-dir "$OUT/neurad/$SEQ" --max-num-iterations 30000 --machine.seed 42 \
    --steps-per-eval-all-images 999999 \
    nvidia-av-fog-data --data "$DATA_ROOT" --sequence "$SEQ" --splits-json nvs_50_50
done

# --- EmerNeRF (own trainer; reads nvs_50_50.json via the FIXED loader) ---
cd EmerNeRF
for SEQ in 002 003 004; do
  python train_emernerf.py --config_file configs/nvidia_av_fog_lidar.yaml \
    --output_root "$OUT/emernerf/$SEQ" --run_name "nvidia_fog_${SEQ}_7cam" \
    data.data_root="$DATA_ROOT" data.scene_idx="$SEQ" optim.num_iters=25000
done
```
Or just submit the ready drivers: `sbatch --export=ALL,SEQ=002 jobs/nvidia_fog_splatad.sbatch`
(these also render + compute metrics in later steps).

PandaSet: use `jobs/paperfaithful_blackwell.sbatch` / `jobs/neurad_paper_blackwell.sbatch` /
`jobs/emernerf_blackwell.sbatch` with the `split05` (50/50) configs.

---

## 7. Render the 7-camera videos + metrics (from a checkpoint)

```bash
# SplatAD/NeuRAD: render val frames then stitch the 3x3 panorama
ns-render dataset --load-config "$OUT/splatad/002/.../config.yml" \
  --output-path "$OUT/splatad/002/renders" --rendered-output-names rgb gt-rgb depth --pose-source val
python jobs/pngs_to_mp4_nvidia_fog.py "$OUT/splatad/002/renders" "$OUT/splatad/002/videos/val" \
  --split val --fps 10 --panorama-only           # ★ 7-cam panorama_rgb.mp4

# RGB metrics (PSNR/SSIM/LPIPS over rendered val PNGs)
python jobs/compute_rgb_metrics.py --pred "$OUT/splatad/002/renders/val/rgb" \
  --gt "$OUT/splatad/002/renders/val/gt-rgb" --output "$OUT/splatad/002/rgb_metrics.json"

# LiDAR depth metrics (needs a raw-depth re-render first)
sbatch --export=ALL,METHOD=splatad,SEQ=002 jobs/nvidia_fog_rerender_rawdepth.sbatch
python jobs/compute_lidar_metrics.py --method splatad --seq 002 --output "$OUT/splatad/002/lidar_metrics.json"
```
EmerNeRF emits its own `test_videos/*.mp4` during eval (config `render.render_test: true`);
LPIPS for it via `jobs/compute_lpips_emernerf.py`.

---

## 8. Verify

Compare your numbers against `results/metrics/nvidia/*.json` and `RESULTS.md`. PSNR should
land within ~0.3 dB (SplatAD/NeuRAD). EmerNeRF will differ from the *old* table because the
split was fixed — that's expected; it's now on the same held-out frames as the others.
