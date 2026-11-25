# NCF-mini crop vs mask ablation — how to run it

Compare the two ways of handling the black lens-cone corners of the rectification-V2
images, on 5 clear-weather NCF-mini clips, for SplatAD and NeuRAD.

| variant | meaning | `ns-train` dataparser flags |
|---|---|---|
| `safe` | crop to the `safe_crop` rectangle; no black borders, narrower vertical FOV | `--apply-safe-crop True --use-valid-masks False` |
| `mask` | full V2 canvas; black corners excluded from the RGB loss by `valid_mask` | `--apply-safe-crop False --use-valid-masks True` |

Matrix: 5 clips (`NCF-02 NCF-07 NCF-08 NCF-10 NCF-15`) x 2 variants x 2 methods = **20 runs**.

## The one rule this harness exists to enforce

**The two variants are not comparable on their native pixels.** A `mask` model leaves the
corners unconstrained (its unmasked PSNR is ~17–18 dB against a masked ~28), and scoring a
canvas that still contains black GT pixels inflates PSNR because black is a zero-cost
prediction. Either comparison gives the *opposite* of the right answer.

So **every primary number is computed on one identical pixel set: the `safe_crop`
rectangle**. `safe_crop` is by construction the largest rectangle with zero invalid
pixels, so `safe_crop ∩ valid_mask == safe_crop` — a fair common region for both variants.
The `safe` render already *is* that rectangle; the `mask` render is cropped to it first.
`eval_common_region.py` asserts the extracted region has exactly the `safe_crop` shape in
both cases, so an offset mistake fails loudly instead of producing a plausible number.

Measured on these 5 clips (`results/safe_crop_audit.json`, produced by the pre-flight):
`safe_crop` keeps **85.3–85.5 %** of the valid pixels on the three 120° cameras, 93.8 % on
the 70° cameras, 99.0 % on the 30° teles — **90.1 %** over all 7 cameras together. Zero
invalid pixels were found inside any `safe_crop`, on any camera of any clip, so the
common-region premise holds.

A secondary **wide** number (full canvas ∩ `valid_mask`) and an **outside-safe_crop**
number are reported for the `mask` variant only, to quantify what the discarded ~10 % of
valid pixels buys. Both are labelled NOT comparable to the primary number.

## Files

| file | what it does |
|---|---|
| `env.sh` | paths, caches (shared with `splatad-nff-repro` on purpose — that is where the sm_90 venv and the compiled extensions live), and every overridable knob |
| `run_matrix.sbatch` | the SLURM driver: pre-flight, then the 20-run matrix over 4 GPUs, resumable |
| `lane.sh` | one run: train → render held-out frames → common-region metrics → videos |
| `preflight.py` | CPU checks; fails before any GPU is taken (see below) |
| `selftest_eval.py` | CPU end-to-end proof that the two variants score *identically* on identical pixels |
| `clip_paths.py` | clip resolution + `safe_crop` geometry, cross-checked against the dataparser's own resolver |
| `eval_common_region.py` | the metric: primary = safe_crop region; secondary = wide / outside / native |
| `make_videos.py` | 7 per-camera side-by-side mp4s + a montage per run |
| `collect_results.py` | `results/SUMMARY.md`: per-clip paired tables, per-clip deltas, verdict |

Outputs land in `outputs/<method>/<variant>/<clip>/` (bulk, disposable),
`results/metrics/metrics_<method>__<variant>__<clip>.json` (keeper),
`viz/<method>/<variant>/<clip>/<camera>.mp4` (keeper), `logs/lanes/*.log`.

## What the pre-flight checks (before a single GPU is occupied)

1. every clip resolves, and **our resolver agrees with the dataparser's** `_resolve_clip_root`;
2. `images_rectv2/`, `masks_rectv2/`, `calib/K_rectified_v2.json` exist for all 7 cameras;
3. **zero invalid pixels inside `safe_crop`** on every camera of every clip (the premise);
4. the split file exists with the same held-out count on every clip;
5. the dataparser exposes both `apply_safe_crop` and `use_valid_masks`;
6. an mp4 encoder exists (there is no system `ffmpeg` here — PyAV/libx264 is used, with
   imageio-ffmpeg and cv2/mp4v as fallbacks);
7. the projected host-RAM image cache for 4 concurrent lanes fits in the node;
8. the metric self-test: two render trees built from the *same* pixels — one full-canvas,
   one pre-cropped — must come back with identical PSNR/SSIM/LPIPS, a canvas-sized render
   labelled `safe` must be rejected, and the unmasked-PSNR trap must reproduce;
9. a 10-iteration `ns-train` on **one** GPU for every method×variant combination.

## Launch

Smoke subset first (one clip, all four method×variant combinations, 500 iterations, the
first 15 % of the clip). Its numbers are **not results** — it only proves the plumbing:

```bash
cd /capstor/scratch/cscs/yitiwang/ncf-crop-ablation
sbatch --time=02:00:00 \
  --export=ALL,CLIPS=NCF-02,ITERS=500,END_FRACTION=0.15 \
  jobs/run_matrix.sbatch
```

Then the full matrix. The `normal` partition caps at 12 h and the matrix needs more, so
submit a chain — the job is resumable, so each later job just picks up what is left:

```bash
cd /capstor/scratch/cscs/yitiwang/ncf-crop-ablation
J1=$(sbatch --parsable jobs/run_matrix.sbatch)
J2=$(sbatch --parsable --dependency=afterany:$J1 jobs/run_matrix.sbatch)
sbatch          --dependency=afterany:$J2 jobs/run_matrix.sbatch
```

Watch it: `tail -f logs/matrix_<jobid>.log`, per-run detail in `logs/lanes/*.log`,
per-stage seconds in `results/timings.csv`, failures in `logs/FAILURES_<jobid>.txt`.

Rebuild the summary at any time (CPU, no job needed):

```bash
python jobs/collect_results.py --results results --clips "$CLIPS" --methods "splatad neurad"
```

### Useful overrides

| variable | default | note |
|---|---|---|
| `CLIPS` | the 5 NCF clips | space-separated |
| `METHODS` | `splatad neurad` | e.g. `METHODS=splatad` to run SplatAD alone first |
| `VARIANTS` | `safe mask` | |
| `SPLATAD_ITERS` / `NEURAD_ITERS` | 30000 / 20000 | per-method |
| `ITERS` | *(unset)* | overrides BOTH — for smoke runs. The LR schedules are **not** rescaled, so a short run is a plumbing test, not a cheap result |
| `END_FRACTION` | *(unset)* | use only the first fraction of each clip (smoke) |
| `MAX_EVAL_FRAMES` | *(unset)* | cap held-out timesteps; seed 123, so identical for both variants |
| `NLANES` | 4 | one GPU per lane |
| `FORCE` | 0 | 1 = redo finished stages |
| `SPLATAD_EXTRA` / `NEURAD_EXTRA` | empty | appended verbatim to `ns-train` |
| `PANEL_W` | 960 | per-panel width of the side-by-side videos |

## Expected cost

Anchors: SplatAD 30k on the V1 rectification (8.6 Mpx per timestep over 7 cameras) took
27–45 min per clip on one GH200, render+metrics ~9 min. V2 is **2.3x** those pixels for
`safe` (19.9 Mpx/timestep) and **2.95x** for `mask` (25.4 Mpx/timestep), and NeuRAD is
slower than SplatAD, so:

| | per run | 20-run matrix |
|---|---|---|
| SplatAD train | 1.5–3 h | |
| SplatAD render + metrics + video | 0.5–1 h | |
| NeuRAD train | 1.5–2.5 h | |
| NeuRAD render + metrics + video | 1.5–3 h | ← the biggest unknown |
| **total** | | **≈ 75 GPU-h (range 50–100)** |
| **wall clock on 4 GPUs** | | **≈ 19 h (range 13–25 h) → 2–3 submissions of the 12 h job** |

Smoke subset: ~25 min of pre-flight (CPU checks + self-test + four 10-iteration trains)
plus ~10–25 min per run, four runs in parallel → **≈ 45–60 min wall, ≈ 3 GPU-h**.

Run the smoke first and read `results/timings.csv`: it is the only honest input to the
full-matrix estimate, because nothing on this cluster has yet trained either method at V2
resolution.

Disk: the PNG renders dominate — about 4–7 GiB per run, **~120 GiB** for the full matrix,
plus checkpoints. Scratch has 150 TiB free, so this is not a constraint; inodes are
~1,400 files per run, ~28k total.

## What to delete afterwards

Keep (`★`), review then keep (`☆`), delete (`–`):

```
★ results/          metrics JSON, SUMMARY.md, summary.json, timings.csv, safe_crop_audit.json   (~2 MB)
★ jobs/             these scripts
☆ viz/              140 mp4 + 20 montages (~2-6 GB) — pick the ones worth keeping, then mirror
☆ logs/             small; keep logs/lanes/*.log with the runs you report
–  outputs/*/*/*/renders/          the PNG dumps, ~120 GB — delete once metrics + videos exist
–  outputs/*/*/*/*/*/nerfstudio_models/*.ckpt   only needed to re-render without retraining
–  outputs/_preflight, outputs/_queue_*         transient
```

```bash
cd /capstor/scratch/cscs/yitiwang/ncf-crop-ablation
du -sh outputs results viz logs
rm -rf outputs/*/*/*/renders            # the big one; metrics + videos are already derived
# then mirror the keepers to the durable tier
STORE=/capstor/store/cscs/swissai/a0195/yitiwang/ncf-crop-ablation
mkdir -p "$STORE" && rsync -a results jobs viz "$STORE/"
```

The workspace is on `/capstor/scratch`, which is purged on a 30-day atime policy — nothing
here survives on its own.

## Known risks to check before launching

1. **NeuRAD + `mask` may not train at all.** NeuRAD's datamanager uses
   `ScaledPatchSampler`, whose `_check_extra_keys` raises
   `NotImplementedError("Patch sampler not implemented for extra_keys")` for any batch key
   other than `image`/`image_idx`. `--use-valid-masks True` adds a `mask` key, so unless
   the V2 branch teaches that sampler to honour (or drop) masks, `neurad/mask` dies at step
   0. The pre-flight's 10-iteration train catches it in seconds; if it does, either fix the
   sampler or run `METHODS=splatad` first.
2. **Host RAM.** nerfstudio caches every train *and* eval image. SplatAD caches uint8
   (~14 GiB per `mask` run); NeuRAD caches float32 (~57 GiB per `mask` run). Four NeuRAD
   lanes ≈ 226 GiB on an ~856 GiB node — it fits, but the pre-flight re-checks on the real
   node and tells you the exact `NEURAD_EXTRA` flags to cap the cache if it does not.
3. **Short runs are not cheap results.** `ITERS` does not rescale the LR schedules or
   SplatAD's densification schedule.
