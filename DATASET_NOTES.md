# Dataset clip notes — lighting / weather / naming (READ BEFORE COMPARING CLIPS)

The clip folder names encode scene attributes. **Some clips are captured at NIGHT** —
this matters when interpreting/comparing per-clip metrics, because lighting differs.

## NVIDIA_AV_Fog (the fog benchmark)
All benchmark clips are **fog + DAYTIME** (folder names `..._fog_day_...`):

| Clip | Folder | Weather | Lighting | Scene |
|---|---|---|---|---|
| 002 | `002_..._fog_day_rural` | fog (medium) | day | rural, ~113 km/h |
| 003 | `003_..._fog_day_residual` | fog (light) | day | residential |
| 004 | `004_..._fog_day_highway_with_cars` | fog (heavy) | day | highway |
| 006 | `006_..._fog_day_residence` | fog | day | residence (available, not in benchmark) |
| 008 | `008_..._fog_day_rural` | fog | day | rural (available, not in benchmark) |

→ No nighttime clips in the NVIDIA_AV_Fog set used here.

## PandaSet (fog-free reference) — ⚠️ contains NIGHT clips
The PandaSet clip folders were **renamed to mark lighting/scene** (e.g. `_night`, `_night_glare`).
Our two benchmark clips differ in lighting:

| Clip | Folder | Lighting | Note |
|---|---|---|---|
| **011** | `011-1` | **day** | daytime reference |
| **078** | `078_night_1` | **NIGHT** | nighttime — higher PSNR partly reflects the night scene, **not** directly comparable to 011 |

**Full list of NIGHTTIME PandaSet clips** (renamed locally): `057_night`, `059_night_difficult`,
`062_night_padestrains`, `068_night_glare`, `070_night_galre_noisy`, `072_night`, `073_night`,
`074_night_padestrains`, `077_night`, `078_night_1`, `149_night`.

## Why this reminder exists
PandaSet is fog-free, so it serves as the "no-fog" baseline that isolates the fog-specific
behaviour seen on NVIDIA_AV_Fog. But **clip 078 is a night scene** — keep that in mind:
- when comparing 011 (day) vs 078 (night) numbers, lighting is a confound;
- if you later add more PandaSet clips, check the folder name for a `_night` tag first.

> The processed datasets themselves are **not** in this repo or on HuggingFace
> (NVIDIA_AV_Fog stays local / regenerable from NVIDIA PhysicalAI-AV; PandaSet is public).
> Only this naming/lighting metadata is tracked here as a reminder.

---

# Dataparser defects found 2026-10-19 (F1–F6) — read before using any old number

All six were verified against the clips on disk under
`Dataset/physicalai_av/{NVIDIA_CLEAR_Field,NVIDIA_FOGGY_Field}/clips/`. They are fixed
on branch `fix/camera-timing-lidar-frame-v2`. Magnitudes below are measured, not
estimated.

## F1 — camera timestamps (most severe)

`_get_cameras` gave **every** camera the LiDAR spin-centre timestamp. The seven cameras
are triggered in **two groups**:

* group A — `front_wide`, `rear_left`, `rear_right`, `front_tele`
* group B — `cross_left`, `cross_right`, `rear_tele`

Median offset from `spin_center_us` (group A / group B):

| clip | group A | group B | gap |
|---|---|---|---|
| NFF-01 | +2.74 ms | −14.17 ms | 16.91 ms |
| NFF-07 | +5.59 ms | −11.31 ms | 16.91 ms |
| NCF-02 | −4.71 ms | −15.75 ms | 11.04 ms |
| NCF-08 | −0.28 ms | −2.31 ms | 2.04 ms |

Intra-group jitter is ±0.1 ms, so this is a fixed per-group bias. The worst gap,
16.91 ms, is **15.5 cm** of ego motion at 32.9 km/h and ~23 cm at 49 km/h. It also
corrupts dynamic-actor pose interpolation, which is keyed on time.

Real timestamps: `calib/camera_timestamps.parquet` (`timestep_idx` + one column per
camera, plus `<cam>_src_frame`), or `videos/<cam>.timestamps.parquet` column
`timestamp` matched to each spin centre by `argmin |t_cam − spin_center_us|`.
`calib/camera_timestamps_offsets.json` states the same offsets independently
(NCF-08: −0.279 / −2.318 ms, `group_spread_ms` 2.041) — our loader reproduces them.

## F2 — rolling shutter IS published

`calib/camera_intrinsics_ftheta.json` carries `"shutter_type":
"ROLLING_TOP_TO_BOTTOM"` and `"shutter_delay_us": 30559` for all 7 cameras in all 32
clips (224/224 entries, identical). The old config set `rolling_shutter_time = 0.0`
with a comment claiming the values were "not published". 30.559 ms of readout inside a
33.331 ms frame period.

`time_to_center_pixel` stays 0. `calib/rs_time_offset/README.json` states the per-pixel
offsets are "seconds, relative to the frame exposure timestamp in
calib/camera_timestamps.parquet (**mid-frame readout**)" and derives them as
`(row/src_h − 0.5) · shutter_delay_us`, symmetric about 0 (±15.279 ms). That is the
dataset toolkit's assertion about NVIDIA's timestamps, not NVIDIA's own — treat it as a
documented assumption. If it is ever contradicted, the start-of-frame value would be
`shutter_delay_us/2 = 0.0152795 s` and the error from 0 would be 15.3 ms.

## F3 — LiDAR frame is per clip and is changing under you

`lidar/sweep_*.npy` xyz is in the **sensor** frame on pre-fix clips and in the **ego**
frame once converted; `provenance.lidar_frame` marks it. Measured near-ground z (p05):
NFF-01 −2.34, NFF-02 −1.97, NFF-07 −2.66 against a lidar mount height of +1.94 m.
Loading a sensor-frame sweep as "ego" applies `inv(T_lidar_rig)` a second time and
drops the world-frame ground to **−1.94 m**.

The clear set is already converted and marked `"ego"`; the fog set is being converted
clip by clip, so the same directory tree currently contains both states. Never decide
this from a config default — read the marker. An unmarked clip requires an explicit
opt-in flag.

## F4 — the real class vocabulary

Counted over every box of all 15 fog + 17 clear clips:

| class | fog | clear | | class | fog | clear |
|---|---|---|---|---|---|---|
| automobile | 22710 | 103684 | | train_or_tram_car | 0 | 755 |
| person | 93 | 53585 | | stroller | 0 | 724 |
| heavy_truck | 3763 | 1844 | | animal | 856 | 31 |
| rider | 50 | 1827 | | protruding_object | 64 | 325 |
| trailer | 1256 | 1416 | | other_vehicle | 521 | 206 |
| bus | 217 | 1055 | | | | |

The old rigid whitelist listed `motorcycle`/`bicycle`/`train`/`trolley`/`tram`, none of
which NCore emits. The deformable list was `pedestrian`/`person`/`two_wheeler`:
`pedestrian` and `two_wheeler` do not exist and **`rider` — NCore's cyclist/motorcyclist
class — was missing**. `animal` and `protruding_object` were in neither list and were
dropped silently, as were `train_or_tram_car` and `stroller`. With
`include_deformable_actors=False` (the old default) `person` and `rider` went too:
~3.6% of fog boxes, but NCF-08 alone has **19,396 person boxes**.

Current assignment: rigid+symmetric = automobile, heavy_truck, bus, trailer,
other_vehicle, train_or_tram_car; rigid+asymmetric = stroller; deformable = person,
rider, animal; static object (not an actor by default) = protruding_object.
`class_taxonomy.json` in each clip carries NCore's own nuScenes/COCO mapping.

## F5 — `stationary` was hardcoded False

`obstacles_3d_world.parquet` has `velocity_x_mps` / `velocity_y_mps`. At a 0.5 m/s peak
speed threshold: **4 of 704** fog tracks (0.6%) and **30 of 3316** clear tracks (0.9%)
are stationary. NCF-08 alone contributes 16 of the clear 30. Small, but free to compute.

## F6 — 128 beams, not 64

The top LiDAR is a 128-beam row-offset spinning unit. `calib/lidar_intrinsics.json`
publishes it per clip (per-vehicle calibration): `n_rows` 128, `n_columns` 3600,
`spinning_frequency_hz` 10.0, `row_elevations_rad[128]` (−25.02…+14.45° on NCF-08),
`column_azimuths_rad[3600]` (0.1° step), `row_azimuth_offsets_rad[128]`. The parser
declared `LidarType.PANDAR64`, so the datamanager rasterised onto Pandar64's 64 static
elevations — which drives the ray-drop model. Beam divergence genuinely is not
published; the Pandar64 values remain as a stated assumption.

`lidar/range_image/` (128 × 3600 with explicit non-returns) is being generated and will
replace the inferred missing rays. It does not exist in any clip yet.

## Verified correct — do NOT "fix" these

* LiDAR sweep column 4 is a **relative** per-point offset, measured −50006…+50008 µs
  (half a 10 Hz revolution either side of the spin centre). The dataset's
  `CONVENTIONS.md` claims it is absolute; the doc is what is wrong.
* `T_camera_rig.json` / `T_lidar_rig.json` are **rig ← sensor** and are used directly.
* The world frame is already ISO 8855. The `[[0,1,0],[1,0,0],[0,0,1]]` "ISO remap" that
  used to sit in both loaders has determinant **−1** — a reflection, not a rotation. It
  has been deleted from both.
* `eval_utils.py` `weights_only=False` is correct.
* Moving the chamfer `cdist` to CPU is correct; "numerically identical" was not —
  `use_mm_for_euclid_dist` differs between CPU and CUDA, so they agree only to
  round-off.

## Rectification v2

`calib/K_rectified_v2.json` is **per camera** with different fx/fy/w/h. Read the
`safe_crop` block (x, y, w, h, cx, cy), not the top-level w/h, or black rectification
corners enter the loss: fx/fy are unchanged by the crop, cx/cy come from inside it.
Resolution rises from 1280×960 to up to 2692×1880 (4.1× the pixels — re-check memory
and batching). 2-D labels move to `labels_rectv2/`. Valid masks default to ON because
the v2 valid region is not rectangular. See `NVIDIA_CLEAR_Field/RECTIFY_V2.md`.

`calib/rs_time_offset/<cam>.npy` stores per-output-pixel readout time at 1/8 resolution
on the **full canvas** (not the safe crop), bilinear-upsampled at load. Rectification
smears one v2 output row across a median of ~96 source rows (max 177) of 1080 on the
120° cameras = 2.7 ms (max 5.0), ~35 px on the 70° cameras, ~6 px on the teles. Real,
but **smaller than F1's 16.9 ms** — F1 matters more.
