# results/metrics — SUPERSEDED, kept on purpose

Every JSON in `nvidia/` and `clariden_2026-10/` was produced before the dataparser
fixes on branch `fix/camera-timing-lidar-frame-v2`. They are retained as a record of
the pre-fix pipeline. **They are not current results.**

Two verified defects were present in all of them:

* **F1** — every camera was given the LiDAR spin-centre timestamp instead of its own.
  The real offsets are clip-specific: up to **16.06 ms** per camera and **29.77 ms**
  between the two trigger groups. Worst measured ego displacement **65 cm** (NFF-14,
  19.43 ms at 120 km/h); 57 cm on NFF-01, 55 cm on NCF-01. The cameras were posed at
  the wrong instant, and dynamic-actor poses were interpolated to the wrong time.
* **F3** — sensor-frame LiDAR sweeps were transformed as if they were ego-frame, a
  **1.94 m** vertical offset: the world-frame ground sat at −1.94 m instead of 0.

Four smaller defects were also present (F2 rolling shutter disabled, F4 wrong class
vocabulary, F5 `stationary` hardcoded False, F6 64-beam LiDAR model for a 128-beam
sensor). See `../../DATASET_NOTES.md`.

Re-create the old regime deliberately with `jobs/regimes.sh` → `set_regime
legacy_prefix`, which writes to `legacy_prefix_SUPERSEDED/`. Corrected runs go to
`v2fixed/`. Do not write new metrics into this directory.
