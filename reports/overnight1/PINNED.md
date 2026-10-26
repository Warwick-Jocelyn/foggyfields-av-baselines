# Vendored LiDAR fog simulators (FoggyFields T7)

Cloned 2026-10-26 on the Clariden login node (aarch64). Treated as untrusted:
code was read before anything was run; scripts are executed from outside the
clone dirs with `python3 -I` and paths as arguments.

| Repo | URL | Pinned commit | Last upstream commit |
|---|---|---|---|
| LiDAR_fog_sim | https://github.com/MartinHahner/LiDAR_fog_sim | `1cb9e072976047a7d11f859797d939676d37d88f` | 2025-08-16 "Replace PWC with DeepWiki badge" |
| LISA | https://github.com/velatkilic/LISA | `ce06a7fbf2d0c9c4ad35d9dad55a79dfa8d5d7d4` | 2023-08-14 "cross compile" |

Local modification to the LISA clone (recorded, intentional):
`pylisa/mie.so` shipped as x86-64; recompiled for aarch64 with
`gcc -std=c99 -fPIC -O2 -shared mie.c -o mie.so -lm` (mie.c read first: plain
Bohren–Huffman Mie translation, no I/O). Original binary kept as
`pylisa/mie.so.x86_64.orig`. `git status` of the LISA clone is dirty by
exactly these two files.

Adapter: `/capstor/scratch/cscs/yitiwang/foggyfields/medium/foggyfields/fogsim_adapter.py`
(`fog_sweep_hahner`, `fog_sweep_lisa`, identical signatures). Smoke test:
`third_party/smoke_fogsim.py` (results at the bottom).

---

## 1. Hahner et al., ICCV 2021 (LiDAR_fog_sim)

**Entry point.** `fog_simulation.simulate_fog(p: ParameterSet, pc: np.ndarray,
noise: int, gain: bool = False, noise_variant: str = 'v1', hard: bool = True,
soft: bool = True) -> (augmented_pc, simulated_fog_pc, info_dict)`
(repo root is the module dir; `fog_simulation.py`). Their own `__main__` uses
`simulate_fog(ParameterSet(alpha=a, gamma=1e-6), points, noise=10)`.

**Input.** `pc` is `[N, >=4]` float; columns 0:3 xyz in meters in the SENSOR
frame (range is computed as `||xyz||` from the origin), column 3 intensity in
**0..255** (the code rounds it to integers, caps fog responses at 255, and the
optional `gain` rescales to a 255 max). Columns beyond 4 are NOT preserved for
scattered points (`augmented_pc` is zero-initialised and only col 4 is copied),
so the adapter passes `[N, 4]` and reattaches metadata by index.

**Extinction parameter.** `ParameterSet.alpha` in 1/m == our `beta_true`.
Two gotchas, both handled by the adapter:
1. `ParameterSet.__init__` derives `mor = ln(20)/alpha` and the backscatter
   coefficient `beta = 0.046/mor` (1/sr) from the DEFAULT `alpha = 0.06`
   BEFORE `kwargs` are applied, so `ParameterSet(alpha=0.03)` silently keeps
   `mor`/`beta` of alpha=0.06 (verified at runtime). The adapter passes
   consistent values explicitly: `ParameterSet(alpha=beta, mor=ln(20)/beta,
   beta=0.046*beta/ln(20), gamma=1e-6)`.
2. The soft-scatter path reads a precomputed pulse-integral lookup table that
   exists only for `alpha in {0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.12, 0.15,
   0.2}` (`integral_lookup_tables/original/*.pickle`, tau_h = 20 ns);
   `get_integral_dict` snaps to the NEAREST available alpha. The adapter warns
   when the snap is inexact. Hard attenuation always uses the exact beta.

**Other parameters (defaults used).** pulse half-width `tau_h = 2e-8 s` (only
20 ns tables shipped), peak power `p_0 = 80 W`, receiver aperture/loss
`a_r = 0.25`, `l_r = 0.05`, hard-target reflectivity `gamma = 1e-6`
(=> `beta_0 = gamma/pi`), `noise = 10` m uniform range noise on scattered
points (variant 'v1'), `r_1 = 0.9 m`, `r_2 = 1.0 m` FOV overlap ranges.
Randomness: module-global `fog_simulation.RNG = np.random.default_rng(42)`;
the adapter reassigns it from `rng_seed`.

**Returns.** `augmented_pc` `[N, 4]`, same N and order (NO points dropped):
hard path `I' = round(I * exp(-2*alpha*r))`; soft path moves a point to the
fog-response range (xyz scaled by `r'/r`, so ray direction is kept) with the
fog response as its new intensity when that response exceeds the attenuated
hard return. Also returns the scattered subset and an info dict (unused).

**Deps.** numpy, scipy (`scipy.constants.speed_of_light`), tqdm (module level
import), pickle lookup tables. All import on the login `python3 -I`
(miniforge numpy 2.5.0 / scipy 1.18.0); no container needed. The alpha=0.03
pickle was opcode-scanned with `pickletools` before loading: only
numpy `dtype`/`scalar` reconstructors, no arbitrary globals.

## 2. LISA, Kilic et al. 2021

**Entry points.** `pylisa.lisa.Lisa(m=1.328, lam=905, rmax=200, rmin=1.5,
bdiv=3e-3, dst=0.05, dR=0.09, atm_model=..., mode='strongest')` with
- `Lisa.augment(pc)` -> for fog models (`'strong_advection_fog'`,
  `'moderate_advection_fog'`, `'chu_hogg_fog'`) this is `augment_avg(pc)`:
  average-effects only — attenuates `I' = I*exp(-2*alpha*r)`, adds range
  jitter `sigma = dR/sqrt(2*snr)`, zeroes points with received power below
  `Pmin = 0.9/rmax^2`. It NEVER creates near-range backscatter points.
- `Lisa.haze_point_cloud(pts_3D)` -> the fog model with discrete backscatter
  (ported into LISA from SeeingThroughFog/Hahner-2020 by the LISA authors):
  attenuates kept points, moves "cloud scatter" points to
  `dnew = ln(2)/alpha`, adds 5%-subsampled "random scatter" points uniform in
  `(2 m, min(dmax, d))`, and DROPS points beyond
  `dmax = -ln(0.05/(I+0.35))/(2*alpha)` that are not scattered.

The adapter uses the `haze_point_cloud` fog model (the only LISA fog path
that produces near-range scatter, as fog must) via `_lisa_haze_tracked`, a
line-for-line port in the adapter that additionally returns source indices —
the vendored method concatenates kept/scattered arrays and cannot carry
per-point metadata (`t_offset_us`, `ring`). Math and constants are verbatim
(`pylisa/lisa.py:339-424`). Vendored quirk kept: a random-scattered point is
not removed from the kept set, so it can appear twice (original attenuated
return + near noise return).

**Input.** `[N, 4]` x, y, z (meters, SENSOR frame — range is `||xyz||`),
reflectivity normalised to **[0, 1]** (docstring: "Reflection data must be
normalized to range [0 1]"). Our intensity01 passes through unchanged.

**Extinction parameter.** LISA does not take alpha directly for fog: it
computes `alpha = (pi/4)*1e-6 * \int D^2 Q_ext(D) N(D) dD` (1/m) from a fixed
drop-size distribution N(D) via Bohren–Huffman Mie
(`atmos_models.alpha_beta`). alpha is LINEAR in N(D), so scaling the
distribution by `beta/alpha0` makes the extinction exactly our beta;
equivalently `rho = 20 * beta / alpha0` in `nd_strong_advection_fog`.
Measured on the vendored code (lam=905 nm, m=1.328): alpha0 =
0.05950 1/m (strong advection fog), 0.03847 (moderate advection),
0.00350 (Chu/Hogg). Since the fog model consumes only the scalar alpha
(`beta_usefull = alpha`), the adapter feeds `alpha = beta` directly —
identical to the density scaling, exact by construction.

**Other parameters (defaults).** wavelength 905 nm, droplet refractive index
m=1.328, beam divergence bdiv=3 mrad, range accuracy dR=0.09 m, rmax=200 m,
rmin=1.5 m; haze_point_cloud hardcodes Velodyne HDL64-S2 noise floor n=0.05,
gain g=0.35, dmin=2 m, random-scatter subsample fraction 0.05.
Randomness: global `np.random`; the adapter calls `np.random.seed(rng_seed)`.

**Returns.** `[M, 5]` (x, y, z, intensity, label 0 kept / 1 cloud scatter /
2 random scatter), M <= N + duplicates; lost points and points with
range <= 2 m are dropped. Scattered points keep their ray direction (xyz
scaled by r'/r).

**Deps.** numpy, scipy (`scipy.special.gamma`, `scipy.integrate.trapz`),
ctypes + `pylisa/mie.so`. Two login-node fixes:
- `mie.so` was x86-64 → recompiled for aarch64 (see header). With that,
  `python3 -I -c "import pylisa.lisa"` works on the login node.
- `scipy.integrate.trapz` was removed in scipy>=1.14; the adapter shims
  `scipy.integrate.trapz = np.trapezoid` before importing pylisa.
No container needed (NGC container fallback not required).

## 3. Adapter contract

`/capstor/scratch/cscs/yitiwang/foggyfields/medium/foggyfields/fogsim_adapter.py`

`fog_sweep_hahner(points_ego, beta, T_lidar_rig, rng_seed)` and
`fog_sweep_lisa(...)`, same signature. Input `[N, 6]` float32
(x, y, z, intensity01, t_offset_us, ring) in EGO frame; transformed to the
sensor frame with `inv(T_lidar_rig)` (rig<-lidar, `p_rig = T @ p_lidar`, same
convention as `calib/T_lidar_rig.json`) before the sim and back after.
Intensity mapped to sim units (x255 for Hahner, passthrough for LISA) and
back to [0, 1] (clipped). t_offset_us and ring carried per source point
(by index for Hahner, by tracked source index for LISA). Dropped points
removed (LISA only; Hahner drops none). `sys.path` insertion of the pinned
vendored dirs happens inside the functions. Verified: same seed → bitwise
identical output; different seed differs; frame round trip exact;
t/ring preserved.

## 4. Smoke test results (login node, 2026-10-26)

Command: `python3 -I third_party/smoke_fogsim.py <sweep_0000.npy> <T_lidar_rig.json> 0.03`
Sweep: `/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/data_all/clips/NCF-02__02805273/lidar/sweep_0000.npy`
(304,773 pts, ranges 1.3–193 m from sensor), beta = 0.03 1/m, rng_seed = 0.
"near" = range < 10 m from the sensor origin.

```
== Hahner (LiDAR_fog_sim) ==
  n points in/out:        304773 / 304773
  near-range (<10 m) in:  62537 (0.2052)
  near-range (<10 m) out: 80484 (0.2641)
  fraction of output moved to range < 10 m (source >= 10 m): 0.0589
  mean intensity in/out:  0.0684 / 0.0233
  runtime:                1.79 s
  SANITY more near points: PASS (80484 vs 62537)
  SANITY lower mean intensity: PASS (0.0233 vs 0.0684)

== LISA (haze_point_cloud, fog) ==
  n points in/out:        304773 / 286712
  near-range (<10 m) in:  62537 (0.2052)
  near-range (<10 m) out: 64483 (0.2249)
  fraction of output moved to range < 10 m (source >= 10 m): 0.0046
  mean intensity in/out:  0.0684 / 0.0406
  runtime:                1.49 s
  SANITY more near points: PASS (64483 vs 62537)
  SANITY lower mean intensity: PASS (0.0406 vs 0.0684)

OVERALL: PASS
```

## 5. Deviations from the task spec

1. LISA's canonical fog entry (`Lisa.augment` = `augment_avg`) cannot satisfy
   the required sanity check (it creates no near-range scatter), so the
   adapter uses LISA's `haze_point_cloud` fog model instead, ported with
   index tracking (identical math) to preserve t_offset_us/ring — the
   vendored function discards per-point correspondence. No fallback
   closed-form reimplementation was needed; both clones work.
2. `pylisa/mie.so` recompiled for aarch64 inside the vendored tree (original
   kept as `mie.so.x86_64.orig`).
3. Hahner `ParameterSet` init-order quirk fixed by passing `mor`/backscatter
   `beta` explicitly (upstream would silently use alpha=0.06-derived values).
