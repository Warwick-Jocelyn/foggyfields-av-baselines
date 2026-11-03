# HANDOFF — FoggyFields Physics-First Redesign (for the next design agent)

You are designing **architecture v2**: an explicit, physics-first neural rendering model
for camera+LiDAR in adverse weather (fog/spray), in the spirit of NFL and DyNFL, replacing
the Phase-1 approach (a medium bolted onto NeuRAD). Phase 1 is CLOSED; treat everything
below as validated input. Target: top-tier venue (ICCV/CVPR quality bar).

## 0. One-paragraph context
Phase 1 proved: single-scatter transport with one shared world-space extinction field
β — camera one-way T, LiDAR round-trip T² — is correct, calibrated and identifiable
(controlled inversion to 2–4% β error), decomposes real fog without hallucinating on
clear data, BUT on an implicit feature backbone physical honesty trades 4.7 dB against
fidelity (the seesaw) because the backbone renders CNN features, not radiance, its
density is a reconstruction proxy, and its LiDAR intensity is non-radiometric. The fix
is architectural: make the physical quantities the primary representation.

## 1. The physics to keep (verbatim)
- Medium: τ(s)=∫β dt; camera C = Σ w_i T(s_i) c_i + Σ_k W_sc[T(a_k)−T(b_k)]·A
  (exact ordered airlight, zero behind opaque surfaces; sky = surface at s_max).
- LiDAR surface echo E_S = a_S·T²(D)·K(D), K(s)=1/(s²+s0²); fog echo per bin
  E_F(s_k)=η·β_k·T²(s_k)·K(s_k)Δs; detection threshold θ; return = selection
  (strongest / mixture NLL), never a soft average of ranges.
- Coupling rule: σ_scene and β_medium ADD inside one transmittance exponent (RTE);
  attenuation is multiplicative on radiance, in-scatter is additive.

## 2. Calibrated constants (real-sensor, reuse directly)
| Quantity | Value | Source |
|---|---|---|
| β_C (NFF-15 day fog) | 0.0321±0.003 m⁻¹ (MOR≈93 m) | dark-channel camera anchor |
| A (airlight, NFF-15) | 0.544 | sky-region fit |
| θ (detector floor) | 1.43e-5 | bisection on real clear-clip drop rate |
| η (backscatter/extinction) | 0.0154 = 0.046/ln20 | Hahner constant, pin it |
| k_L (905nm/visible extinction ratio) | ≈1, fixed | fog droplets ≫ λ |
| clear floor of dark-channel β metric | 0.0077 | measured on clear clips |
| s0 (near-field receiver) | 2 m | fit |
NFF-13 is near-clear SPRAY (β≈floor; halo = road spray, not suspended fog) — keep it
as the specificity/contrast clip. NFF-05 is night (camera anchor ill-posed).

## 3. Identifiability map (design around these BEFORE training anything)
Degenerate pairs — pin or anchor one side of every product:
gain×reflectance-scale; β×appearance (if appearance is unconstrained, fog hides there);
A×c_sky where T→0 (use a sky-radiance prior = observed pixel); η×β in the fog echo.
Anchored warm-up (freeze β,A at measured values, release later) prevented every
collapse that free joint optimization produced.

## 4. Ten lessons from Phase-1 failures (one line each; log in reports/overnight1/)
1. Fog bakes into appearance, not density — medium must exist from step 0.
2. Feature-space rendering has no radiometry — render physical radiance end-to-end.
3. Every multiplicative pair degenerates — anchor one side.
4. Sensor constants don't port from simulation — calibrate θ on real clear data.
5. AD LiDAR intensity is range-compensated + censored — model the sensor processing
   (log-ratio residuals on confident returns; censoring via θ), don't fit raw ln(I·r²).
6. A physical sensor model cannot share one data stream with a second non-physical
   renderer — v2 must have ONE forward model per sensor.
7. Don't mask fog returns away — explain them (they are signal for β).
8. Spatial β(x): saturation penalty against flat attractors + weak TV (0.05, not 0.5).
9. Hahner is two-way, LISA one-way in intensity — check conventions in code.
10. Verify every pipeline stage produced its artifact (silent no-ops cost 3× once).

## 5. Requirements for architecture v2
R1 Radiometric end-to-end: field outputs physical radiance/reflectivity; sensor
   equations ARE the decoder (no CNN between field and loss). Accept the PSNR cost;
   the paper's claim is decomposition+explainability, measured also by LiDAR.
R2 One transmittance: T = exp(−∫(σ_scene+β_med)); surfaces as SDF (DyNFL-style) for
   sharp returns, medium as the volumetric term. The A8 trap (carving/proposal losses
   emptying free space) must not apply to β.
R3 Full NFL sensor chain for LiDAR: beam divergence (sub-rays), two returns, θ
   censoring, K(s) — plus our medium generalization (E_F fog echo, T²). Supervise on
   the range-image grid (128×3600, true no-return labels exist; half-rate column pairs).
4  Camera shares the same β: single-pass T + ordered airlight. Cross-modal consistency
   of one field is the headline claim — report LiDAR defog results with the same
   prominence as RGB.
R5 Dynamics: explicit actor composition (DyNFL SDF composition / SplatAD boxes).
R6 Keep the Phase-1 protocol: anchored warm-up → release; pinned sensor constants;
   TV+saturation prior; specificity clip (NFF-13) and zero-fog reduction as standard
   evals; dark-channel-off separation metric vs the 0.0077 floor.

## 6. Reusable assets (paths)
Workspace: /capstor/scratch/cscs/yitiwang/foggyfields (symlink ~/foggyfields)
- medium/foggyfields/{medium,transport,mixture,trusted,fogsim_adapter}.py — backbone-
  independent; 18 unit tests in tests/ (run in the NGC container).
- jobs/p3_eval.py — metrics harness (rgb on/off, dark-channel separation, LiDAR bins,
  defog-vs-true-clear via --defog-gt-root); bench 3D viz: ncf-crop-ablation/jobs/lidar_viz.py.
- data_fogsim/ — synthetic fog clips with TRUE clear GT (the only defog-GT eval).
- splits/overnight1.json (locked); analysis/ A1/A2/A4; results_P3_all.csv (full ablation).
- Checkpoints kept: V2b (best visual defog), V2t-tv0 (best true separation), V0, N13*.
- Repo: Warwick-Jocelyn/foggyfields-av-baselines (docs + eval code pushed).
- Reference papers in play: NFL (Huang et al.), DyNFL, NeuRAD, SplatAD, Hahner fog sim,
  LISA (note its one-way intensity convention).

## 7. Key numbers to beat / reference
NFF-15: V0 26.68 PSNR (no medium) · V2t-tv0 22.04 with TRUE separation · LiDAR median
err 0.55 m, intRMSE 0.096 · bench baseline B0 (CNN path) ≈ 30 PSNR / NN med 0.19–0.26 m.
NFF-13: ref 28.88/0.20 m. Specificity cost of medium: ≤0.1 dB. P2: β to 2–4%.
SYNfog (synthetic fog, TRUE clear GT): Phase-1 joint training fits the fog obs at 30.8
PSNR but defog-vs-clear-GT is only **9.7 PSNR / 0.54 LPIPS, below the 11.4 do-nothing baseline** (beta collapsed post-release)
— the quantitative proof that data-driven decomposition fails; v2 must beat this by a
wide margin on the same clip (data_fogsim NCF02F-Hb060, --defog-gt-root eval).
v2 succeeds if it holds separation (dark-β ≤ floor) at ≤1–2 dB cost instead of 4.7,
AND improves fog-clip LiDAR (fewer halo returns explained as surface, correct drop
rate by range bin — see analysis/A2 classes).
