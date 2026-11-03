# FoggyFields Phase 1 — Final Report (condensed)

**Status: CLOSED 2026-11-03.** NeuRAD-based modelling stopped by decision. This phase is
re-framed as the **feasibility study** for the physics-first redesign: it proves that
purely data-driven physical decomposition on an implicit backbone is *possible but
structurally capped* — physics enters only at the renderer's edge and trades directly
against fidelity (the seesaw). Full experimental log: `REPORT_full.md`. Distilled
handoff for the new design: `../../HANDOFF_PHYSICS_V2.md`.

## 1. Verdict in four sentences

1. **The physics layer itself is correct and identifiable**: with a frozen scene and
   radiometric quantities, 3-scalar inverse rendering recovers β to 2–4% (camera) /
   ~12% (LiDAR) and explains +13–17 dB of synthetic fog (P2, controlled).
2. **On real fog, no variant achieves a valid decomposition**: the visually-valid
   defog (V2b, 25.2 PSNR) fails the separation metric (dark-β 0.019), while every
   metric-passing variant (V2f 0.005, V2t-tv0 0.005) produces a visually invalid
   defogged render (black sky, degraded scene) — and the dark-channel metric itself
   rewards dark renders, so those "passes" are partly artifact. Enforced physical
   honesty also costs 4.7 dB, because the backbone's appearance is where fog hides.
3. **It does not hallucinate**: on near-clear data the released medium converges to
   β=0.0054 (below its own anchor) at a specificity cost of only 0.1 dB.
**Closing evidence (SYNfog, the only run with true defog GT):** trained jointly on
synthetic β=0.06 fog, the model reconstructs the foggy observation at 30.8 PSNR, but
after the anchor is released β collapses to 0.0025 and the defogged render scores only
**9.7 PSNR / 0.54 LPIPS against the true clear images — below the 11.4 PSNR of the
foggy observation itself (the do-nothing baseline)**: zero defog gain with ground truth
available. This is the cleanest single number behind the pivot. (Fairness caveat: this
run used a single released anchor; a double-anchored variant would hold β=0.06 by
construction but pay the known ~4-7 dB homogeneity/fidelity cost — not run, phase closed.
Note: lane-generated videos render the CNN path, which is UNSUPERVISED under
rgb_supervision=rgb_ray and outputs garbage colors — judge only rgb_ray renders.)

4. **The cap is architectural, not a tuning failure**: NeuRAD/SplatAD render *features
   through a CNN*, not radiance; density is a reconstruction proxy, not σ_t; LiDAR
   intensity is a non-radiometric readout. A medium bolted onto that pipeline can only
   act on retrofitted scalar quantities → redesign NFL/DyNFL-style, physics-first.

## 2. Final results (held-out, NFF-15 real fog unless noted)

| Variant | What it tests | rgb-ON PSNR/SSIM/LPIPS | Separation (dark-β off, floor 0.0077) | LiDAR med err / intRMSE |
|---|---|---|---|---|
| V0 reference | no medium, ray head | 26.68/.844/.241 | — | **0.39 / 0.057** |
| V2b | medium, free β | 25.2/.84/.25 | ✗ 0.019–0.022 (partial) | 0.55 / 0.096 |
| V2f | homogeneous, double-anchored | 18.98/.78/.347 | 0.0034–0.0055 — defog visually degraded (same caveat) | 0.56 / 0.097 |
| V2t-tv0 | spatial β(x), TV 0.05 + sat-pen | 22.04/.804/.299 | 0.005 — but defog is visually INVALID (black sky; metric gamed) | 0.55 / 0.096 |
| V2m | + in-loop return mixture | 21.8/.807/.496 | over-corrects (−0.010) | **1.11** / 0.106 → FAIL |
| N13med (near-clear) | specificity | 28.77/.852/.151 | β→0.0054 (no hallucination) | 0.21 / 0.079 |
| N13ref (near-clear) | reference | 28.88/.857/.148 | — | 0.20 / 0.057 |
| SYNfog (synthetic β=0.06) | defog vs TRUE clear GT | 30.8/.898/.135 (fits fog obs well) | ✗ β collapsed 0.06→0.0025 after anchor release; **defog vs clear GT = 9.7 PSNR / 0.54 LPIPS** | — |

P2 controlled inversion (frozen scene, synthetic fog): camera β error 2–4% (both
simulators), LiDAR −12% (Hahner convention), β(t) ramp 6–17%, A to 0.75 of truth;
zero-fog and self-consistency controls clean.

## 3. Lessons (error → experience; processes omitted, details in REPORT_full.md)

1. **Fog bakes into appearance, not density** (A8: free-space extinction of a
   fog-trained NeuRAD equals the clear floor). Post-hoc un-baking fails; the medium
   must be present from step 0 with an anchored warm-up.
2. **Feature-space rendering has no radiometry.** Physics needs per-sample radiance;
   the retrofitted ray-resolution head caps quality — this is the root of the seesaw,
   and no scheduler/regularizer removes it.
3. **Every multiplicative pair degenerates.** gain×head-scale, β×appearance, A×c_sky
   at T→0. Rule: pin or anchor one side of each product (gain frozen, A anchored,
   β anchored warm-up, sky-radiance prior in optically thick regions).
4. **Sensor constants don't port from simulation.** Synthetic θ was catastrophic on
   real data; calibrate θ on real clear clips (bisection on drop rate → 1.43e-5).
5. **AD LiDAR "intensity" is range-compensated and threshold-censored.** ln(I·r²)
   anchor fits fail by construction; a log-ratio residual on confident surface
   returns works. η pinned to Hahner 0.0154; mind LISA's one-way convention.
6. **The return mixture is a correct sensor model but cannot share a data stream
   with a second, non-physical renderer.** Offline (P2) it passes; in-loop it
   destabilized geometry twice (V2 collapse; V2m: 1.11 m vs 0.55 m median).
7. **Heuristic input masks (trusted-return) hurt once a medium exists** — let the
   physics explain fog returns instead of hiding them from the loss.
8. **Spatial fields need an anti-flat mechanism AND gentle smoothness.** tanh
   saturation penalty cured the flat attractor; TV 0.5 re-flattens, TV 0.05 works.
9. **Verify β conventions across simulators** (Hahner two-way vs LISA one-way
   intensity attenuation) before using either as calibration.
10. **Dark-channel β of the defogged render is a cheap screening metric BUT is gamed
    by dark/degraded renders** (a black sky scores "separated"). Separation claims need
    visual inspection and, where possible, GT-based scoring; never a darkness-based
    metric alone.
11. **Check flag defaults when a fix lands mid-campaign.** `gain_learnable` defaulted
    True; only SYNfog pinned it. Measured drift: V2f 1670×, V2t-tv0 41×, V2t 32×,
    V2m 6.9×, N13med 3× — the LiDAR intensity "physics" in those runs is contaminated
    by the gain degeneracy, so their intensity numbers are predictions, not attenuation
    physics.
12. **An accepted CLI flag is not an implemented feature** (`lidar_viz.py --video`
    parsed and silently did nothing — same class as lesson: verify the artifact, not
    the exit code).

## 4. What carries over (backbone-independent)

Modules `medium/foggyfields/` (MediumIntegrator τ-LUT, transport/airlight, NFL-style
mixture, fogsim adapters) + 18 unit tests; calibrations (β_C=0.0321±0.003, A=0.544,
θ=1.43e-5, η=0.0154, clear floor 0.0077); A1/A2/A4 analyses; eval harness
(p3_eval.py + bench lidar_viz 3D); synthetic clips with true clear GT (data_fogsim);
locked splits; the anchored-warm-up/alternation training protocol; the specificity
control as a standard eval. See HANDOFF_PHYSICS_V2.md.

## 5. Useful visualizations (keep; everything else expendable)

- `viz/p3_rounds/*_GT_withMedium_defogged.png` — RGB trios: GT | medium-on | defogged
- `viz/defog_video_V2b/GT_withMedium_defogged.mp4` — 99-frame RGB defog video
- `viz/lidar_medium_V2b/{bev,range}_intensity_sweep{10,50}.png` — LiDAR defog trios
- `viz/lidar3d_{V2b,B0}/{bev,range,occupancy}.png` (+ bev.mp4) — NFL-style 3D point
  clouds, same tool as the published benchmark → directly comparable
- `analysis/A1_halo*`, `A2_tail*`, `A4_beta_anchors*` — fog phenomenology figures
- P2 fit figures under `p2/`
- `~/foggyfields/rgb_transport_single_ray.html` — interactive 3-panel explainer
- `store-shared/yitiwang/FoggyFields_Advisor_Update.pptx`


## 6. Validation addendum (2026-11-03, requested re-audit; real-fog NFF focus)

Re-verified: configs match claimed recipes (7 variants, flags inspected in config.yml);
all table numbers match raw eval JSONs; evals reproduce on fresh GPU runs to ±0.01
(V0/V2b/V2t-tv0 re-evaluated); 0.0077 floor provenance (A4, NCF-02 dark-channel,
±0.0012); A8 numbers on record.
Corrected by this audit: (a) V0 LiDAR row was wrong — true values 0.387 m / intRMSE
0.057, i.e. **adding the medium degraded LiDAR geometry (0.39→0.54 m) and intensity
(0.057→0.10)**; (b) "true separation" claims for V2f/V2t-tv0 downgraded — their defogged
renders are visually invalid and the dark-channel metric rewards exactly that failure;
(c) "gain pinned" was false for every variant except SYNfog (measured drifts above).
Net effect: the case for the physics-first pivot is STRONGER — Phase 1 produced either
a visually plausible defog without measurable separation (V2b) or metric-artifact
separation with broken appearance, never both, while degrading LiDAR throughout.
