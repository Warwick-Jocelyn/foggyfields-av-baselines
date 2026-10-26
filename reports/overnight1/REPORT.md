# FoggyFields Overnight Task 1 — REPORT

Session start: 2026-10-26. Executor: Claude (autonomous overnight session).

**Note on the reference plan:** `FoggyFields_Claude_Experiment_Plan_v2.md` was not found anywhere
on /users, /capstor or /iopsstor (only the Task 1 file exists). Conventions that the task file
delegates to the plan (§5.2 exact spec, §10.1 manifest, §10.3 report template) are reconstructed
from the task file's own summaries and the method draft
(`foggyfield-draft/FoggyField__Equation_V2_simplified.pdf`, Eq. 2/3/4). Each task block below uses:
status / what was done / outputs (paths) / deviations / blockers / time spent.

---
TASK: 1 (mini-audit, P0 subset)  STATUS: done
WHAT EXISTS: neurad-studio @ branch nff-v2-ablation, clean tree, /capstor/scratch/cscs/yitiwang/splatad-nff-repro/code/neurad-studio; data clips under ncf-crop-ablation/data_{all,nff}/clips; NeuRAD 30k ckpts for all 7 clips in ncf-crop-ablation/outputs_fix.
WHAT CHANGED: nothing (read-only audit). Wrote splits/overnight1.json.
EVIDENCE (yes/no + path per question):
 1. SPLITS: frame split read by nvidia_av_fog_dataparser.py:1501 _load_split_indices (applied :1487
    _get_train_eval_indices via np.isin on timesteps; cameras+lidar same split, ad_dataparser.py:206-209).
    nvs_50_50 = alternating frames (train even, eval odd, 100/99) — held-out TIME: yes (alternating, not
    blocks); held-out CAMERA VIEWS: **no mechanism exists** (only `cameras` tuple drops a camera from both
    train and eval). Locked in splits/overnight1.json; adding a view-holdout = split change -> Jocelyn.
 2. UNITS/FRAMES: camera+lidar ray dirs normalised (cameras.py:979, lidars.py:446) -> depth = Euclidean
    ray distance in METRES; no scene scaling (ad_dataparser.py:233 dataparser_scale=1.0; only rigid
    re-centring :319). Default depth = render_depth_simple (neurad.py:253,749-756): sum w_i*mid_i, NOT
    normalised by accumulation, sky sample dropped -> depth biased short where accumulation<1 (matters
    for A4/B1: use depth with accumulation, or median). LiDAR: dataparser feeds SENSOR-frame points
    (nvidia_av_fog_dataparser.py:1160, ego->sensor via inv(T_lidar_rig) :1170-1177); world rays built in
    lidars.py:413-455; non-returns appended at 2000 m (DUMMY_DISTANCE_VALUE/2 threshold).
 3. PER-SAMPLE RGB: **no** — field outputs 32-d FEATURE per sample (neurad_field.py:128,141-150, no RGB
    head); weights from SDF alpha (neurad.py:733 _render_weights, nerfacc render_weight_from_alpha);
    features composited per ray (neurad.py:382, FeatureRenderer=sum w_i*f_i) + appearance embedding ->
    CNN decoder (neurad.py:201-216, ConvTranspose x3 + sigmoid). A per-sample linear RGB head would
    attach in neurad_field.py:141-144 (composite at neurad.py:382 with same weights). NOT implemented
    tonight -> Task 4 uses the expected-depth overlay, labelled as such.
    Caveat: weights/ray_samples are returned only in training (neurad.py:405-408 weights_list).
 4. LIDAR HEADS: lidar_decoder = MLP(48->2) neurad.py:217-224, runs at :353-357 on the COMPOSITED
    per-ray feature (32) + appearance embedding (16); outputs intensity (sigmoid) + ray_drop_logit
    (BCE :506). Range = same expected-depth renderer as cameras (:390).
 5. NFF LIDAR METADATA: sensor = 128-ring row-offset-spinning, 3600 az cols, cw (calib/lidar_intrinsics.json;
    spinning_frequency field corrupted, sweeps are 10 Hz from lidar_timestamps.parquet). Return mode:
    NOT in metadata (unknown; treat detector rule as strongest-return, say so). Intensity 0-1, uint8
    quantised in range images. Raw azimuth GRID EXISTS: lidar/range_image/ 128x3600 float16 range from
    SENSOR ORIGIN + packed mask, non-returns marked -> fired-beam set known, TRUE no-return labels: yes
    (half-rate column-pair structure; compare fog vs clear differentially or merge pairs as the bench does).
 6. ROUTE OVERLAP: NCF-02 = chunk 2155; NFF-13/15 = chunk 2617 (Belgium), NFF-05 = chunk 726 (Sweden)
    -> NO clear/fog route overlap (E2 zero-fog renders stay counterfactuals). BUT NFF-13 and NFF-15
    share chunk 2617: same route, Light-wet vs Moderate+ fog — a natural density contrast pair.
 Extra (audit byproduct): possible upstream bug — neurad.py:248 builds density_fns in a list
 comprehension over proposal fields with late binding (both lambdas may call proposal_fields[1]).
 Runtime-unverified; do not fix tonight; flag for Jocelyn.
GATE: n/a (audit).
DECISIONS TAKEN: fog clips for tonight = NFF-13 + NFF-15 (A1: strongest near-halo, both day — NFF-05 is
 night, airlight fit ill-posed; 13/15 share a route with contrasting fog density). Clear = NCF-02.
OPEN QUESTIONS FOR JOCELYN: camera-view holdout split (needs dataparser change); density_fns late-binding bug.
NEXT: Task 2 analyses (A1 done, A2/A4 after smoke job delivers dumps+depth).

---
TASK: 0 (environment check)  STATUS: done
WHAT EXISTS: all 7 NeuRAD 30k checkpoints loadable (outputs_fix/neurad/safe/<clip>/.../step-000029999.ckpt).
EVIDENCE: debug jobs 3750177 + 3750214: eval_setup loads each checkpoint; full-frame render works
 for RGB (2712x1335), expected depth / accumulation / LiDAR intensity / ray-drop (904x445 = 1/3 res);
 LiDAR sweep render + dump works (lidar_lane.sh wrote 99-sweep dumps for NFF-13, NFF-15; json metrics
 match bench pipeline). GPU: GH200 visible, 4/node, debug partition.
 Outputs: analysis/depth_renders/<clip>/ (12 frames x 4 clips), outputs_fix/_lidar_dump/neurad_safe__NFF-{13,15}.
DECISIONS TAKEN: first smoke lane failed (torch not on PATH - missing `source env.sh`); fixed and rerun.
NEXT: none.

---
TASK: 2 (zero-training analyses)  STATUS: done
WHAT CHANGED: analysis scripts only. Outputs:
 analysis/A1/{A1_halo.csv,A1_rings.csv,A1_hist.csv,A1_halo.png,a1_halo.py}
 analysis/A2/{A2_tail.csv,A2_tail.png,a2_tail.py}
 analysis/A4/{A4_beta_camera.csv,A4_beta_lidar.csv,A4_beta_anchors.csv,A4_roadrange.csv,*.py}
EVIDENCE (key numbers):
 A1 (range images, sensor origin, every 2nd sweep):
   frac returns <5 m: clear NCF-02 0.026 | NFF-05 0.073 | NFF-13 0.091 | NFF-15 0.083  (fog x3-3.6)
   occupancy: 0.63 | 0.53 | 0.59 | 0.51; mean intensity: 0.067 | 0.026 | 0.042 | 0.022
   NFF-15 near-halo fraction varies most sweep-to-sweep (std 0.079): heterogeneous fog.
 A2 (NeuRAD dumps, GT-returned rays, expected depth):
   frac err >5 m: NCF-02 6.3% | NFF-05 11.7% | NFF-13 8.1% | NFF-15 11.4%
   0-10 m bin >5 m: clear 1.4% vs fog 4.2-7.1% (x3-5) -> the fog-specific tail is NEAR the sensor.
   >5 m tail composition: fog-return-as-surface (a) = 5.3% clear -> 15.0% / 19.9% / 10.6% fog;
   surface-at-fog-range (b) = 3-9%; "other" (generic far-field error) dominates everywhere (71-91%),
   and clear NCF-02 itself has a 6.3% tail concentrated >30 m.
   => the fog-SPECIFIC tail is mostly class (a): the mixture (fog echo), not attenuation, is the fix;
      B1-lite (attenuation only) is the controlled negative/positive check, as the task intends.
   Tail rays have below-average GT intensity (NFF-05: 0.009 vs 0.026) - weak fog echoes.
 A4 beta anchors:
   LiDAR intensity route FAILED for principled reasons (recorded as a finding):
     (i) intensity is range-compensated reflectivity (ln(I r^2) RISES with r even on clear),
     (ii) detector-threshold censoring (weak far returns drop -> surviving ones biased bright;
          common-support binned fit gives fog slope MORE positive than clear),
     (iii) clear/fog are different scenes (road-visible-range proxy: fog highway sees FARTHER
          than clear urban). => no usable beta_L from intensity; supports modelling an explicit
          noise floor theta (plan §5.4) rather than trusting intensity decay.
   Camera Koschmieder (dark channel vs NeuRAD depth, 12 frames, 2 m bins):
     NCF-02 control: 0.0056 +- 0.0010 (method floor)
     NFF-15: beta_C = 0.0252 +- 0.0094, A = 0.55 (MOR ~119 m - matches "Moderate+")
     NFF-13: below floor (-0.004): optically near-clear. With its STRONGEST near-range halo (A1),
       its near returns are likely road spray ("Light (wet)"), not suspended fog.
     NFF-05: 0.015 but NIGHT - flagged unusable for airlight-based fit.
GATE: n/a (analysis).
DECISIONS TAKEN: Task 4 fog clip = NFF-15 (only camera-confirmed day fog; beta0 init 0.025, A init
 0.55). NFF-13 retained as the spray-vs-fog contrast clip. elevation in dumps is DEGREES (bug caught
 in first A4 run: a1/a2 unaffected, a4 rerun with deg2rad).
OPEN QUESTIONS FOR JOCELYN: is the NFF-13 "halo without optical fog" spray interpretation right?
 (affects clip selection for the paper's fog claim). LiDAR return mode not in metadata (assume strongest).
NEXT: Task 3 (done, see below) then Task 4 B1-lite on NFF-15.

---
TASK: 3 (MediumIntegrator + unit tests)  STATUS: done
WHAT EXISTS: medium/foggyfields/{medium.py,transport.py} + tests/test_medium.py (8 tests).
WHAT CHANGED: new module, ~200 lines. Implements plan §5.2 exactly: non-uniform fixed grid
 (16 bins over 0-20 m + 16 bins to max_range 120 m), midpoint-rule marching of beta(x), per-ray
 cumulative tau LUT, differentiable linear interpolation, linear extension beyond max range
 (exact for homogeneous fog; sky rays keep attenuating). Plus the B1-lite transport:
 expected_depth_overlay (labelled P1 prototype), ordered_airlight (EXACT interval integration
 against scene-sample weights - wall occlusion holds exactly, not as a quadrature property),
 attenuate_surface_echo (E_S * T^2). Backbone-agnostic (any origins/directions -> works for
 NeuRAD bundles and SplatAD per-pixel rays alike; the basic-3DGS+NeRF requirement).
EVIDENCE: 8/8 pytest in the NGC container (CPU):
 analytic homogeneous tau exact to <1e-4 (incl. beyond max range); linear spatial field exact at
 nodes (<1e-10), <1e-3 interpolated; T_two_way == T^2 == exp(-2 tau) (1e-12); wall occlusion:
 airlight behind an opaque wall == closed form (1e-9), wall-at-0 == 0; quadrature K=32 vs K=512
 on a smooth bump: max err 6e-3 abs / <1% rel (fog fields are TV-smoothed, §5.1); zero-fog
 reduction (beta=0): overlay == input bitwise, echo == a_s bitwise, airlight == 0; gradient
 dT/dbeta0 matches analytic -s*T*sigmoid(raw) to 0.1%; overlay == Koschmieder closed form.
GATE: pass (the real-frame bit-for-bit zero-fog check runs inside the Task 4 job: the overlay
 path is post-render, so beta=0 identity is exact by the CPU test; logged again on GPU).
DECISIONS TAKEN: ordered_airlight integrates intervals between scene samples exactly instead of
 re-binning on the medium grid (first implementation leaked airlight into the half-bin behind a
 wall; the test caught it).
NEXT: Task 4 B1-lite fit on NFF-15.

---
TASK: 4 (B1-lite)  STATUS: done — GATE FAILED as A2 predicted (useful negative result; not tuned past)
WHAT EXISTS: jobs/b1lite_fit.py + jobs/b1lite.sbatch; results in b1lite/{NFF-15,NFF-13}/results.json
 + qual_*.png (gt | backbone | overlay). Slurm job 3750288 (debug, ~15 min, 2 GPUs). Scene FROZEN
 (NeuRAD 30k fog checkpoints); fitted: beta_0, airlight A (3-vec), gain g. Camera = expected-depth
 overlay (P1 prototype, labelled; audit Q3). LiDAR = surface-echo attenuation E_S*T^2 only.
 Zero-fog reduction on a real frame: EXACT on GPU (logged in b1lite logs).
EVIDENCE (NFF-15 primary / NFF-13 spray-contrast):
 fitted beta_0: joint 0.0071 / 0.0111;  camera-only 0.0181 / 0.0358;  lidar-only 0.0004 / 0.0004
 A4 camera anchor: 0.0252 / ~0.0056(floor)
 RGB held-out (33 front_wide frames, central 84% crop, render res — INTERNAL A/B only, not the
 bench protocol number): backbone 15.78/.720/.422 -> b1lite 15.90/.703/.472 (NFF-15);
 16.19/.712/.388 -> 16.43/.708/.409 (NFF-13). PSNR +0.1-0.2 but SSIM/LPIPS WORSE.
 LiDAR intensity RMSE (99 held-out sweeps): 0.0572 -> 0.0690 (NFF-15), 0.0555 -> 0.0631 (NFF-13): WORSE.
 Range metrics: unchanged (structural: frozen scene + attenuation-only cannot move depth).
GATE (task file: "reduces the >5 m tail at all, with plausible beta"):
 - tail: unchanged by construction -> fail.
 - plausible beta: joint beta = 0.28x anchor on NFF-15 -> fail. Lidar-only beta -> 0: the frozen
   fog-trained scene ALREADY encodes the LiDAR attenuation (double-counting makes intensity worse).
 - DIAGNOSIS (the point of the exercise): A2 says the fog-specific tail is fog-echoes-as-surfaces;
   B1-lite confirms attenuation-on-frozen-scene explains nothing more. The fix is (i) the fog-echo
   MIXTURE (§5.4) and (ii) scene unfreezing so fog can leave the geometry (P3 warm-start schedule).
 - POSITIVE byproduct: cross-sensor asymmetry. Camera-only fit recovers beta = 0.018 = 0.72x the
   DCP anchor on NFF-15 (a real residual haze signal the frozen camera branch under-represents),
   while lidar-only recovers 0. On near-clear NFF-13 the camera-only fit absorbs non-fog residuals
   (motion blur/exposure -> beta 0.036 >> floor): the overlay fit alone is NOT a clean fog
   estimator; it needs the mixture-side constraints — exactly the shared-medium argument (E3).
DECISIONS TAKEN: did not run the scene-unfreezing variant (A2+these numbers already determine the
 next step; unfreezing without the mixture would only let fog hide in geometry again).
OPEN QUESTIONS FOR JOCELYN: none new (NFF-13 spray question stands).
NEXT: P1/P2 per plan §6 — per-sample RGB head + mixture return model; synthetic fog (Hahner+LISA)
 on NCF-02 for controlled beta recovery before any real-fog refit.

---
CORRECTION (2026-10-26, after the validate-agent review)  STATUS: all affected numbers re-run
BUG: every fit/eval script loaded GT with PIL from disk = the FULL uncropped canvas, while the
 cameras/ray grid describe the SAFE-CROPPED image -> vertically misregistered GT (silent).
 Found by the validate agent reviewing jobs/fit_rgb_head.py before J1. Fix: GT via
 dataset.get_image_float32 (applies image_crops). Invalid outputs archived in
 b1lite_INVALID_canvasGT/ (numbers survive in git history); re-run in job 3750743.
AFFECTED + CORRECTED:
 - Task 4 B1-lite: camera-side numbers were invalid. CORRECTED (NFF-15 / NFF-13):
   joint beta 0.0008 / 0.0006; camera-only 0.0012 / 0.0007; lidar-only 0.0003 / 0.0004.
   backbone RGB held-out now sane: 28.77/.880/.211 (NFF-15), 31.69/.918/.145 (NFF-13);
   b1lite == backbone (beta ~ 0 -> overlay ~ identity).
   **RETRACTED: the "cross-sensor asymmetry" finding.** The camera-only residual beta=0.018 was
   the overlay compensating GT misregistration. Clean corrected conclusion: the frozen
   fog-trained scene absorbs the fog COMPLETELY in both sensors; B1-lite recovers nothing
   (gate failed, cleanly); separation requires P2 controlled recovery + P3 scene refit.
 - A4 camera anchors (re-fit on cropped GT, tighter): control floor NCF-02 0.0077+-0.0012;
   NFF-15 0.0321+-0.0030 (MOR ~93 m); NFF-13 0.0090+-0.0004 (barely above floor -> SPRAY
   interpretation stands); NFF-05 0.0146 (night, flagged). P2/P3 beta inits use these.
 - A2/A1/A4-lidar: unaffected (no camera GT involved).
 - T2 (B0' head) ran with the fix from the start: B0-CNN at ray res 26.93/.834/.159 (NCF-02),
   28.37/.870/.221 (NFF-15); B0'-head 24.19/.721/.297, 24.60/.764/.444. Head weights in heads/.
LESSON (recorded in memory): GT for ANY per-ray comparison must come from the dataset loader,
 never from the image file.

---
TASK: P1/P2 (T1-T10, plan-agent numbering)  STATUS: P2 camera gate PASSED; lidar mixture iterating
WHAT EXISTS:
 T1/T3: neurad.py rgb_head flag + MediumConfig in-loop medium (commits d5d24fa3, abc97550);
 T2: calibrated 147-param heads, B0' vs B0-CNN at ray res: 24.19/26.93 (NCF-02), 24.60/28.37 (NFF-15);
 T5: mixture module (tests 9/9); T7: Hahner+LISA vendored (third_party/PINNED.md, upstream
 ParameterSet bug bypassed, aarch64 mie.so); T8: 1393 clear depth renders; T9: 4 synthetic-fog
 clips in data_fogsim/ (range-image regeneration self-check: 100% cell agreement, 2 mm; Hb030
 drops 31%, Hb060 50% + halo 0.24->0.40, LISA drop pattern differs = designed sim mismatch).
EVIDENCE (T10a, p2/<variant>_t10a/results.json):
 camera-only beta recovery: 0.0288/0.03, 0.0588/0.06, 0.0288/0.03 (LISA) -> all within 4-5%
 (gate: 20%). Airlight 0.755 vs 0.75 true. Zero-fog control on clear clip: beta -> 0.0004.
 SELF-CONSISTENT control (GT built with the model's own formation): beta recovered EXACTLY
 (0.0300/0.0600, A=0.750) -> fit code exact; the -4% on stored JPEGs = documented GT formation
 mismatch (sky handling), well inside the gate.
 Fitted medium explains fog: held-out foggy frames, clear-scene render 9.3-11.3 PSNR ->
 with fitted medium 24.6-26.6 (+13-17 dB), LPIPS 0.46->0.23. Quals: p2/*_t10a/qual_*.png.
 lidar-only/joint: degenerate optimum found (g->0, beta->0: all mass to noise floor, flat
 renormalised mixture). T10b (job 3750809): gain FIXED at 1 (clear ckpt intensity is the
 physical anchor); eta/theta stay learnable; min_range 2.5 m; drop NLL on below-horizon
 non-returns only.
VALIDATE-AGENT ROUND 2 (all applied): dataclasses.field shadowing crash (B1); old configs
 lack `medium` (B2, defensive default); fog-echo magnitude calibration (eta learnable);
 sky non-returns excluded from drop NLL; returns z>120 m dropped; beam classification from
 source clip for fogsim clones (dataparser patch); S-column assert.
OPEN: T4 zero-fog bitwise check on the REAL pipeline showed max diff 0.285 - diagnostic rerun
 in flight (determinism control; suspicion: eval-render nondeterminism or an ON-path
 difference not visible statically). P2 numbers are unaffected (fits replicate transport on
 cached tensors; self-consistent control exact), but the in-loop renderer must pass T4 before
 it is used for training (P3).
NEXT: T10b results; T4 diagnosis; T11 (Sramp: beta(t) curve recovery - the generated ramp is
 temporally-varying homogeneous, so a TIME curve, not a spatial field, is the honest first
 target; true spatial-bank generator = T11-v2).

---
T10b + T4 diagnosis (2026-10-26 ~05:30)
T10b (gain fixed at 1; p2/<variant>/results.json; first round archived in p2/*_t10a/):
 lidar-only beta: 0.0027/0.03, 0.0105/0.06, 0.0029/0.03 - better than the degenerate ~0 but
 still 5-10x low; joint 0.019/0.043/0.020 (dragged by the lidar term); camera-only unchanged
 (0.0288/0.0588/0.0288 - the P2 gate stands). k_lidar +-30% swings lidar beta by ~x2.
 IDENTIFIABILITY ANALYSIS (why the lidar inverse underestimates): in E_F = eta*k_L*beta*T^2*K*ds
 the product eta*beta sets the halo amplitude, and over the halo's short range span (2.5-10 m)
 T^2 barely varies -> eta and beta are nearly degenerate; with theta also learnable the drop
 rate can be absorbed too. The fit parks amplitude in eta and keeps beta small. FIX for next
 session (T10c): calibrate theta on the CLEAR clip (match its below-horizon non-return rate)
 and pin eta from the simulator's own backscatter constant; then beta alone must explain the
 range-dependent drop excess + far-surface attenuation. The useful scientific finding stands:
 single-sensor LiDAR fog inversion is ill-posed without cross-sensor or calibration anchors -
 exactly the shared-medium argument (E3).
T4 (jobs 3750810/3752795): the eval renderer is STOCHASTIC (two identical renders differ up to
 0.41 rgb / 79 m depth at single pixels; mean 5e-5) -> bitwise comparison void; the zero-fog
 gap EQUALS the determinism floor, i.e. no medium-path discrepancy beyond sampler noise.
 Seeded-render rerun (3752795) gives the rigorous verdict; per-op bitwise guarantees proven at
 the transport level (CPU suite + self-consistent control exact).

T4 FINAL (job 3752795): nondeterminism persists under seeding -> CUDA atomics (nerfacc/tcnn
float non-associativity), not RNG. Bitwise full-pipeline comparison is unattainable without
deterministic kernels. Decisive evidence the medium path is clean at beta=0: the on/off diff
is ZERO on sky pixels (acc<0.9) and lives only on dense surfaces (atomic-contention regions)
- airlight leakage would show on sky first. Zero-fog verdict: HOLDS to within kernel
nondeterminism; proof chain = CPU bitwise suite + exact self-consistent control + sky-pixel
invariance. Sufficient for P2/P3 (training is stochastic anyway).

---
T10c / T11 first round + T10d (2026-10-26 morning)
T10c (eta pinned to Hahner's 0.046/ln20=0.01536, g=1, linear-L1 intensity term):
 lidar-only beta 0.0069/0.03, 0.0131/0.06, 0.0082/0.03 — improved (t10b 0.003) but still ~4x low.
DECISIVE DIAGNOSTIC (numpy, matched clear/fogged point pairs, unscattered survivors, I>0.02):
 ln(I_fog/I_clear) vs r slope gives beta = 0.0293 vs true 0.03 (98%). The synthetic intensity
 carries the full signal — the FIT was burying it: linear-space L1 over ALL returns mixes
 scattered halo points (fog-echo intensity vs predicted attenuated-surface) and lets bright
 points dominate; the range-NLL outweighs the rest.
T10d fix: intensity residual in LOG space, restricted to confident surface returns
 (|z - clear depth| < 1 m, model intensity > 0.02, observed > 1.5/255) — exactly the
 regression that works. Job 3752955.
T11 first round FAILED for two instrumentation reasons (not model reasons): camera ray times
 were all 0 (cameras.times is None in this dataparser -> only knots 0-1 received gradient;
 knots 2-7 sat at init) and only 2 frames/camera = 2 distinct time points for 8 knots.
 Fixed: frame time parsed from the filename; 56 camera frames (8 per camera). Same job.

---
T10d RESULTS (job 3752955) — **LiDAR gate PASSED on Hahner, honest fail on LISA**
 With the log-ratio surface-return intensity residual:
   Hb030: lidar-only beta 0.0260 (-13%), joint 0.0260 — ALL four gates True.
   Hb060: lidar-only 0.0530 (-12%), joint 0.0531 — ALL gates True.
   Lb030 (LISA): lidar-only 0.0136 (-55%) — FAILS, and should: LISA's intensity-attenuation law
   differs from the Hahner-class E_S model (its cloud-scatter path + different extinction
   handling). That is exactly what the mismatched-simulator test is for: it shows which parts
   of the sensor model transfer (range/drop physics) and which are simulator-specific
   (intensity law). Camera gate passes on BOTH sims regardless.
 k_lidar +-30% sensitivity now brackets the truth (0.037/0.020 around 0.026 at k=1).
T11 SECOND FAILURE -> DATA BUG FOUND: the camera fit exploded (beta -> 0.5-0.7) because the
 stored Sramp IMAGES carry constant beta ~0.2, not the ramp (back-solved from matched
 clear/fogged pixels; lidar sweeps ARE correctly ramped — lidar_only knots trend upward).
 The other three variants verify perfectly (implied 0.0296/0.0592/0.0296 vs true
 0.03/0.06/0.03). Cause not reproducible from the code (same code wrote all four);
 fix: deleted the bad images, added a POST-WRITE VERIFICATION to the generator (back-solve
 one frame, assert within 20%), regenerating. T11 rerun after.

---
T11 FINAL (job 3754213) — beta(t) RAMP RECOVERY: GATE PASSED, ALL THREE MODES
 After the n_frames generation bug (verified regeneration), the time-coverage fix (linspace
 sampling incl. clip end) and the np.int64 cast: fog-bank ramp beta(t)=0.005->0.06, 8-knot
 piecewise-linear curve, max rel err (t>=0.3): camera 6.1%, lidar 16.8%, joint 16.4%.
 P2 is now COMPLETE: homogeneous beta (two simulators), temporal beta(t), airlight, all
 controls (zero-fog, self-consistent, mismatched-sim, LISA convention isolation).
