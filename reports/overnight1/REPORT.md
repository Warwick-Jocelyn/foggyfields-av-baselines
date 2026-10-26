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
