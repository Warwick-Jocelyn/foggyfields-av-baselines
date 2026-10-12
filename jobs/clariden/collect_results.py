"""Collect metrics from all lanes into RESULTS.md (+ results.json)."""
import json, csv
from pathlib import Path
W = Path("/capstor/scratch/cscs/yitiwang/splatad-nff-repro"); OUT = W / "outputs"
PUB = {"NFF-03": dict(name="clip004 highway heavy fog", psnr=23.94, ssim=0.882, lpips=0.324, lidar_med=15.20),
       "NFF-02": dict(name="clip003 residential light fog", psnr=24.60, ssim=0.873, lpips=0.353, lidar_med=8.63)}
JUNE = {"NFF-02": dict(psnr=25.22, lpips=0.33)}
rows, res = [], {}
for variant in ("faithful", "corrected", "hf_ckpt"):
    for clip in ("NFF-03", "NFF-02"):
        d = OUT / variant / clip; r = {}
        if (d / "rgb_metrics.json").exists(): r["rgb"] = json.loads((d / "rgb_metrics.json").read_text())
        if (d / "lidar_metrics.json").exists(): r["lidar"] = json.loads((d / "lidar_metrics.json").read_text())
        r["done"] = (d / "LANE_DONE").exists(); res[f"{variant}/{clip}"] = r
tim = {}
tp = OUT / "_timings/timings.csv"
if tp.exists():
    for row in csv.reader(tp.open()):
        if len(row) >= 5: tim[(row[0], row[1], row[3] if row[2] == "render" else "train")] = row[4]
f = lambda x, n=2: "—" if x is None else f"{x:.{n}f}"
L = ["# SplatAD on NFF — reproduction results (Clariden GH200, %s)" % __import__("datetime").date.today(), "",
     "Protocol: nvs_50_50 (odd frames held out), 30k iters, seed 42, MCMC cap 5M, 2M seed points, camera-opt off — identical to the published config.",
     "`faithful` = original code; `corrected` = valid-pixel mask in RGB loss + LiDAR sweeps read in their true sensor frame (the original code offsets them by the 1.94 m mount height); `hf_ckpt` = the published Hugging Face checkpoint evaluated on the NFF copy of the clip.",
     "Metrics: torchmetrics PSNR/SSIM, LPIPS-AlexNet over all 7 cameras × 99 held-out frames. `masked` columns exclude the black rectification corners.", "",
     "## RGB (held-out)", "", "| Clip | Run | PSNR | SSIM | LPIPS | PSNR masked | SSIM masked | LPIPS masked | train min | render min |", "|---|---|---|---|---|---|---|---|---|---|"]
for clip in ("NFF-03", "NFF-02"):
    p = PUB[clip]; L.append(f"| {clip} | **published** ({p['name']}, Blackwell) | {p['psnr']:.2f} | {p['ssim']:.3f} | {p['lpips']:.3f} | — | — | — | — | — |")
    if clip in JUNE: L.append(f"| {clip} | June-2026 Clariden run (orig cfg) | {JUNE[clip]['psnr']:.2f} | — | {JUNE[clip]['lpips']:.2f} | — | — | — | — | — |")
    for variant in ("hf_ckpt", "faithful", "corrected"):
        r = res[f"{variant}/{clip}"].get("rgb")
        if not r: L.append(f"| {clip} | {variant} | (missing) | | | | | | | |"); continue
        tt = tim.get((f"splatad_{variant}", clip, "train")); rt = tim.get((f"splatad_{variant}", clip, "sec"))
        L.append(f"| {clip} | {variant} | {f(r['psnr'])} | {f(r['ssim'],3)} | {f(r['lpips'],3)} | {f(r['psnr_masked'])} | {f(r['ssim_masked'],3)} | {f(r['lpips_masked'],3)} | "
                 f"{'—' if not tt else int(tt)//60} | {'—' if not rt else int(rt)//60} |")
L += ["", "## LiDAR depth (camera-rendered expected depth vs GT returns, held-out frames, metres)", "",
      "Caveat: SplatAD's camera depth is gsplat's alpha-weighted *expected* depth (`RGB+ED`). In fog, semi-transparent Gaussians near the camera dominate it, so it saturates at roughly 10 to 17 m frame-wide even where PSNR is fine (verified on faithful/NFF-02: pred/GT ratio 0.6 median, GT 20 to 40 m rendered at about 9.6 m). The numbers below therefore measure that saturation, not projection quality, and are similar under both conventions. The method's native LiDAR rasterization (ns-eval `median_depth`, used for the June CD/RCD study) is the meaningful geometry metric.", "",
      "| Clip | Run | convention | median L1 | mean L1 | acc@5m | points |", "|---|---|---|---|---|---|---|"]
for clip in ("NFF-03", "NFF-02"):
    L.append(f"| {clip} | **published** | legacy (orig script, wrong) | {PUB[clip]['lidar_med']:.2f} | — | — | — |")
    for variant in ("hf_ckpt", "faithful", "corrected"):
        r = res[f"{variant}/{clip}"].get("lidar")
        if not r: continue
        for k, v in r.items():
            if isinstance(v, dict) and "depth_l1_median" in v:
                L.append(f"| {clip} | {variant} | {k} | {v['depth_l1_median']:.2f} | {v['depth_l1_mean']:.2f} | {v['acc@5m']:.3f} | {v['num_points']} |")
L += ["", "## Per-camera PSNR (masked / unmasked)", ""]
for key, r in res.items():
    if r.get("rgb"):
        L.append(f"**{key}**: " + ", ".join(f"{c.replace('camera_','')}: {f(v['psnr_masked'])}/{f(v['psnr'])}" for c, v in r["rgb"]["per_camera"].items()))
(W / "RESULTS.md").write_text("\n".join(L) + "\n"); (W / "results.json").write_text(json.dumps(res, indent=2)); print("\n".join(L))
