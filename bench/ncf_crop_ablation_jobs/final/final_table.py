#!/usr/bin/env python3
"""FINAL_TABLE.md for RGB-Branch-Metrics: every method x NFF-02/08/10 under the ONE common evaluator (nothing typed by hand).

  python3 final_table.py --out /users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics/FINAL_TABLE.md [--status "text"]

Sources: results_final/{allfix,opreg5e4,std,v13}/{metrics,lidar}/*.json (written by the bench lanes).
RGB = safe_crop primary (7 cams x 99 held-out frames); LiDAR = returns_only protocol (cd_returns_m, cd2_returns_m2,
depth_median_l2_returns_m). Mean column only when all three clips exist. Best external/ours value per column in bold.
"""
import argparse, json, time
from pathlib import Path

R = Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/results_final")
FOG = ["NFF-02", "NFF-08", "NFF-10"]
CLIPS = FOG + ["NFF-05"]   # NFF-05 = night / dark weather, reported separately (not in the fog mean)


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def ext(tag, meth):
    return lambda c: (R / tag / "metrics" / f"metrics_{meth}__safe__{c}.json", R / tag / "lidar" / f"lidar_{meth}_safe__{c}.json")


def v13(*labels):  # first label that exists per clip (NFF-02 BRDF lane is named C7gR, = U0 recipe)
    def f(c):
        for lb in labels:
            m = R / "v13" / "metrics" / f"metrics_v13__{lb}__{c}.json"
            if m.is_file():
                return m, R / "v13" / "lidar" / f"lidar_v13__{lb}__{c}.json"
        return m, R / "v13" / "lidar" / f"lidar_v13__{labels[0]}__{c}.json"
    return f


def v13key(*labels):
    def f(c):
        for lb in labels:
            if (R / "v13" / "metrics" / f"metrics_v13__{lb}__{c}.json").is_file():
                return f"v13-{lb}"
        return f"v13-{labels[0]}"
    return f


BUDGET = 30000
ROWS = [  # (name, group, iters, path fn, convergence key fn) -- the FAIR comparison: every row trained BUDGET steps
    ("SplatAD (allfix: opacity reg 5e-4 + F1b/F2b)", "SOTA", "30k", ext("allfix", "splatad"), lambda c: "splatad"),
    ("SplatAD (opreg5e4, pre-fix data path)", "SOTA", "30k", ext("opreg5e4", "splatad"), lambda c: "splatad-opreg5e4"),
    ("NeuRAD (standard, corrected bench)", "SOTA", "30k", ext("std", "neurad"), lambda c: "neurad"),
    ("UniSim (corrected bench, no ray drop; schedule x1.5)", "SOTA", "30k", ext("std", "unisim"), lambda c: "unisim"),
    ("EmerNeRF (corrected bench)", "SOTA", "30k", ext("std", "emernerf"), lambda c: "emernerf"),
]
ROWS_5K = [  # earlier 5k-step lanes: NOT budget-matched, appendix only
    ("NeuRAD A0 (V1.3 suite, 5k)", "ours-suite", "5k", v13("A0"), v13key("A0")),
    ("FoggyField V1.3 BRDF (U0 / C7gR, 5k)", "ours", "5k", v13("V13-U0", "V13-C7gR"), v13key("V13-U0", "V13-C7gR")),
    ("FoggyField V1.3 Colour (UcnnR, 5k)", "ours", "5k", v13("V13-UcnnR"), v13key("V13-UcnnR")),
    ("FoggyField V1.3 BRDF + LiDAR (F, 5k)", "ours", "5k", v13("V13-F"), v13key("V13-F")),
    ("FoggyField V1.3 Colour + LiDAR (Fc, 5k)", "ours", "5k", v13("V13-Fc"), v13key("V13-Fc")),
]
CONV = Path("/users/yitiwang/FoggyfieldV1/RGB-Branch-Metrics/baselines/convergence.json")


def verdicts():
    try:
        return {(r["method"], r["clip"]): r["verdict"] for r in json.loads(CONV.read_text())}
    except Exception:
        return {}
METRICS = [  # (label, kind, key, higher_better, digits)
    ("PSNR ↑", "rgb", "psnr", True, 2), ("SSIM ↑", "rgb", "ssim", True, 3), ("LPIPS ↓", "rgb", "lpips", False, 3),
    ("LiDAR CD (m) ↓", "lidar", "cd_returns_m", False, 3), ("CD² (m²) ↓", "lidar", "cd2_returns_m2", False, 2),
    ("depth median |Δr| (m) ↓", "lidar", "depth_median_l2_returns_m", False, 3),
    ("LiDAR accuracy pred→GT (m) ↓", "ext", "acc_m", False, 3),
    ("LiDAR completeness GT→pred (m) ↓", "ext", "comp_m", False, 3),
    ("LiDAR F-score @0.2 m ↑", "ext", "F@0.2", True, 3),
    ("LiDAR occupancy IoU @0.2 m voxels ↑", "ext", "occ_iou@0.2", True, 3),
    ("LiDAR occupancy IoU @0.5 m voxels ↑", "ext", "occ_iou@0.5", True, 3),
    ("LiDAR range MAE (m) ↓", "ext", "range_mae_m", False, 3),
    ("LiDAR intensity RMSE (all returns, /255) ↓", "ext", "int_rmse", False, 4),
    ("LiDAR intensity RMSE (unsaturated returns, /255) ↓", "ext", "int_rmse_unsat", False, 4),
]


def value(fn, c, kind, key, min_iters=0):
    mp, lp = fn(c)
    if min_iters and int((load(mp) or {}).get("iters") or 0) < min_iters - 1:
        return None          # e.g. the superseded 20k UniSim / 25k EmerNeRF results
    if kind == "ext":   # jobs/final/lidar_extended.py result next to the LiDAR JSON
        import re as _re
        xp = Path(_re.sub(r"/lidar/lidar_v13__", "/lidar_ext/lidar_ext__", str(lp)))
        if "/v13/" not in str(lp):
            xp = Path(_re.sub(r"/lidar/lidar_([a-z]+)_safe__", r"/lidar_ext/lidar_ext__\1__", str(lp)))
        d = load(xp)
        return None if d is None else d.get(key)
    if kind == "rgb":
        d = load(mp)
        return None if d is None else d.get("primary", {}).get(key)
    d = load(lp)
    return None if d is None else d.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status", default="")
    a = ap.parse_args()
    out = [f"# FINAL TABLE — SOTA baselines vs FoggyField V1.3 on NFF-02 / NFF-08 / NFF-10 (one common evaluator)", "",
           f"_Auto-generated by `jobs/final/final_table.py` at {time.strftime('%Y-%m-%d %H:%M')} from the metric JSONs; "
           f"`n/a` = run not finished yet._ {a.status}", "",
           "Protocol: split `nvs_50_50` (train even timesteps, eval the 99 odd ones). RGB = safe_crop pixels, 7 cameras x 99 frames = 693 "
           "images (PSNR/SSIM/LPIPS-alex). LiDAR = 99 held-out sweeps, GT-returned rays only, one predicted point per ray (rendered "
           "range along the GT ray): CD = symmetric mean NN distance, CD² = squared variant / |G|, depth median = median |r_pred−r_gt|. "
           "No method's ray-drop head is used (oracle drop) so every row is scored on the same rays. Details + per-run wall time: "
           "`baselines/RESULTS_BASELINES.md`.", "",
           "**Fair budget:** every row of the main tables trained exactly 30 000 steps (each method's LR schedule stretched to 30k "
           "where its default was shorter: UniSim 20k -> 30k incl. its stage schedule, EmerNeRF 25k -> 30k, V1.3 5k -> 30k with its 4 stages x6). "
           "Same split (nvs_50_50), same 693 held-out images and 99 held-out LiDAR sweeps, same evaluator for every row. "
           "Bold = best in the column.", "",
           "**Validation marks** (`baselines/CONVERGENCE.md`: finished, finite loss, loss plateau, train/held-out gap, renders not grey): "
           "plain number = PASS; `⚠` = a check FAILED (do not cite without reading CONVERGENCE.md); `†` = not fully validated yet.", ""]
    V = verdicts()
    def tables(rows, min_iters, title):
        res = [title, ""]
        for label, kind, key, hb, nd in METRICS:
            vals = {(r[0], c): value(r[3], c, kind, key, min_iters) for r in rows for c in CLIPS}
            best = {}
            for c in CLIPS + ["mean"]:
                col = []
                for r in rows:
                    v = vals[(r[0], c)] if c != "mean" else (
                        sum(vals[(r[0], k)] for k in FOG) / 3 if all(vals[(r[0], k)] is not None for k in FOG) else None)
                    if v is not None:
                        col.append(v)
                best[c] = (max(col) if hb else min(col)) if col else None
            res += [f"### {label}", "", "| method | group | steps | " + " | ".join(c + (" (night)" if c == "NFF-05" else "") for c in CLIPS) + " | mean (3 fog clips) |",
                    "|---|---|---|" + "---|" * (len(CLIPS) + 1)]
            for r in rows:
                cells = []
                vs = [vals[(r[0], c)] for c in CLIPS]
                fv = [vals[(r[0], c)] for c in FOG]
                mean = sum(fv) / 3 if all(v is not None for v in fv) else None
                for c, v in list(zip(CLIPS, vs)) + [("mean", mean)]:
                    if v is None:
                        cells.append("n/a")
                    else:
                        s_ = f"{v:.{nd}f}"
                        s_ = f"**{s_}**" if best[c] is not None and abs(v - best[c]) < 1e-12 else s_
                        if c != "mean":
                            vd = V.get((r[4](c), c))
                            s_ += "" if vd == "PASS" else (" ⚠" if vd == "FAIL" else " †")
                        cells.append(s_)
                res.append(f"| {r[0]} | {r[1]} | {r[2]} | " + " | ".join(cells) + " |")
            res.append("")
        return res

    out += tables(ROWS, BUDGET, f"## Fair comparison — every method trained {BUDGET} steps, same held-out set")
    # training cost from convergence.json (pure optimisation time = sum of per-step times; wall = incl. start-up / restarts)
    try:
        C = {(r["method"], r["clip"]): r for r in json.loads(CONV.read_text())}
    except Exception:
        C = {}
    out += ["## Training cost (one GH200 GPU per run; same 30k steps)", "",
            "Cell = optimisation time h (logged wall-clock h incl. start-up and chunk restarts) · s/step. Steps are NOT equal work: "
            "SplatAD = 1 camera image per step; NeuRAD / UniSim = ray batches (camera + LiDAR rays); EmerNeRF = ray batch; "
            "V1.3 = NeuRAD ray batch + physics terms.", "",
            "| method | " + " | ".join(CLIPS) + " | total optimisation h (3 clips) |", "|---|" + "---|" * (len(CLIPS) + 1)]
    for r in ROWS:
        cells, tot = [], 0.0
        for c in CLIPS:
            d = C.get((r[4](c), c))
            if d and d.get("train_compute_h") is not None:
                w = d.get("train_wall_h")
                cells.append(f"{d['train_compute_h']:.2f} h ({w:.2f} h) · {d['sec_per_step']:.3f} s" if w else f"{d['train_compute_h']:.2f} h · {d['sec_per_step']:.3f} s")
                tot += d["train_compute_h"]
            else:
                cells.append("n/a")
        out.append(f"| {r[0]} | " + " | ".join(cells) + f" | {tot:.2f} h |" if tot else f"| {r[0]} | " + " | ".join(cells) + " | n/a |")
    out.append("")
    out += tables(ROWS_5K, 0, "## Appendix (records) — our earlier FoggyField V1.3 lanes at 5k steps (NOT budget-matched; V1.3 paused 2026-11-25, baselines only)")
    a.out.write_text("\n".join(out) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
