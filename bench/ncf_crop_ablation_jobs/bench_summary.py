#!/usr/bin/env python3
"""Cross-method summary of the safe_crop bench (NeuRAD / UniSim / EmerNeRF, + SplatAD as RGB reference).

Reads
  <bench>/metrics/metrics_<m>__safe__<clip>.json      RGB (eval_common_region.py), primary = safe_crop
  <bench>/lidar/lidar_<m>_safe__<clip>.json           LiDAR (lidar_dump_ns.py / emernerf lidar_metrics_from_dump.py)
  <bench>/timings*.csv                                stage wall-clock per segment (killed segments included)
  SplatAD reference: results/metrics (NCF) and results_nff/metrics (NFF-05), same split / script / region
Writes <bench>/BENCH_SUMMARY.md and <bench>/bench_summary.json

    python jobs/bench_summary.py --bench results_bench

Means are over the four clear-weather NCF clips; NFF-05 (night + fog) is reported on its own, never averaged in.
Every row is checked: n_images == 693 and the full iteration budget in run_signature; a row that fails is
listed under "Excluded" and the exit code is 1.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

W = Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation")
NCF = ["NCF-02", "NCF-07", "NCF-10", "NCF-15"]
FOG = ["NFF-05"]
METHODS = ["neurad", "unisim", "emernerf"]
ITERS = {"splatad": 30000, "neurad": 30000, "unisim": 20001, "emernerf": 25000}
N_IMAGES = 693
LIDAR_SET = "returns+range_image_nonreturns_v2"


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def iters_of(sig):
    m = re.search(r"iters=(\d+)", sig or "")
    return int(m.group(1)) if m else None


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=W / "results_bench")
    a = ap.parse_args()
    excluded = []

    # ---- RGB --------------------------------------------------------------------------------
    rgb = {}
    for m in ["splatad"] + METHODS:
        for c in NCF + FOG:
            if m == "splatad":
                p = W / ("results_nff" if c in FOG else "results") / "metrics" / f"metrics_splatad__safe__{c}.json"
            else:
                p = a.bench / "metrics" / f"metrics_{m}__safe__{c}.json"
            d = load(p)
            if d is None:
                excluded.append(f"{m}/{c}: no RGB metrics ({p})")
                continue
            it, n = iters_of(d.get("run_signature")), d.get("n_images")
            if it != ITERS[m] or n != N_IMAGES:
                excluded.append(f"{m}/{c}: RGB iters={it} n_images={n} (want {ITERS[m]}/{N_IMAGES})")
                continue
            rgb[(m, c)] = dict(d["primary"], file=str(p))

    # ---- LiDAR ------------------------------------------------------------------------------
    lid = {}
    for m in METHODS:
        for c in NCF + FOG:
            p = a.bench / "lidar" / f"lidar_{m}_safe__{c}.json"
            d = load(p)
            if d is None:
                excluded.append(f"{m}/{c}: no LiDAR json")
                continue
            want_set = "returns_only" if m == "emernerf" else LIDAR_SET
            if d.get("eval_ray_set") != want_set or d.get("n_frames") != 99:
                excluded.append(f"{m}/{c}: LiDAR eval_ray_set={d.get('eval_ray_set')} n_frames={d.get('n_frames')}")
                continue
            if not d.get("has_ray_drop_head", False):   # (UniSim, EmerNeRF) -- same rule as lidar_collect.py
                d = dict(d, **{k: None for k in ("ray_drop_acc", "ray_drop_acc_returns", "ray_drop_acc_nonreturns")})
            lid[(m, c)] = d

    # ---- compute ----------------------------------------------------------------------------
    sec = defaultdict(float)
    for t in ["timings.csv", "timings_emernerf.csv"]:
        p = a.bench / t
        if p.exists():
            for r in csv.DictReader(p.open()):
                if r["variant"] == "safe" and r["method"] in METHODS:
                    sec[(r["method"], r["clip"], r["stage"])] += float(r["seconds"])

    L = ["# safe_crop cross-method bench — summary", "",
         "All runs: safe_crop dataparser (`apply_safe_crop`), nvs_50_50 split (train even frames, eval the 99 odd",
         "frames x 7 cameras = 693 images per clip), full iteration budget (NeuRAD 30k, UniSim 20001, EmerNeRF 25k;",
         "SplatAD reference 30k from the crop ablation). RGB metrics on the safe_crop rectangle (identical pixel set",
         "for every method). Means over the 4 clear-weather NCF clips; NFF-05 (night + fog) separate.", ""]

    def rgb_table(clips, title):
        L.extend([f"## RGB — {title}", "",
                  "| method | " + " | ".join(f"{c} PSNR / SSIM / LPIPS" for c in clips) +
                  (" | **NCF mean** PSNR / SSIM / LPIPS |" if len(clips) > 1 else " |"),
                  "|---" * (len(clips) + 1 + (len(clips) > 1)) + "|"])
        for m in ["splatad"] + METHODS:
            cells = []
            for c in clips:
                r = rgb.get((m, c))
                cells.append("—" if r is None else f"{r['psnr']:.2f} / {r['ssim']:.3f} / {r['lpips']:.3f}")
            if len(clips) > 1:
                rs = [rgb.get((m, c)) for c in clips]
                cells.append("—" if None in rs else "**" + " / ".join(
                    f"{mean([r[k] for r in rs]):.{nd}f}" for k, nd in (("psnr", 2), ("ssim", 3), ("lpips", 3))) + "**")
            lab = "SplatAD (ref)" if m == "splatad" else m
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")

    rgb_table(NCF, "clear weather (NCF)")
    rgb_table(FOG, "night + fog (NFF-05)")
    L.extend(["SplatAD NFF-05: the model collapsed (renders are a grey blur, see the montage); its numbers are",
              "listed for completeness only. PSNR on NFF-05 is inflated for every method by the large dark,",
              "low-texture sky/fog area — compare methods within a clip, not across clips.", ""])

    LCOL = [("cd_returns_m", "CD returns (m)↓", 3), ("cd2_returns_m2", "CD2 returns (m²)↓", 2),
            ("depth_median_l2_returns_m", "depth median |err| (m)↓", 3)]
    DCOL = [("cd_m", "CD (m)↓", 3), ("cd2_m2", "CD2 (m²)↓", 2), ("rcd", "RCD↓", 4),
            ("ray_drop_acc", "ray-drop acc↑", 3), ("ray_drop_acc_returns", "acc returns↑", 3),
            ("ray_drop_acc_nonreturns", "acc non-returns↑", 3)]

    def lidar_table(cols, methods, title, note):
        L.extend([f"## LiDAR — {title}", "", note, "",
                  "| method | clip | " + " | ".join(h for _, h, _ in cols) + " |",
                  "|---|---|" + "---|" * len(cols)])
        for m in methods:
            for c in NCF + ["NCF mean"] + FOG:
                if c == "NCF mean":
                    ds = [lid.get((m, x)) for x in NCF]
                    if None in ds:
                        continue
                    vals = [mean([d.get(k) for d in ds]) for k, _, _ in cols]
                    L.append(f"| **{m}** | **NCF mean** | " + " | ".join(f"**{f(v, nd)}**" for v, (_, _, nd) in zip(vals, cols)) + " |")
                    continue
                d = lid.get((m, c))
                if d is not None:
                    L.append(f"| {m} | {c} | " + " | ".join(f(d.get(k), nd) for k, _, nd in cols) + " |")
        L.append("")

    lidar_table(LCOL, METHODS, "geometry on returned rays (all three methods, like-for-like)",
                "One predicted point per GT-returned ray (rendered depth along the GT ray), no ray-drop filtering: "
                "the only LiDAR numbers comparable across all three methods. 99 held-out sweeps per clip.")
    lidar_table(DCOL, ["neurad", "unisim"], "drop-aware CD and ray-drop accuracy (NeuRAD; UniSim CD only)",
                f"Ray set `{LIDAR_SET}` (GT returns + range-image non-returns). Only NeuRAD has a trained ray-drop "
                "head; UniSim (paper config, no ray-drop modelling) has its CD/CD2/RCD built with the neurad-studio "
                "fallback return rule (rendered depth < 150 m) and no ray-drop accuracy, so compare it on CD returns "
                "above. EmerNeRF has no ray-drop head at all and is not in this table.")

    L.extend(["## Compute (one GH200 per lane)", "",
              "| method | clip | train (h) | render (min) | train ms/it |", "|---|---|---|---|---|"])
    comp = {}
    for m in METHODS:
        for c in NCF + FOG:
            tr, rd = sec.get((m, c, "train"), 0.0), sec.get((m, c, "render"), 0.0)
            comp[(m, c)] = dict(train_s=tr, render_s=rd)
            L.append(f"| {m} | {c} | {tr / 3600:.2f} | {rd / 60:.0f} | {1000 * tr / ITERS[m]:.0f} |")
    L.extend(["", "Train time sums every segment (wall-clock-killed segments included, so it slightly over-counts "
              "the steps re-done after a resume; model loading included).", ""])

    if excluded:
        L.extend(["## Excluded", ""] + [f"- {e}" for e in excluded] + [""])
    out = a.bench / "BENCH_SUMMARY.md"
    out.write_text("\n".join(L))
    js = dict(rgb={f"{m}/{c}": v for (m, c), v in rgb.items()},
              lidar={f"{m}/{c}": {k: v for k, v in d.items() if not isinstance(v, (list, dict))} for (m, c), d in lid.items()},
              compute={f"{m}/{c}": v for (m, c), v in comp.items()}, excluded=excluded)
    (a.bench / "bench_summary.json").write_text(json.dumps(js, indent=2))
    print(f"wrote {out}" + (f"  ({len(excluded)} excluded)" if excluded else ""))
    return 1 if excluded else 0


if __name__ == "__main__":
    sys.exit(main())
