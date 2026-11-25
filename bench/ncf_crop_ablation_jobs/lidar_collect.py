#!/usr/bin/env python3
"""Collect the native-LiDAR benchmark JSONs into one summary (CPU only).

Reads   <results>/lidar/lidar_<method>_<variant>__<clip>.json   for every method/clip that exists
        (written by lidar_dump_ns.py; an EmerNeRF writer must use the same keys, see below)
Writes  <results>/LIDAR_SUMMARY.md  and  <results>/lidar_summary.json

    python jobs/lidar_collect.py --results results_bench

Tables: per clip; mean over the clear-weather NCF clips (a method's mean is only given when it has
ALL of them, otherwise it says which are missing); fog clips (NFF-*) reported separately, never averaged in.

Refusal rule: every row must have been evaluated on the SAME ray set. The expected set is
--ray-set (default "returns+range_image_nonreturns_v2"); a JSON whose `eval_ray_set` is missing or
different is NOT put in any table, it is listed under "Excluded" with the reason, and the exit code is 1.

Keys read: cd_m, cd2_m2, rcd, depth_median_l2_m, ray_drop_acc, ray_drop_acc_returns,
ray_drop_acc_nonreturns, cd_returns_m, cd2_returns_m2, depth_median_l2_returns_m, has_ray_drop_head,
eval_ray_set, n_frames, n_return, n_nonreturn, step.  A method with has_ray_drop_head false (or absent)
gets "—" in the ray-drop columns: its primary CD is then built with its fallback drop rule (or no drop
at all), and cd_returns_m is the like-for-like geometry comparison.
"""
import argparse, json, math, sys
from pathlib import Path

DEFAULT_SET = "returns+range_image_nonreturns_v2"
COLS = [  # (key, header, fmt, needs_ray_drop_head)
    ("cd_m", "CD (m)", "{:.4f}", False),
    ("cd2_m2", "CD2 (m²)", "{:.3f}", False),
    ("rcd", "RCD", "{:.4f}", False),
    ("depth_median_l2_m", "depth med L1 (m)", "{:.4f}", False),
    ("ray_drop_acc", "drop acc", "{:.4f}", True),
    ("ray_drop_acc_returns", "acc returns", "{:.4f}", True),
    ("ray_drop_acc_nonreturns", "acc non-ret", "{:.4f}", True),
    ("cd_returns_m", "CD returns (m)", "{:.4f}", False),
    ("cd2_returns_m2", "CD2 returns (m²)", "{:.3f}", False),
]


def fmt(v, f):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/results_bench"))
    ap.add_argument("--methods", default="splatad neurad unisim emernerf")
    ap.add_argument("--variant", default="safe")
    ap.add_argument("--clips", default="NCF-02 NCF-07 NCF-10 NCF-15", help="clear-weather clips (averaged)")
    ap.add_argument("--fog-clips", default="NFF-05", help="reported separately, never averaged in")
    ap.add_argument("--ray-set", default=DEFAULT_SET, help="the eval_ray_set every row must have")
    a = ap.parse_args()
    methods, clips, fog = a.methods.split(), a.clips.split(), a.fog_clips.split()
    lid = a.results / "lidar"

    rows, excluded = {}, []
    for m in methods:
        for c in clips + fog:
            p = lid / f"lidar_{m}_{a.variant}__{c}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rs = d.get("eval_ray_set")
            if rs != a.ray_set:
                excluded.append(dict(method=m, clip=c, file=str(p), reason=f"eval_ray_set={rs!r} != {a.ray_set!r}"))
                continue
            head = bool(d.get("has_ray_drop_head", False))
            r = {k: d.get(k) for k, *_ in COLS}
            for k, _, _, needs in COLS:
                if needs and not head:
                    r[k] = None
            r.update(has_ray_drop_head=head, n_frames=d.get("n_frames"), n_return=d.get("n_return"),
                     n_nonreturn=d.get("n_nonreturn"), step=d.get("step"),
                     depth_median_l2_returns_m=d.get("depth_median_l2_returns_m"), file=str(p))
            rows[(m, c)] = r

    present = [m for m in methods if any((m, c) in rows for c in clips + fog)]
    means = {}
    for m in present:
        have = [c for c in clips if (m, c) in rows]
        miss = [c for c in clips if c not in have]
        mm = {"n_clips": len(have), "missing": miss}
        for k, *_ in COLS:
            vals = [rows[(m, c)][k] for c in have if rows[(m, c)][k] is not None]
            mm[k] = (sum(vals) / len(vals)) if (vals and not miss) else None
        means[m] = mm

    hdr = "| method | clip | " + " | ".join(h for _, h, _, _ in COLS) + " | sweeps |"
    sep = "|" + "---|" * (len(COLS) + 3)
    no_head = [m for m in present if not all(rows[(m, c)]["has_ray_drop_head"] for c in clips + fog if (m, c) in rows)]
    def line(m, c, r):
        star = "" if r["has_ray_drop_head"] else " ¹"
        return f"| {m}{star} | {c} | " + " | ".join(fmt(r[k], f) for k, _, f, _ in COLS) + f" | {r['n_frames']} |"

    md = [f"# Native-LiDAR benchmark ({a.variant})", "",
          f"Eval ray set (every row): `{a.ray_set}` — real returns + real non-returns from the range image, "
          "the same held-out sweeps (nvs_50_50) for every method; LiDAR in the sensor frame.", "",
          "CD = symmetric mean NN distance; CD2 = nerfstudio chamfer (sum of squared NN both ways / |GT|); "
          "RCD = CD / mean GT range; depth med L1 = median |pred − GT| range on returned rays (metres, not squared); "
          "drop acc = ray-drop accuracy over all rays (non-returns are the drop class), per class in the next two columns; "
          "CD returns = drop-independent CD: one predicted point for EVERY GT-returned ray (definition in lidar_dump_ns.py).", ""]
    md += ["## Per clip", "", hdr, sep]
    for c in clips + fog:
        for m in present:
            if (m, c) in rows:
                md.append(line(m, c, rows[(m, c)]))
    md += ["", f"## Mean over the clear-weather clips ({', '.join(clips)})", "",
           "| method | " + " | ".join(h for _, h, _, _ in COLS) + " | clips |", "|" + "---|" * (len(COLS) + 2)]
    for m in present:
        mm = means[m]
        star = " ¹" if m in no_head else ""
        tail = f"{mm['n_clips']}/{len(clips)}" + (f" (missing {', '.join(mm['missing'])})" if mm["missing"] else "")
        md.append(f"| {m}{star} | " + " | ".join(fmt(mm[k], f) for k, _, f, _ in COLS) + f" | {tail} |")
    if fog:
        md += ["", f"## Fog ({', '.join(fog)}) — reported separately, not in the mean", "", hdr, sep]
        for c in fog:
            for m in present:
                if (m, c) in rows:
                    md.append(line(m, c, rows[(m, c)]))
    md += ["", "¹ no trained ray-drop head: ray-drop metrics are not reported (—). Its primary CD/CD2 use the "
           "method's fallback return rule (rendered depth < 150 m for neurad-studio models), so compare it on "
           "**CD returns**, which is independent of any drop decision."]
    mism = []
    for c in clips + fog:
        nf = {m: rows[(m, c)]["n_frames"] for m in present if (m, c) in rows}
        if len(set(nf.values())) > 1:
            mism.append(f"- {c}: held-out sweep counts differ between methods {nf} -- NOT the same sweeps (smoke / max_eval_frames run?)")
    if mism:
        md += ["", "## WARNING: sweep sets differ", ""] + mism
    if excluded:
        md += ["", "## Excluded (different eval ray set — NOT comparable, not in any table)", ""]
        md += [f"- {e['method']} / {e['clip']}: {e['reason']}  (`{e['file']}`)" for e in excluded]
    out_md = a.results / "LIDAR_SUMMARY.md"
    out_md.write_text("\n".join(md) + "\n")
    summary = dict(eval_ray_set=a.ray_set, variant=a.variant, clips=clips, fog_clips=fog,
                   per_clip={f"{m}/{c}": r for (m, c), r in rows.items()}, mean_clear=means, excluded=excluded,
                   sweep_count_mismatch=mism)
    (a.results / "lidar_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(md))
    print(f"\nwrote {out_md} and {a.results / 'lidar_summary.json'}")
    if excluded:
        print(f"!! {len(excluded)} result(s) EXCLUDED for a different eval_ray_set", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
