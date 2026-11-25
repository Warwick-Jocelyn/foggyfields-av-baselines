#!/usr/bin/env python3
"""Convergence / sanity validation of every finished final-ablation row (CPU, login node). Nothing is reported as a result
unless it passes here.

  python3 convergence_check.py --out-md <CONVERGENCE.md> --out-json <convergence.json>

Checks per (method, clip):
  C1 finished      last logged train step >= 98 % of the schedule
  C2 finite        no NaN / inf in the training loss
  C3 plateau       median train loss of the last 10 % of steps vs the 70-80 % window: relative drop <= 15 %
                   (still converging slowly is fine; a big drop = stopped too early). Also last-10 % < first-10 % (it learned).
  C4 gap           train PSNR (median of the last 10 % of logged train-batch PSNR) - held-out PSNR (common evaluator)
                   in [-1, +8] dB (outside = overfitting / train-eval mismatch / a broken eval).
  C5 not grey      held-out renders vs GT on 21 images (3 per camera): contrast ratio std(render)/std(GT) >= 0.70 and
                   detail ratio mean|grad render| / mean|grad GT| >= 0.50 (the SplatAD opacity-collapse failure gives
                   grey blobs: both ratios collapse). Reported as medians.
Sources: nerfstudio tfevents (SplatAD / NeuRAD / UniSim / V1.3; tags 'Train Loss', 'Train Metrics Dict/psnr'), EmerNeRF
lane log lines '[step/25001] ... psnr: x (avg) total_pixel_loss: y'.
"""
import argparse, json, math, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from tfevents import read_scalars  # noqa: E402

BW = Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation")
O, R = BW / "outputs_final", BW / "results_final"
CLIPS = ["NFF-02", "NFF-08", "NFF-10", "NFF-05"]
V13CFG = {  # label -> clip -> run dir (config.yml parent); from RESULTS_LANES.md / v13_queue.txt
}


def v13_runs():
    out = {}
    q = BW / "jobs/final/v13_queue.txt"
    lanes = Path("/users/yitiwang/projects/foggyfields-v1/reports/v13_overnight/final_ablation_2026-11-23/RESULTS_LANES.md")
    txt = (q.read_text() if q.is_file() else "") + "\n" + (lanes.read_text() if lanes.is_file() else "")
    for cfg in sorted(set(re.findall(r"(/capstor/\S+?/config\.yml)", txt))):
        m = re.search(r"/(V13-[A-Za-z0-9]+|A0)/", cfg)
        c = re.search(r"nff-?(\d\d)", cfg, re.I)
        if m and c and Path(cfg).is_file():
            out[(m.group(1), f"NFF-{c.group(1)}")] = Path(cfg).parent
    return out


def rows():
    r = []
    for c in CLIPS:
        for meth, tag, sub, it in (("splatad", "allfix", "splatad", 30000), ("splatad-opreg5e4", "opreg5e4", "splatad", 30000),
                                   ("neurad", "std", "neurad", 30000), ("unisim", "std", "unisim", 30000), ("emernerf", "std", "emernerf", 30000)):
            rd = O / f"{tag}/{sub}/safe/{c}"
            r.append((meth, tag, c, rd, it, R / f"{tag}/metrics/metrics_{sub}__safe__{c}.json", rd / "renders/val"))
    for (lb, c), rd in sorted(v13_runs().items()):
        r.append((f"v13-{lb}", "v13", c, rd, 5000, R / f"v13/metrics/metrics_v13__{lb}__{c}.json", O / f"v13/{lb}__{c}/renders/val"))
    OV = Path("/capstor/scratch/cscs/yitiwang/foggyfields-v1/v13_overnight")
    for c in CLIPS:   # fair-budget 30k lanes of our full models
        for L in ("F", "Fc"):
            lb = f"V13-{L}30k"
            r.append((f"v13-{lb}", "v13", c, OV / f"20261123_{c}_v13_final30k/train/V13-{L}", 30000,
                      R / f"v13/metrics/metrics_v13__{lb}__{c}.json", O / f"v13/{lb}__{c}/renders/val"))
    return r


EM_RE = re.compile(r"\[(\d+)/(\d+)\].*?\bpsnr: ([-\d.eE+naninf]+) \(.*?total_pixel_loss: ([-\d.eE+naninf]+) .*?\btime: ([\d.eE+-]+) ")


def run_start(run):
    """mtime of the marker written when the CURRENT configuration started from scratch (older segments are ignored)."""
    ts = [p.stat().st_mtime for p in (run / "TRAIN_START", run / "TRAIN_SIG") if p.is_file()]
    return min(ts) - 120 if ts else 0.0


def curves(meth, run, log, iters):
    """-> (steps, loss, psnr_steps, psnr, sec_per_step) of the current training only"""
    if meth == "emernerf":
        best = {}
        if log.is_file():
            for ln in log.read_text(errors="replace").splitlines():
                m = EM_RE.search(ln)
                if m and abs(int(m.group(2)) - iters) <= 1:          # only the run with this schedule length
                    best[int(m.group(1))] = (float(m.group(4)), float(m.group(3)), float(m.group(5)))
        st = np.array(sorted(best)); ls = np.array([best[s][0] for s in st]); ps = np.array([best[s][1] for s in st])
        tm = np.array([best[s][2] for s in st])
        return st, ls, st, ps, (float(np.median(tm)) if len(tm) else float("nan"))
    t0 = run_start(run)
    ev = sorted(p for p in run.rglob("events.out.tfevents*") if p.stat().st_mtime >= t0)
    if not ev:
        return np.array([]), np.array([]), np.array([]), np.array([]), float("nan")
    s = read_scalars(ev)
    L = s.get("Train Loss", []); P = s.get("Train Metrics Dict/psnr", []); T = s.get("Train Iter (time)", [])
    sps = float(np.median([x[2] for x in T])) if T else float("nan")
    return (np.array([x[0] for x in L]), np.array([x[2] for x in L]), np.array([x[0] for x in P]), np.array([x[2] for x in P]), sps)


def wall_hours(meth, tag, clip, run):
    """logged wall-clock of the training stage of the current run (incl. start-up and restarts of resumed chunks)."""
    import csv
    t0 = run_start(run); tot = 0.0
    if tag == "v13":
        tl = run.parents[1] / "logs/timings.txt"
        L = run.name.replace("V13-", "")
        if tl.is_file():
            for ln in tl.read_text().splitlines():
                m = re.match(rf"V13-{re.escape(L)} train_to=\d+ from=\d+ rc=\d+ sec=(\d+)", ln)
                if m:
                    tot += float(m.group(1))
        return tot / 3600 if tot else None
    for f in (R / tag).glob("timings*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r.get("method") == meth.split("-")[0] and r.get("clip") == clip and r.get("stage") == "train":
                    try:
                        ts = time.mktime(time.strptime(r["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
                    except Exception:
                        continue
                    if ts >= t0:
                        tot += float(r.get("seconds") or 0)
    return tot / 3600 if tot else None


def window(st, v, a, b, n):
    m = (st >= a * n) & (st < b * n)
    return float(np.median(v[m])) if m.any() else float("nan")


def grey_check(rend, k=3):
    rgb, gt = rend / "rgb", rend / "gt-rgb"
    if not rgb.is_dir() or not gt.is_dir():
        return None
    cr, dr = [], []
    for cam in sorted(p.name for p in rgb.iterdir() if p.is_dir()):
        fr = sorted((rgb / cam).glob("*.png"))
        for f in [fr[int(i)] for i in np.linspace(0, len(fr) - 1, k)] if fr else []:
            g = gt / cam / f.name
            if not g.is_file():
                continue
            a = np.asarray(Image.open(f).convert("L"), np.float32) / 255
            b = np.asarray(Image.open(g).convert("L"), np.float32) / 255
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]); a, b = a[:h, :w], b[:h, :w]
            cr.append(a.std() / max(b.std(), 1e-6))
            ga = np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
            gb = np.abs(np.diff(b, axis=0)).mean() + np.abs(np.diff(b, axis=1)).mean()
            dr.append(ga / max(gb, 1e-6))
    if not cr:
        return None
    return {"contrast_ratio": float(np.median(cr)), "detail_ratio": float(np.median(dr)), "n_images": len(cr)}


def check(meth, tag, clip, run, iters, metp, rend):
    met = json.loads(metp.read_text()) if metp.is_file() else None
    if met is None:
        return None
    log = BW / f"logs_final/{tag}/lanes/{meth.split('-')[0]}__safe__{clip}.log"
    if int(met.get("iters") or iters) < iters - 1:      # e.g. the old 20k UniSim metrics: not this budget's result
        return None
    st, ls, pst, ps, sps = curves(meth.split("-")[0] if meth.startswith("splatad") else meth, run, log, iters)
    res = {"method": meth, "variant": tag, "clip": clip, "iters": iters, "heldout_psnr": met.get("primary", {}).get("psnr"),
           "sec_per_step": sps, "train_compute_h": (sps * iters / 3600) if math.isfinite(sps) else None,
           "train_wall_h": wall_hours(meth, tag, clip, run)}
    checks = {}
    if len(st):
        n = iters
        checks["C1_finished"] = {"ok": bool(st.max() >= 0.98 * n - 100), "last_step": int(st.max())}
        checks["C2_finite"] = {"ok": bool(np.all(np.isfinite(ls))), "n_points": int(len(ls))}
        l_last, l_prev, l_first = window(st, ls, 0.9, 1.01, n), window(st, ls, 0.7, 0.8, n), window(st, ls, 0.0, 0.1, n)
        drop = (l_prev - l_last) / l_prev if l_prev and math.isfinite(l_prev) else float("nan")
        checks["C3_plateau"] = {"ok": bool(math.isfinite(drop) and drop <= 0.15 and l_last < l_first),
                                "loss_first10": l_first, "loss_70_80": l_prev, "loss_last10": l_last, "rel_drop_last": drop}
        if len(pst):
            tp = window(pst, ps, 0.9, 1.01, n)
            gap = tp - res["heldout_psnr"] if res["heldout_psnr"] is not None and math.isfinite(tp) else float("nan")
            checks["C4_gap"] = {"ok": bool(math.isfinite(gap) and -1.0 <= gap <= 8.0), "train_psnr_last10": tp, "gap_db": gap}
        else:
            checks["C4_gap"] = {"ok": None, "note": "no train PSNR logged"}
    else:
        checks["C1_finished"] = {"ok": None, "note": f"no training log found under {run} / {log}"}
    g = grey_check(rend)
    checks["C5_not_grey"] = ({"ok": bool(g["contrast_ratio"] >= 0.70 and g["detail_ratio"] >= 0.50), **g} if g
                             else {"ok": None, "note": "no renders/val"})
    oks = [c["ok"] for c in checks.values()]
    res["checks"] = checks
    res["verdict"] = "PASS" if all(o is True for o in oks) else ("FAIL" if any(o is False for o in oks) else "INCOMPLETE")
    return res


def f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    a = ap.parse_args()
    out = [r for r in (check(*x) for x in rows()) if r]
    a.out_json.write_text(json.dumps(out, indent=1))
    L = [f"# Convergence / sanity validation of every scored row ({time.strftime('%Y-%m-%d %H:%M')}, jobs/final/convergence_check.py)", "",
         "C1 finished (≥98 % of schedule) · C2 finite loss · C3 plateau (median loss last 10 % vs 70–80 % window drops ≤15 %, and below the first 10 %) · "
         "C4 train−held-out PSNR gap in [−1, +8] dB · C5 not grey/blurred (held-out render/GT contrast ratio ≥0.70 and detail ratio ≥0.50, 21 images).", "",
         "| method | clip | verdict | last step | loss first10 → 70-80 → last10 (rel. drop) | train PSNR | held-out PSNR | gap dB | contrast | detail | s/step | train compute h | train wall h |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        c = r["checks"]; c3 = c.get("C3_plateau", {}); c4 = c.get("C4_gap", {}); c5 = c.get("C5_not_grey", {})
        bad = [k for k, v in c.items() if v.get("ok") is False]
        v = r["verdict"] + (f" ({', '.join(b.split('_')[0] for b in bad)})" if bad else "")
        L.append(f"| {r['method']} | {r['clip']} | {v} | {f(c.get('C1_finished', {}).get('last_step'))} | "
                 f"{f(c3.get('loss_first10'), 4)} → {f(c3.get('loss_70_80'), 4)} → {f(c3.get('loss_last10'), 4)} ({f(c3.get('rel_drop_last'), 2)}) | "
                 f"{f(c4.get('train_psnr_last10'), 2)} | {f(r['heldout_psnr'], 2)} | {f(c4.get('gap_db'), 2)} | "
                 f"{f(c5.get('contrast_ratio'), 2)} | {f(c5.get('detail_ratio'), 2)} | {f(r.get('sec_per_step'), 3)} | "
                 f"{f(r.get('train_compute_h'), 2)} | {f(r.get('train_wall_h'), 2)} |")
    a.out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
