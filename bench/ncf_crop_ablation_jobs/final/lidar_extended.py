#!/usr/bin/env python3
"""Extended LiDAR metrics from a per-sweep dump (CPU). Same ray set as the bench JSON: every GT-returned ray of the 99 held-out
sweeps, one predicted point per ray (rendered range along the GT ray; no ray-drop filtering = oracle drop).

  python3 lidar_extended.py --dump <dir with frame_*.npz> --out <json> [--label L --clip C]

Per sweep, then plain mean over sweeps:
  acc_m        mean NN distance pred -> GT (accuracy, m)          comp_m   mean NN distance GT -> pred (completeness, m)
  cd_m         (acc + comp) / 2  (= the bench's cd_returns_m)      cd2_m2   (sum d^2 both ways) / |GT|
  P/R/F@0.2    precision / recall / F-score at 0.2 m;  F@0.1, F@0.5 as well
  occ_iou@0.2  IoU of occupied 0.2 m voxels (pred vs GT points, both within 0.2-80 m of the sensor); occ_iou@0.5 likewise
  range_mae_m, range_median_m   |r_pred - r_gt| over the rays;  range_rmse_m
  int_rmse     RMSE of intensity / 255 over all returns;  int_rmse_unsat over rays with GT code < 255 (255 = saturated/censored)
"""
import argparse, glob, json, os
import numpy as np
from scipy.spatial import cKDTree

THR = (0.1, 0.2, 0.5)


def voxels(p, s):
    return set(map(tuple, np.floor(p / s).astype(np.int64)))


def sweep(z):
    if "gt_did_return" in z.files:   # lidar_dump_ns.py schema: arrays over ALL fired rays + return mask
        m = z["gt_did_return"].astype(bool)
        g = z["gt_xyz"].astype(np.float64)                       # GT points of the returned rays (sensor frame)
        gr, pr = z["gt_range"].astype(np.float64)[m], z["pred_range"].astype(np.float64)[m]
        u = g / np.linalg.norm(g, axis=1, keepdims=True)
        p = g + u * (pr - gr)[:, None]                            # rendered range along the GT ray (oracle drop)
        gi_all = {k: z[k][m] for k in ("gt_intensity", "pred_intensity") if k in z.files}
    else:
        g, p = z["gt_xyz"].astype(np.float64), z["pred_xyz"].astype(np.float64)
        gr, pr = z["gt_range"].astype(np.float64), z["pred_range"].astype(np.float64)
        gi_all = {k: z[k] for k in ("gt_intensity", "pred_intensity") if k in z.files}
    ok = np.isfinite(p).all(1) & np.isfinite(pr)
    g, p, gr, pr = g[ok], p[ok], gr[ok], pr[ok]
    dp, _ = cKDTree(g).query(p, workers=8)
    dg, _ = cKDTree(p).query(g, workers=8)
    o = {"acc_m": dp.mean(), "comp_m": dg.mean(), "cd_m": 0.5 * (dp.mean() + dg.mean()),
         "cd2_m2": (np.sum(dp ** 2) + np.sum(dg ** 2)) / len(g)}
    for t in THR:
        P, R = (dp < t).mean(), (dg < t).mean()
        o[f"P@{t}"], o[f"R@{t}"], o[f"F@{t}"] = P, R, (2 * P * R / (P + R) if P + R > 0 else 0.0)
    # occupancy IoU in the dump frame; points kept within 0.2-80 m of the sensor (range along the ray)
    org = z["lidar_to_world"][:, 3].astype(np.float64) if np.abs(np.linalg.norm(g, axis=1) - gr).mean() > 1.0 else np.zeros(3)
    # (V1.3 dumps: points in world frame -> sensor origin = lidar_to_world translation; ns dumps: sensor frame)
    gm = (gr > 0.2) & (gr < 80); pm = (np.linalg.norm(p - org, axis=1) > 0.2) & (np.linalg.norm(p - org, axis=1) < 80)
    for s in (0.2, 0.5):
        vg, vp = voxels(g[gm], s), voxels(p[pm], s)
        u = len(vg | vp)
        o[f"occ_iou@{s}"] = len(vg & vp) / u if u else 0.0
    e = np.abs(pr - gr)
    o.update(range_mae_m=e.mean(), range_median_m=np.median(e), range_rmse_m=np.sqrt(np.mean(e ** 2)))
    if len(gi_all) == 2:
        gi, pi = gi_all["gt_intensity"].astype(np.float64)[ok], gi_all["pred_intensity"].astype(np.float64)[ok]
        sc = 255.0 if max(gi.max(), pi.max()) > 1.5 else 1.0
        o["int_rmse"] = np.sqrt(np.mean(((pi - gi) / sc) ** 2))
        u = gi < (254.5 if sc == 255.0 else 0.999)
        o["int_rmse_unsat"] = np.sqrt(np.mean(((pi[u] - gi[u]) / sc) ** 2)) if u.any() else float("nan")
        o["int_saturated_frac"] = 1 - u.mean()
    o["n_rays"] = int(len(g))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--clip", default="")
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.dump, "*.npz")))
    assert fs, f"no npz in {a.dump}"
    rows = [sweep(np.load(f)) for f in fs]
    keys = [k for k in rows[0] if k != "n_rays"]
    res = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}
    res.update(n_sweeps=len(rows), n_rays=int(sum(r["n_rays"] for r in rows)), dump=a.dump, label=a.label, clip=a.clip,
               protocol="GT-returned rays of the held-out sweeps, one predicted point per ray (oracle drop); mean over sweeps",
               definitions=__doc__.split("Per sweep, then plain mean over sweeps:")[1].strip())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(a.label, a.clip, {k: round(res[k], 4) for k in ("cd_m", "F@0.2", "occ_iou@0.2", "int_rmse") if k in res})


if __name__ == "__main__":
    main()
