#!/usr/bin/env python3
"""LiDAR metrics from the per-sweep npz dumps (method-agnostic; CPU only).

jobs/lidar_chamfer_eval.py is SplatAD-specific (it re-renders through nerfstudio's
eval_setup). This script computes the SAME metric definitions from the npz schema that
jobs/lidar_dump.py (SplatAD) and jobs/emernerf/lidar_dump_emernerf.py (EmerNeRF) write:

  per frame (P = pred_xyz, G = gt_xyz, Pm = pred_median_xyz; frames with empty P or G are
  skipped, as in lidar_chamfer_eval.py):
    cd_m              = 0.5 * (mean_p min_g |p-g| + mean_g min_p |g-p|)          [m]
    cd2_m2            = (sum_p min_g |p-g|^2 + sum_g min_p |g-p|^2) / |G|         [m^2]
    cd_median_pc_m    = cd_m computed with Pm instead of P
    gt_range_m        = mean_g |g|   (G is in the sensor frame, so |g| is the range)
    depth_median_l2_m = median |pred_range - gt_range| over returned rays
    depth_mean_l2_m   = mean   |pred_range - gt_range| over returned rays
  aggregate: the mean over frames of every per-frame value; rcd = mean(cd_m) / mean(gt_range_m).

These are line-for-line the formulas of lidar_chamfer_eval.py (cKDTree k=1 NN distances,
same normalisation, same aggregation). Two differences, both forced by the npz schema:
  * "returned rays" for the depth terms = entries with gt_range > 0 and a finite prediction.
    For an EmerNeRF dump every entry is a GT return, so this IS lidar_chamfer_eval's
    raster_pts_valid_depth_and_did_return set. For a SplatAD dump the raster also holds
    non-return cells that cannot be told apart in the npz, so depth_* may differ from
    lidar_chamfer_eval's numbers there (CD/CD2/RCD/cd_median_pc do not depend on this).
  * ray_drop_acc needs the GT ray-drop flags, which the npz does not store -> null.
Validation: --check-against <lidar_chamfer_eval json> compares the CD-family numbers.

--eval-ray-set returns_only  (EmerNeRF; benchmark protocol of jobs/lidar_dump_ns.py)
  The dump holds ONLY GT-returned rays (GT range in (0, 1000) m) with one predicted point per ray
  and no ray-drop prediction. The drop-independent metrics are then written, with the definition
  copied from jobs/lidar_dump_ns.py (frame_metrics, "SECONDARY, DROP-INDEPENDENT GEOMETRY METRIC"):
    R = GT-returned rays (0 < GT range < 1000 m);  P = {o + d*depth_pred : i in R};  G = {GT point_i : i in R}
    cd_returns_m              = 0.5 * (mean_p min_g |p-g| + mean_g min_p |g-p|)
    cd2_returns_m2            = (sum_p min_g |p-g|^2 + sum_g min_p |g-p|^2) / |G|
    depth_median_l2_returns_m = median_{i in R} |depth_pred_i - range_gt_i|      (metres, not squared)
    JSON value = plain mean of the per-sweep values.
  The drop-aware primary metrics (cd_m, cd2_m2, rcd, cd_median_pc_m, ray_drop_acc) are null: the
  model has no ray-drop head and no drop rule is applied. Extras: rcd_returns, cd_median_pc_returns_m.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def frame_metrics(d, returns_only: bool = False) -> dict | None:
    P, G, Pm = d["pred_xyz"].astype(np.float64), d["gt_xyz"].astype(np.float64), d["pred_median_xyz"].astype(np.float64)
    if returns_only:
        # every entry must be a GT return with its own predicted point (no drop filtering)
        if not (len(P) == len(G) == len(Pm) == len(d["gt_range"])):
            raise ValueError("returns_only needs one prediction per GT-returned ray (pred/gt/range lengths differ)")
        R = (d["gt_range"] > 0) & (d["gt_range"] < 1000.0)
        P, G, Pm = P[R], G[R], Pm[R]
        d = {"gt_range": d["gt_range"][R], "pred_range": d["pred_range"][R]}
    P, Pm = P[np.isfinite(P).all(1)], Pm[np.isfinite(Pm).all(1)]
    if len(P) == 0 or len(G) == 0:
        return None
    tg, tp = cKDTree(G), cKDTree(P)
    d_pg = tg.query(P, k=1, workers=16)[0]
    d_gp = tp.query(G, k=1, workers=16)[0]
    if len(Pm):
        d_pmg = tg.query(Pm, k=1, workers=16)[0]
        d_gpm = cKDTree(Pm).query(G, k=1, workers=16)[0]
        cdm = 0.5 * (d_pmg.mean() + d_gpm.mean())
    else:
        cdm = float("nan")
    gr, pr = d["gt_range"].astype(np.float64), d["pred_range"].astype(np.float64)
    ok = (gr > 0) & np.isfinite(pr)
    dd = np.abs(pr[ok] - gr[ok])
    return dict(
        n_pred=int(len(P)), n_gt=int(len(G)), gt_range_m=float(np.linalg.norm(G, axis=1).mean()),
        cd_m=float(0.5 * (d_pg.mean() + d_gp.mean())),
        cd2_m2=float(((d_pg ** 2).sum() + (d_gp ** 2).sum()) / len(G)),
        cd_median_pc_m=float(cdm),
        depth_median_l2_m=float(np.median(dd)) if len(dd) else float("nan"),
        depth_mean_l2_m=float(dd.mean()) if len(dd) else float("nan"),
        ray_drop_acc=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True, help="dir of frame_*.npz")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--method", default="")
    ap.add_argument("--clip", default="")
    ap.add_argument("--check-against", type=Path, default=None,
                    help="a lidar_chamfer_eval.py json for the SAME run; compare CD/CD2/RCD/cd_median_pc")
    ap.add_argument("--eval-ray-set", choices=["drop_aware", "returns_only"], default="drop_aware",
                    help="returns_only: write cd_returns_* (lidar_dump_ns.py definition), drop-aware keys null")
    a = ap.parse_args()
    ro = a.eval_ray_set == "returns_only"

    files = sorted(a.dump.glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no frame_*.npz under {a.dump}")
    rows = []
    for f in files:
        with np.load(f) as d:
            r = frame_metrics(d, ro)
        if r is None:
            continue
        r["frame"] = f.stem
        rows.append(r)
        print(f"  {f.stem}  cd {r['cd_m']:.4f}  cd2 {r['cd2_m2']:.4f}  depth_med {r['depth_median_l2_m']:.4f}", flush=True)
    keys = [k for k in rows[0] if k not in ("n_pred", "n_gt", "frame", "ray_drop_acc")]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    man = a.dump / "DUMP_MANIFEST.json"
    res = dict(
        method=a.method, clip=a.clip, dump=str(a.dump),
        dump_manifest=json.loads(man.read_text()) if man.is_file() else None,
        n_frames=len(rows), n_pred=int(sum(r["n_pred"] for r in rows)), n_gt=int(sum(r["n_gt"] for r in rows)),
        **agg, ray_drop_acc=None, rcd=agg["cd_m"] / agg["gt_range_m"],
        definitions="cd_m = symmetric mean NN distance (m); cd2_m2 = nerfstudio chamfer (sum sq NN both ways / |gt|); "
                    "rcd = cd_m / mean GT range; cd_median_pc_m uses the model's median-depth point cloud; "
                    "depth_*_l2_m = |pred_range - gt_range| over GT returns. Same formulas as jobs/lidar_chamfer_eval.py, "
                    "computed from the npz dump (jobs/emernerf/lidar_metrics_from_dump.py). ray_drop_acc: not computable "
                    "from the dump (null).",
        per_frame=rows,
    )
    if res["dump_manifest"]:
        res["dump_manifest"].pop("frames", None)
    if ro:
        # same numbers, protocol names: drop-independent -> *_returns_*, drop-aware -> null
        res.update(
            eval_ray_set="returns_only", has_ray_drop_head=False, ray_drop_rule=None,
            cd_returns_m=agg["cd_m"], cd2_returns_m2=agg["cd2_m2"],
            depth_median_l2_returns_m=agg["depth_median_l2_m"],
            rcd_returns=agg["cd_m"] / agg["gt_range_m"], cd_median_pc_returns_m=agg["cd_median_pc_m"],
            cd_m=None, cd2_m2=None, rcd=None, cd_median_pc_m=None, ray_drop_acc=None,
            definitions_returns="cd_returns_m / cd2_returns_m2 / depth_median_l2_returns_m: drop-independent, one predicted "
                                "point (rendered depth along the GT ray) for EVERY GT-returned ray (0 < range < 1000 m), no "
                                "ray-drop filtering; definition copied from jobs/lidar_dump_ns.py; plain mean over sweeps. "
                                "Drop-aware cd_m/cd2_m2/rcd/cd_median_pc_m/ray_drop_acc are null (no ray-drop head).",
        )
        for r in rows:
            r.update(cd_returns_m=r.pop("cd_m"), cd2_returns_m2=r.pop("cd2_m2"),
                     depth_median_l2_returns_m=r["depth_median_l2_m"], cd_median_pc_returns_m=r.pop("cd_median_pc_m"))
    if a.check_against:
        ref = json.loads(a.check_against.read_text())
        cmp = {k: {"dump": res[k], "lidar_chamfer_eval": ref.get(k), "rel_diff": (abs(res[k] - ref[k]) / abs(ref[k]) if ref.get(k) else None)}
               for k in ("cd_m", "cd2_m2", "rcd", "cd_median_pc_m", "gt_range_m", "depth_median_l2_m") if k in ref}
        res["check_against"] = {"file": str(a.check_against), "n_frames_ref": ref.get("n_frames"), "compare": cmp}
        print(json.dumps(res["check_against"], indent=2))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_frame", "dump_manifest", "check_against")}, indent=2))
    print("LIDAR_METRICS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
