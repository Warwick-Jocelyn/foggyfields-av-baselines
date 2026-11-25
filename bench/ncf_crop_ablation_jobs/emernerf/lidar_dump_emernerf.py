#!/usr/bin/env python3
"""EmerNeRF counterpart of jobs/lidar_dump.py: one npz per HELD-OUT LiDAR sweep, same keys,
same frame, so jobs/lidar_viz.py, jobs/lidar_3d.py and lidar_metrics_from_dump.py read it
unchanged.

Rays = the GT returns of the sweep (every point in lidar/sweep_TTTT.npy, no range cut --
SplatAD's dump keeps them all too). For each return p (LiDAR SENSOR frame, at the point's
own capture time spin_centre + t_offset_us; the sweeps are not deskewed):
    T_i    = world <- sensor at that time = ls.ego_to_world_at(t, t_offset) @ T_lidar_rig
             (E3, lidar_pose_time="point"; "sweep" = the spin-centre pose for every ray)
    origin = T_i translation
    dir    = R(T_i) @ p / |p|
the same per-ray pose the training loader used (shared code, and asserted below).
EmerNeRF renders expected depth and median depth along that unit ray at the sweep's
normalised timestamp; depth is a range in metres because the direction is unit length.

    gt_xyz          = p                                   (sensor frame, all returns)
    pred_xyz        = p/|p| * expected_depth              (same rays, same frame)
    pred_median_xyz = p/|p| * median_depth
    lidar_to_world  = sensor -> clip-canonical world (3x4) at the SPIN CENTRE, = inv(T_rig_world[start]) @
                      T_rig_world[t] @ T_lidar_rig. With per-point poses it is approximate
                      for points away from the spin centre (up to v * 50 ms); the metrics
                      are computed in the sensor frame and do not use it.

Frame: provenance.json "lidar_frame" (via the dataset's resolve_lidar_frame): "ego" points
are mapped to the sensor frame with inv(T_lidar_rig); "sensor" points are used as stored.
The world rays built here are asserted to coincide with the ones the EmerNeRF training
loader built for the same sweep (max deviation printed), so the dump is in exactly the
frame the model was trained in. This is also the frame of SplatAD's gt_xyz (verified on
NCF-15 frame 51: identical points, max |diff| 2.8e-5 m).

Quantities EmerNeRF does not model are written as NaN, never invented:
    pred_intensity       NaN  (no intensity head)
    pred_ray_drop_logit  NaN  (no ray-drop head; prediction is restricted to GT returns)
The "raster" arrays (azimuth, elevation, gt_range, ...) are flat per-return lists, not a
sensor raster: raster_shape = [N]. azimuth/elevation are in degrees in the sensor frame,
the convention SplatAD's raster_pts use and lidar_viz.py re-bins from.
Extra keys (ignored by the existing consumers): pred_opacity, normed_timestamp, timestep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emer_common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-eval-frames", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=65536)
    a = ap.parse_args()

    cfg, ds, model, est, props, ckpt, step = emer_common.load_run(a.run_dir, a.ckpt, a.chunk)
    import torch
    from radiance_fields.render_utils import render_rays

    ls = ds.lidar_source
    if ls is None:
        raise SystemExit("run has no LiDAR source (data.lidar_source.load_lidar=false)")
    clip_root = Path(ds.data_path)
    all_held, todo = emer_common.heldout_timesteps(ds, a.max_eval_frames)
    T_lr = ls.lidar_to_ego.astype(np.float64)            # rig <- lidar
    inv_T_lr = np.linalg.inv(T_lr)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"lidar frame (provenance) = {ls.points_frame}; {len(todo)} of {len(all_held)} held-out sweeps")

    log, t_total = [], 0.0
    with torch.no_grad():
        for t_abs in todo:
            t = t_abs - ds.start_timestep
            raw = np.load(clip_root / "lidar" / f"sweep_{t_abs:04d}.npy").astype(np.float64)
            xyz = raw[:, :3]
            if ls.points_frame == "ego":
                p_s = xyz @ inv_T_lr[:3, :3].T + inv_T_lr[:3, 3]
            elif ls.points_frame == "sensor":
                p_s = xyz
            else:
                raise ValueError(ls.points_frame)
            r = np.linalg.norm(p_s, axis=1)
            keep = r > 1e-3
            p_s, r, inten, toff = p_s[keep], r[keep], raw[keep, 3], raw[keep, 4]
            d_s = p_s / r[:, None]
            l2w = ls.lidar_to_worlds[t].double().cpu().numpy()   # spin-centre pose (npz bookkeeping)
            # Per-ray pose, exactly as the training loader posed it (ls.ego_to_world_at):
            # lidar_pose_time="point" -> ego pose at spin_centre + t_offset_us (E3);
            # "sweep" -> the spin-centre pose for every ray (pre-fix).
            l2w_i = ls.ego_to_world_at(t, toff) @ T_lr                 # (n, 4, 4) world <- sensor
            o = l2w_i[:, :3, 3]
            d_w = np.einsum("nij,nj->ni", l2w_i[:, :3, :3], d_s)

            # --- the training loader's rays for this sweep must be the same rays -------
            m = ls.timesteps == t
            lo = ls.origins[m].double().cpu().numpy()
            lp = lo + ls.directions[m].double().cpu().numpy() * ls.ranges[m].double().cpu().numpy()
            mine = o + d_w * r[:, None]
            sel = (r > max(float(ls.data_cfg.truncated_min_range or 0.0), 0.5)) & (r < float(ls.data_cfg.truncated_max_range))
            if abs(int(sel.sum()) - len(lp)) > 20:   # float32 vs float64 at the 0.5 m / max-range cut
                raise AssertionError(f"t={t_abs}: {sel.sum()} rays in range here vs {len(lp)} in the training loader")
            from scipy.spatial import cKDTree
            dist, nn = cKDTree(mine).query(lp, k=1, workers=16)
            dev = float(dist.max())
            odev = float(np.abs(lo - o[nn]).max())
            if dev > 5e-3 or odev > 1e-4:
                raise AssertionError(f"t={t_abs}: dump rays deviate from training rays by {dev:.4f} m (origin {odev:.5f} m)")
            nts = torch.unique(ls.normalized_timestamps[m])
            if nts.numel() != 1:
                raise AssertionError(f"t={t_abs}: {nts.numel()} distinct normalised timestamps")

            n = len(r)
            data = {
                "lidar_origins": torch.from_numpy(np.ascontiguousarray(o)).float().cuda(),
                "lidar_viewdirs": torch.from_numpy(d_w).float().cuda(),
                "lidar_normed_timestamps": torch.full((n,), float(nts.item()), device="cuda"),
            }
            torch.cuda.synchronize(); t0 = time.time()
            res = render_rays(radiance_field=model, proposal_estimator=est, proposal_networks=props,
                              data_dict=data, cfg=cfg, prefix="lidar_")
            torch.cuda.synchronize(); dt = time.time() - t0; t_total += dt
            dep = res["depth"].reshape(-1).double().cpu().numpy()
            med = res["median_depth"].reshape(-1).double().cpu().numpy()
            opa = res["opacity"].reshape(-1).float().cpu().numpy()
            f32 = lambda x: np.asarray(x, dtype=np.float32)
            nan = np.full(n, np.nan, np.float32)
            np.savez_compressed(
                a.out / f"frame_{t_abs:04d}.npz",
                pred_xyz=f32(d_s * dep[:, None]), gt_xyz=f32(p_s), pred_median_xyz=f32(d_s * med[:, None]),
                raster_shape=np.array([n]),
                azimuth=f32(np.degrees(np.arctan2(d_s[:, 1], d_s[:, 0]))),
                elevation=f32(np.degrees(np.arcsin(np.clip(d_s[:, 2], -1, 1)))),
                gt_range=f32(r), gt_intensity=f32(inten),
                pred_range=f32(dep), pred_median_range=f32(med),
                pred_intensity=nan, pred_ray_drop_logit=nan,
                lidar_to_world=f32(l2w[:3, :4]),
                pred_opacity=opa, normed_timestamp=np.float32(nts.item()), timestep=np.int64(t_abs),
            )
            log.append({"t": t_abs, "n": n, "sec": dt, "max_dev_vs_training_rays_m": dev,
                        "median_abs_range_err_m": float(np.median(np.abs(dep - r)))})
            print(f"  t={t_abs:4d}  n={n}  {dt:.2f}s  |pred-gt| median {log[-1]['median_abs_range_err_m']:.3f} m  "
                  f"(rays == training rays to {dev:.1e} m)", flush=True)

    (a.out / "DUMP_MANIFEST.json").write_text(json.dumps({
        "method": "emernerf", "run_dir": str(a.run_dir), "checkpoint": str(ckpt), "step": step,
        "lidar_frame_provenance": ls.points_frame, "gt_xyz_frame": "LiDAR sensor frame",
        "lidar_pose_time": ls.lidar_pose_time, "camera_pose_time": ds.pixel_source.camera_pose_time,
        "heldout_timesteps_split": all_held, "dumped_timesteps": todo,
        "nan_fields": ["pred_intensity", "pred_ray_drop_logit"],
        "raster": "flat per-return arrays (raster_shape=[N]); no non-return rays",
        "seconds_render": t_total, "frames": log}, indent=2))
    print(f"dumped {len(todo)} sweeps -> {a.out}  ({t_total:.0f} s render)")
    print("LIDAR_DUMP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
