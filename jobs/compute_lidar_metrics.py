"""LiDAR depth metrics: compare model's rendered raw-depth to GT LiDAR sweep.

For each held-out (val) timestep, project each GT LiDAR return into every val
camera, sample the rendered depth at the projected pixel, and compute
|rendered_depth - lidar_depth|. Aggregate to:
  - depth_l1_mean   (m)
  - depth_l1_median (m)
  - depth_rmse      (m)
  - acc@1m, acc@2m, acc@5m (fraction)
  - num_points      (lidar returns used)

Conventions used (match NVIDIA_AV_Fog dataparser):
  - T_camera_rig.json content IS cam->rig (sensor->rig, NOT its inverse).
  - Camera frame is OpenCV (cam+x = right, cam+y = down, cam+z = forward).
  - LiDAR sweep .npy [N,6] = (x, y, z, intensity, t_offset_us, ring_id) in EGO frame.
  - World frame already ISO 8855 (no extra rotation needed).
  - Rendered depth = z-depth (camera-frame Z), same convention as gsplat/nerfacc.

Pixels in the rectified black-corner region (per valid_mask.png) are skipped.

    python compute_lidar_metrics.py --method splatad --seq 002 --output metrics.json
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import pandas as pd


DATA_ROOT = Path("/networkhome/WMGDS/wang3_y/ETH/Dataset/NVIDIA_AV_Fog/processed")
OUT_ROOT = Path("/networkhome/WMGDS/wang3_y/ETH/NVIDIA-Fog-Output")
CAMS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_front_tele_30fov",
    "camera_rear_tele_30fov",
]


def _resolve_clip_root(seq: str) -> Path:
    for child in sorted(DATA_ROOT.iterdir()):
        if child.name.startswith(f"{seq}_"):
            return child / "processed_pinhole"
    raise FileNotFoundError(f"no clip {seq} under {DATA_ROOT}")


def _load_depth(p: Path) -> np.ndarray:
    """Load a gzipped float32 raw-depth npy from ns-render."""
    with gzip.open(p, "rb") as f:
        arr = np.load(f)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["splatad", "neurad"])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    clip = _resolve_clip_root(args.seq)
    K_all = json.loads((clip / "calib" / "K_rectified.json").read_text())
    T_cam_rig_all = {k: np.array(v, dtype=np.float64)
                     for k, v in json.loads((clip / "calib" / "T_camera_rig.json").read_text()).items()}
    # Masks (per-cam, time-invariant)
    masks = {}
    for cam in CAMS:
        mp = clip / "masks" / cam / "valid_mask.png"
        if mp.exists():
            m = imageio.imread(mp)
            if m.ndim == 3:
                m = m[..., 0]
            masks[cam] = m > 128
        else:
            masks[cam] = None

    render_root = OUT_ROOT / args.method / args.seq / "renders_rawdepth" / "val" / "raw-depth"

    # nvs_50_50: val timesteps are 1, 3, 5, ..., 197 (odd indices)
    val_t = list(range(1, 199, 2))

    abs_errs, n_total, per_cam = [], 0, {}

    for t in val_t:
        sweep_p = clip / "lidar" / f"sweep_{t:04d}.npy"
        if not sweep_p.exists():
            continue
        sweep = np.load(sweep_p)
        pts_ego = sweep[:, 0:3].astype(np.float64)  # in ego frame

        for cam in CAMS:
            depth_p = render_root / cam / f"frame_{t:04d}.npy.gz"
            if not depth_p.exists():
                continue
            depth = _load_depth(depth_p)
            H, W = depth.shape[:2]
            mask = masks.get(cam)

            K = K_all[cam]
            fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
            # Use T_camera_rig.json content DIRECTLY as cam_from_rig (rig_to_cam).
            # Empirically verified: this convention gives sensible OpenCV projection
            # (cam_+z = forward, in-image lidar points overlap with the scene),
            # while inverting gives ~3x error increase + 3x fewer in-image points.
            T_cam_rig = T_cam_rig_all[cam]
            R, tvec = T_cam_rig[:3, :3], T_cam_rig[:3, 3]

            # Transform points: p_cam = R @ p_ego + t   (OpenCV: +z forward)
            p_cam = pts_ego @ R.T + tvec[None, :]
            Xc, Yc, Zc = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
            valid = Zc > 0.5  # in front of camera
            if not np.any(valid):
                continue
            u = (fx * Xc / Zc + cx)
            v = (fy * Yc / Zc + cy)
            valid &= (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
            if not np.any(valid):
                continue
            u_i = np.round(u[valid]).astype(np.int32)
            v_i = np.round(v[valid]).astype(np.int32)
            gt_z = Zc[valid]  # z-depth from camera
            # Apply mask if present
            if mask is not None:
                # Resize-ignore: depth maps and masks are at same native resolution
                if mask.shape == depth.shape:
                    in_mask = mask[v_i, u_i]
                    u_i, v_i, gt_z = u_i[in_mask], v_i[in_mask], gt_z[in_mask]
                    if len(gt_z) == 0:
                        continue

            pred_z = depth[v_i, u_i]
            # Skip pixels where the renderer produced 0/inf (no Gaussian/no density)
            ok = (pred_z > 0.1) & (pred_z < 200) & np.isfinite(pred_z)
            if not np.any(ok):
                continue
            err = np.abs(pred_z[ok] - gt_z[ok])
            abs_errs.append(err)
            n_total += err.size
            per_cam.setdefault(cam, []).append(err)

    if not abs_errs:
        out = {"error": "no valid (lidar, depth) pairs found"}
    else:
        all_err = np.concatenate(abs_errs)
        out = {
            "method": args.method,
            "seq": args.seq,
            "num_points": int(n_total),
            "depth_l1_mean_m": float(all_err.mean()),
            "depth_l1_median_m": float(np.median(all_err)),
            "depth_rmse_m": float(np.sqrt((all_err ** 2).mean())),
            "acc_at_1m": float((all_err < 1.0).mean()),
            "acc_at_2m": float((all_err < 2.0).mean()),
            "acc_at_5m": float((all_err < 5.0).mean()),
            "per_camera_median_m": {
                cam: float(np.median(np.concatenate(v)))
                for cam, v in per_cam.items()
            },
        }
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
