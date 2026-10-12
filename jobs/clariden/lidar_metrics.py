"""LiDAR depth metrics: rendered raw z-depth vs GT LiDAR returns projected into each val camera.
Validated 2026-10-12 (3D->2D GT boxes reproject with 0.000 px error under this convention):
  'correct' : p_ego = T_lidar_rig @ p_raw (sweeps are stored in the LIDAR SENSOR frame);
              p_cam = inv(T_camera_rig) @ p_ego  (T_camera_rig is the camera pose in the rig).
  'legacy'  : the math of the original compute_lidar_metrics.py (sweep used as ego; T_camera_rig used
              directly as cam<-rig). Reported only so old numbers can be re-interpreted; it is WRONG."""
import argparse, gzip, json
from pathlib import Path
import numpy as np, imageio.v2 as imageio
CAMS = ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_cross_right_120fov", "camera_rear_left_70fov",
        "camera_rear_right_70fov", "camera_front_tele_30fov", "camera_rear_tele_30fov"]

def load_depth(p):
    with gzip.open(p, "rb") as f: arr = np.load(f)
    return (arr[..., 0] if arr.ndim == 3 else arr).astype(np.float32)

def run(clip, root, mode, val_t):
    K_all = json.loads((clip / "calib/K_rectified.json").read_text())
    T_all = {k: np.array(v, float) for k, v in json.loads((clip / "calib/T_camera_rig.json").read_text()).items()}
    T_lidar_rig = np.array(json.loads((clip / "calib/T_lidar_rig.json").read_text())["lidar_top_360fov"], float)
    Tsel = (lambda T: np.linalg.inv(T)) if mode == "correct" else (lambda T: T)
    masks = {}
    for cam in CAMS:
        mp = clip / "masks" / cam / "valid_mask.png"
        if mp.exists(): m = imageio.imread(mp); masks[cam] = (m[..., 0] if m.ndim == 3 else m) > 128
    errs, per_cam, n_in_img = [], {c: [] for c in CAMS}, 0
    for t in val_t:
        sp = clip / "lidar" / f"sweep_{t:04d}.npy"
        if not sp.exists(): continue
        pts = np.load(sp)[:, :3].astype(np.float64)
        if mode == "correct": pts = pts @ T_lidar_rig[:3, :3].T + T_lidar_rig[:3, 3]  # sensor -> ego
        for cam in CAMS:
            dp = root / cam / f"frame_{t:04d}.npy.gz"
            if not dp.exists(): continue
            depth = load_depth(dp); H, W = depth.shape[:2]
            T = Tsel(T_all[cam]); R, tv = T[:3, :3], T[:3, 3]
            pc = pts @ R.T + tv; Z = pc[:, 2]; ok = Z > 0.5
            u = K_all[cam]["fx"] * pc[:, 0] / np.where(ok, Z, 1) + K_all[cam]["cx"]; v = K_all[cam]["fy"] * pc[:, 1] / np.where(ok, Z, 1) + K_all[cam]["cy"]
            ok &= (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
            if not ok.any(): continue
            ui, vi, gz = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int), Z[ok]
            if cam in masks and masks[cam].shape == depth.shape:
                keep = masks[cam][vi, ui]; ui, vi, gz = ui[keep], vi[keep], gz[keep]
            n_in_img += len(gz)
            pz = depth[vi, ui]; good = (pz > 0.1) & (pz < 200) & np.isfinite(pz)
            if good.any():
                e = np.abs(pz[good] - gz[good]); errs.append(e); per_cam[cam].append(e)
    if not errs: return {"error": "no points"}
    e = np.concatenate(errs)
    res = {"num_points": int(len(e)), "num_points_in_image": int(n_in_img), "depth_l1_mean": float(e.mean()), "depth_l1_median": float(np.median(e)),
           "depth_rmse": float(np.sqrt((e ** 2).mean())), "acc@1m": float((e < 1).mean()), "acc@2m": float((e < 2).mean()), "acc@5m": float((e < 5).mean()),
           "per_camera_median": {c: (float(np.median(np.concatenate(v))) if v else None) for c, v in per_cam.items()}}
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clip", type=Path, required=True); ap.add_argument("--rawdepth-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--val-timesteps", default="odd"); a = ap.parse_args()
    val_t = list(range(1, 199, 2)) if a.val_timesteps == "odd" else [int(x) for x in a.val_timesteps.split(",")]
    out = {"correct": run(a.clip, a.rawdepth_root, "correct", val_t),
           "legacy_original_script(WRONG)": run(a.clip, a.rawdepth_root, "legacy", val_t)}
    out["note"] = "Use 'correct'. 'legacy' reproduces the published script's math (sweep as ego, T_camera_rig direct) and is geometrically wrong; kept only to re-interpret old numbers."
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
