"""Independent cross-check of the NFF LiDAR sweep frame. Three hypotheses for the xyz stored in sweep_*.npy:
  H1 raw = LIDAR SENSOR frame, ego = T_lidar_rig @ raw          (T = lidar pose in rig; matches its translation (1.14,0.01,1.94))
  H2 raw = EGO frame already                                       (CONVENTIONS.md; net effect of the old dataparser)
  H3 raw = sensor frame, ego = inv(T_lidar_rig) @ raw              (the file's 'T_sensor_from_ego' string taken literally; nff-toolkit)
Tests per clip: ground height, beam-elevation residual vs lidar_intrinsics, LiDAR hits inside 3-D GT boxes,
LiDAR hits inside 2-D GT boxes at the box depth, and a projection overlay on the front camera image."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
clips = {"NFF-02": ("/capstor/scratch/cscs/yitiwang/splatad-nff-repro/data/NFF/clips/NFF-02__0031aee2", 50),
         "NFF-03": ("/capstor/scratch/cscs/yitiwang/splatad-nff-repro/data/NFF/clips/NFF-03__5c5006b6", 50),
         "NFF-10": (sys.argv[1] + "/NFF-10__e8cd2d62", 50), "NFF-13": (sys.argv[1] + "/NFF-13__a7b816cd", 50)}
cam = "camera_front_wide_120fov"
def quat_R(qx, qy, qz, qw):
    n = qx*qx+qy*qy+qz*qz+qw*qw; s = 2/n
    return np.array([[1-s*(qy*qy+qz*qz), s*(qx*qy-qz*qw), s*(qx*qz+qy*qw)],[s*(qx*qy+qz*qw), 1-s*(qx*qx+qz*qz), s*(qy*qz-qx*qw)],[s*(qx*qz-qy*qw), s*(qy*qz+qx*qw), 1-s*(qx*qx+qy*qy)]])
print(f"{'clip':7s} {'hyp':3s} {'ground z':>9s} {'ring-elev resid':>16s} {'in 3D boxes':>12s} {'in 2D box@depth':>16s} {'2D-box pts total':>16s}")
for name, (root, t) in clips.items():
    root = Path(root)
    T = np.array(json.load(open(root/"calib/T_lidar_rig.json"))["lidar_top_360fov"])
    K = json.load(open(root/"calib/K_rectified.json"))[cam]; Tcr = np.array(json.load(open(root/"calib/T_camera_rig.json"))[cam])
    rows_el = np.array(json.load(open(root/"calib/lidar_intrinsics.json"))["row_elevations_rad"])
    sw = np.load(root/f"lidar/sweep_{t:04d}.npy"); raw = sw[:, :3].astype(float); ring = sw[:, 5].astype(int)
    b3 = pd.read_parquet(root/"labels/obstacles_3d_ego.parquet"); b3 = b3[b3.timestep_idx == t]
    b2 = pd.read_parquet(root/f"labels/obstacles_2d_{cam}.parquet"); b2 = b2[b2.timestep_idx == t]
    img = Image.open(root/f"images/{cam}/frame_{t:04d}.jpg").convert("RGB"); W_, H_ = img.size
    hyps = {"H1": raw @ T[:3, :3].T + T[:3, 3], "H2": raw.copy(), "H3": raw @ np.linalg.inv(T)[:3, :3].T + np.linalg.inv(T)[:3, 3]}
    panels = []
    for h, ego in hyps.items():
        r = np.hypot(ego[:, 0], ego[:, 1]); near = (r > 5) & (r < 25)
        hist, edges = np.histogram(ego[near, 2], bins=np.arange(-5, 3, 0.1)); ground = edges[hist.argmax()] + 0.05
        # beam elevation about the hypothesised SENSOR origin: H1/H3 sensor origin = raw origin; H2 sensor origin = T[:3,3] in ego
        pts_s = raw if h != "H2" else (raw - T[:3, 3])
        el = np.arctan2(pts_s[:, 2], np.hypot(pts_s[:, 0], pts_s[:, 1])); resid = np.degrees(np.abs(el - rows_el[np.clip(ring, 0, 127)]))
        resid_med = np.median(resid)
        # 3-D boxes (ego frame)
        n3 = 0
        for _, b in b3.iterrows():
            R = quat_R(b.q_x, b.q_y, b.q_z, b.q_w); d = (ego - np.array([b.center_x, b.center_y, b.center_z])) @ R
            n3 += int(((abs(d[:, 0]) < b.size_x/2) & (abs(d[:, 1]) < b.size_y/2) & (abs(d[:, 2]) < b.size_z/2)).sum())
        # project to camera
        Tc = np.linalg.inv(Tcr); pc = ego @ Tc[:3, :3].T + Tc[:3, 3]; Z = pc[:, 2]; ok = Z > 0.5
        u = K["fx"]*pc[:, 0]/np.where(ok, Z, 1)+K["cx"]; v = K["fy"]*pc[:, 1]/np.where(ok, Z, 1)+K["cy"]; ok &= (u >= 0) & (u < W_) & (v >= 0) & (v < H_)
        n2, n2tot = 0, 0
        for _, b in b2.iterrows():
            x0, y0, x1, y1 = b.bbox_modal_x0, b.bbox_modal_y0, b.bbox_modal_x1, b.bbox_modal_y1
            inb = ok & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1); n2tot += int(inb.sum())
            n2 += int((inb & (abs(Z - b.mean_depth_m) < 2.5)).sum())
        # overlay
        ov = img.copy(); dr = ImageDraw.Draw(ov); sel = np.where(ok)[0][::6]
        for i in sel:
            c = int(255 * min(Z[i], 60) / 60); dr.point((u[i], v[i]), fill=(255 - c, c, 0))
        for _, b in b2.iterrows():
            dr.rectangle([b.bbox_modal_x0, b.bbox_modal_y0, b.bbox_modal_x1, b.bbox_modal_y1], outline=(0, 255, 255), width=3)
        dr.text((10, 10), f"{name} t={t} {h}: ground z={ground:+.2f} m, pts in 3D boxes={n3}", fill=(255, 255, 0))
        panels.append(ov.resize((640, 480)))
        print(f"{name:7s} {h:3s} {ground:+9.2f} {resid_med:16.3f} {n3:12d} {n2:16d} {n2tot:16d}")
    out = Image.new("RGB", (640*3, 480)); [out.paste(p, (i*640, 0)) for i, p in enumerate(panels)]
    out.save(f"{sys.argv[2]}/overlay_{name}.jpg", quality=85)
print("2D box columns:", list(b2.columns)[:12])
