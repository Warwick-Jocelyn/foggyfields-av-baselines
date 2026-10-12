"""LiDAR GT-vs-SplatAD visualisation from lidar_dump/*.npz:
  1. bev_gt_vs_pred.mp4      : bird's-eye view, [GT | SplatAD | overlay (GT grey, SplatAD red)], ±60 m, colour = height
  2. range_image_gt_vs_pred.mp4: LiDAR range image (azimuth x beam) [GT ; SplatAD ; |error|], turbo colormap 0-80 m
  3. (optional, --clip + --cam) cam_overlay.mp4: GT LiDAR and SplatAD LiDAR projected onto the GT front image
Usage: make_lidar_videos.py <lidar_dump dir> <out dir> [--clip <clip dir> --cam camera_front_wide_120fov --lidar-frame sensor|legacy]"""
import argparse, json, numpy as np, imageio.v2 as iio
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); from matplotlib import cm
ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("--clip"); ap.add_argument("--cam", default="camera_front_wide_120fov")
ap.add_argument("--lidar-frame", default="sensor", choices=["sensor", "legacy"]); ap.add_argument("--label", default=""); ap.add_argument("--fps", type=int, default=10); ap.add_argument("--stills", default="1,99,197")
a = ap.parse_args(); D, O = Path(a.dump), Path(a.out); O.mkdir(parents=True, exist_ok=True)
files = sorted(D.glob("frame_*.npz")); stills = {int(s) for s in a.stills.split(",")}
R, S = 60.0, 600  # BEV range (m) and panel size (px)
def bev(xyz, col):
    img = np.zeros((S, S, 3), np.uint8)
    m = (abs(xyz[:, 0]) < R) & (abs(xyz[:, 1]) < R); x, y, z = xyz[m, 0], xyz[m, 1], xyz[m, 2]
    px = ((R - y) / (2 * R) * (S - 1)).astype(int); py = ((R - x) / (2 * R) * (S - 1)).astype(int)  # x forward = up, y left = left
    if col is None: c = (cm.viridis(np.clip((z + 2.5) / 5.0, 0, 1))[:, :3] * 255).astype(np.uint8)
    else: c = np.tile(np.array(col, np.uint8), (len(px), 1))
    img[py, px] = c; return img
def lab(arr, txt):
    im = Image.fromarray(arr); d = ImageDraw.Draw(im); d.rectangle([0, 0, 8 + 7 * len(txt), 18], fill=(0, 0, 0)); d.text((4, 3), txt, fill=(255, 255, 0)); return np.asarray(im)
def rng_img(r, vmax=80.0):
    v = np.clip(r, 0, vmax) / vmax; c = (cm.turbo(v)[..., :3] * 255).astype(np.uint8); c[r <= 0] = 0; return c
wb = iio.get_writer(O / "bev_gt_vs_pred.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=2)
wr = iio.get_writer(O / "range_image_gt_vs_pred.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=2)
wc = None
if a.clip:
    C = Path(a.clip); K = json.load(open(C / "calib/K_rectified.json"))[a.cam]; Tcr = np.array(json.load(open(C / "calib/T_camera_rig.json"))[a.cam]); Tlr = np.array(json.load(open(C / "calib/T_lidar_rig.json"))["lidar_top_360fov"])
    wc = iio.get_writer(O / f"cam_overlay_{a.cam}.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=2)
def to_cam(xyz):
    ego = xyz @ Tlr[:3, :3].T + Tlr[:3, 3] if a.lidar_frame == "sensor" else xyz  # legacy loader: model frame == raw-as-ego
    Tc = np.linalg.inv(Tcr); pc = ego @ Tc[:3, :3].T + Tc[:3, 3]; Z = pc[:, 2]; ok = Z > 0.5
    u = K["fx"] * pc[:, 0] / np.where(ok, Z, 1) + K["cx"]; v = K["fy"] * pc[:, 1] / np.where(ok, Z, 1) + K["cy"]
    ok &= (u >= 0) & (u < K["w"]) & (v >= 0) & (v < K["h"]); return u[ok], v[ok], Z[ok]
for f in files:
    t = int(f.stem.split("_")[1]); d = np.load(f); G, P = d["gt_xyz"], d["pred_xyz"]
    # 1. BEV
    ov = np.zeros((S, S, 3), np.uint8); ov = np.maximum(ov, bev(G, (140, 140, 140))); ov = np.maximum(ov, bev(P, (255, 60, 60)))
    row = np.concatenate([lab(bev(G, None), f"GT LiDAR  t={t}  {len(G)} pts"), lab(bev(P, None), f"SplatAD LiDAR render {a.label}  {len(P)} pts"), lab(ov, "overlay: GT grey, SplatAD red  (+-60 m, ego fwd=up)")], 1)
    wb.append_data(row)
    if t in stills: Image.fromarray(row).save(O / f"bev_t{t:04d}.png")
    # 2. range images: bin rays into (128 elevation rows x 1800 azimuth cols) from the per-ray azimuth/elevation (degrees)
    az, el, gr, pr = d["azimuth"], d["elevation"], d["gt_range"], d["pred_range"]; valid = gr > 0
    el_rows = np.linspace(el[valid].max(), el[valid].min(), 128); ri = np.abs(el[:, None] - el_rows[None, :]).argmin(1); ci = ((az % 360) / 360 * 1799).astype(int)
    def raster(vals):
        img = np.zeros((128, 1800), np.float32); img[ri[valid], ci[valid]] = vals[valid]; return img
    G2, P2 = raster(gr), raster(pr); E2 = np.abs(P2 - G2); E2[G2 <= 0] = 0
    up = lambda x: np.repeat(x, 3, 0)
    stack = np.concatenate([lab(up(rng_img(G2)), f"GT range image t={t} (azimuth x 128 beams, turbo 0-80 m)"), lab(up(rng_img(P2)), f"SplatAD rendered range {a.label}"), lab(up(rng_img(E2, 10.0)), "|error| (turbo 0-10 m)")], 0)
    wr.append_data(stack)
    if t in stills: Image.fromarray(stack).save(O / f"range_t{t:04d}.png")
    # 3. camera overlay
    if wc is not None:
        img = Image.open(C / f"images/{a.cam}/frame_{t:04d}.jpg").convert("RGB"); panels = []
        for pts, name in ((G, "GT LiDAR"), (P, "SplatAD LiDAR")):
            u, v, z = to_cam(pts); im = img.copy(); dr = ImageDraw.Draw(im); col = (cm.turbo(np.clip(z, 0, 60) / 60)[:, :3] * 255).astype(int)
            for i in range(0, len(u), 3): dr.ellipse([u[i] - 1.5, v[i] - 1.5, u[i] + 1.5, v[i] + 1.5], fill=tuple(col[i]))
            panels.append(lab(np.asarray(im), f"{name} projected on GT image t={t} ({len(u)} pts, turbo 0-60 m)"))
        fr = np.concatenate(panels, 1); wc.append_data(fr)
        if t in stills: Image.fromarray(fr).save(O / f"cam_overlay_t{t:04d}.png")
wb.close(); wr.close(); wc and wc.close(); print("wrote", O, len(files), "frames")
