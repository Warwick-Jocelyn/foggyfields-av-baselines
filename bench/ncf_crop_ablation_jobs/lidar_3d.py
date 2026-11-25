#!/usr/bin/env python3
"""3-D LiDAR visualisation: an orbiting camera around the reconstructed point cloud, and a
voxel-cube occupancy view of reconstruction-vs-GT.

Software renderer in numpy -- no open3d/GL on this cluster. Points are z-buffered splats;
voxels are real cubes with three shaded faces, which is what makes an occupancy grid read as
3-D rather than as a scatter plot.

  lidar_3d.py --dump <dir> -o <outdir> [--mode points|voxels|both]
              [--frame N] [--n-orbit 120] [--size 1100] [--range 60] [--cell 0.5]

Outputs: orbit_points.mp4, orbit_voxels.mp4, plus still frames at 4 angles.
"""
import argparse, math
from pathlib import Path
import numpy as np, av
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float)
    s = np.cross(f, u); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, -u, f])            # world -> cam (x right, y down, z forward)
    t = -R @ np.asarray(eye, float)
    return R, t


def project(P, R, t, fx, cx, cy, W, H, near=0.3):
    Pc = P @ R.T + t
    z = Pc[:, 2]
    ok = z > near
    x = fx * Pc[ok, 0] / z[ok] + cx
    y = fx * Pc[ok, 1] / z[ok] + cy
    return x, y, z[ok], ok


def zsplat(img, zbuf, x, y, z, rgb, rad):
    """Painter with a z-buffer; rad in pixels (int)."""
    H, W = zbuf.shape
    order = np.argsort(-z)                        # far first
    x, y, z, rgb = x[order], y[order], z[order], rgb[order]
    xi = np.round(x).astype(np.int32); yi = np.round(y).astype(np.int32)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if dx * dx + dy * dy > rad * rad + 1:
                continue
            xx, yy = xi + dx, yi + dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            if not m.any():
                continue
            flat = yy[m] * W + xx[m]
            zb = zbuf.reshape(-1)
            cand = z[m]
            # keep the nearest: process far->near so later writes win
            better = cand <= zb[flat]
            idx = flat[better]
            zb[idx] = cand[better]
            img.reshape(-1, 3)[idx] = rgb[m][better]


def height_colour(zv, lo=-2.5, hi=8.0):
    t = np.clip((zv - lo) / (hi - lo), 0, 1)
    # turbo-ish
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def err_colour(e, vmax=0.8):
    t = np.clip(e / vmax, 0, 1)
    r = np.clip(t * 2.2, 0, 1); g = np.clip(t * 2.2 - 0.9, 0, 1) * 0.9; b = np.clip(0.9 - t * 2.4, 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def cube_faces(centers, cell):
    """Top / +x / +y faces of each voxel -> (N*3, 4, 3) quads and a per-quad shade."""
    h = cell / 2
    top = np.array([[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]])
    px  = np.array([[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]])
    py  = np.array([[-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h]])
    quads = np.concatenate([centers[:, None, :] + top,
                            centers[:, None, :] + px,
                            centers[:, None, :] + py], 0)
    shade = np.concatenate([np.full(len(centers), 1.0),
                            np.full(len(centers), 0.62),
                            np.full(len(centers), 0.80)])
    return quads, shade


def render_quads(img, zbuf, quads, cols, shade, R, t, fx, cx, cy):
    """Flat-fill convex quads with a per-quad depth (centroid)."""
    H, W = zbuf.shape
    Pc = quads.reshape(-1, 3) @ R.T + t
    z = Pc[:, 2].reshape(-1, 4).mean(1)
    vis = z > 0.3
    if not vis.any():
        return
    xs = (fx * Pc[:, 0] / np.maximum(Pc[:, 2], 1e-3) + cx).reshape(-1, 4)
    ys = (fx * Pc[:, 1] / np.maximum(Pc[:, 2], 1e-3) + cy).reshape(-1, 4)
    order = np.argsort(-z)
    pim = Image.fromarray(img)
    dr = ImageDraw.Draw(pim)
    zb = zbuf.reshape(-1)
    for i in order:
        if not vis[i]:
            continue
        px, py = xs[i], ys[i]
        if px.max() < 0 or px.min() > W or py.max() < 0 or py.min() > H:
            continue
        c = (cols[i] * shade[i]).astype(np.uint8)
        dr.polygon(list(zip(px, py)), fill=tuple(int(v) for v in c))
    img[:] = np.asarray(pim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--dump-b", type=Path, help="second run, for a side-by-side orbit")
    ap.add_argument("--label", default=""); ap.add_argument("--label-b", default="")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--mode", default="both", choices=["points", "voxels", "both"])
    ap.add_argument("--n-orbit", type=int, default=120)
    ap.add_argument("--size", type=int, default=1100)
    ap.add_argument("--range", type=float, default=60.0)
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--dist", type=float, default=95.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--rad", type=int, default=2, help="point splat radius in px")
    ap.add_argument("--view", default="orbit", choices=["orbit", "ego-spin", "ego-drive"],
                    help="orbit = camera circles the scene; ego-spin = camera sits in the car and "
                         "turns on the spot; ego-drive = camera sits in the car looking forward "
                         "while the sequence plays")
    ap.add_argument("--ego-h", type=float, default=1.6, help="eye height above the rig origin (m)")
    ap.add_argument("--drive-frames", type=int, default=0, help="ego-drive: how many frames (0 = all)")
    ap.add_argument("--fov", type=float, default=75.0, help="ego views: horizontal FOV in degrees")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(a.dump.glob("frame_*.npz"))
    pick = files[len(files) // 2] if a.frame < 0 else a.dump / f"frame_{a.frame:04d}.npz"
    d = np.load(pick)
    gt, pr = d["gt_xyz"].astype(np.float32), d["pred_xyz"].astype(np.float32)
    R_ = a.range
    keep = lambda P: P[(np.abs(P[:, 0]) < R_) & (np.abs(P[:, 1]) < R_) & (P[:, 2] > -3) & (P[:, 2] < 12)]
    gtk, prk = keep(gt), keep(pr)
    err, _ = cKDTree(pr).query(gt, k=1, workers=-1)
    mk = (np.abs(gt[:, 0]) < R_) & (np.abs(gt[:, 1]) < R_) & (gt[:, 2] > -3) & (gt[:, 2] < 12)
    gte, erre = gt[mk], err[mk]
    lbl = a.label or a.dump.name
    print(f"{pick.name}: GT {len(gtk):,}  pred {len(prk):,}  median NN {np.median(err):.3f} m")

    W = H = a.size
    fx_orbit = W * 0.9
    fx_ego = None  # set from --fov just below
    fx_ego = (W / 2) / math.tan(math.radians(a.fov / 2))
    f_big, f_sm = font(max(17, W // 46)), font(max(13, W // 64))

    def cam_for(ang):
        """ang in [0, 2pi). Returns (R, t, fx)."""
        if a.view == "orbit":
            eye = np.array([a.dist * math.cos(ang), a.dist * math.sin(ang),
                            a.dist * math.sin(math.radians(a.elev))])
            return look_at(eye, (0, 0, 0)) + (fx_orbit,)
        # ego views: the eye sits on the rig, a little above the LiDAR origin
        eye = np.array([0.0, 0.0, a.ego_h])
        if a.view == "ego-spin":
            yaw = ang
        else:                       # ego-drive: look straight ahead (+x is forward)
            yaw = 0.0
        tgt = eye + np.array([math.cos(yaw), math.sin(yaw), -0.12])
        return look_at(eye, tgt) + (fx_ego,)

    def frame_points(ang, dat=None):
        g_, p_, ge_, er_ = (gtk, prk, gte, erre) if dat is None else dat
        Rm, tv, fxx = cam_for(ang)
        panels = []
        for P, C, title in ((g_, height_colour(g_[:, 2]), "GT point cloud"),
                            (p_, height_colour(p_[:, 2]), "RECONSTRUCTED"),
                            (ge_, err_colour(er_), f"NN error  (blue 0 -> red {0.8:.1f} m)")):
            img = np.zeros((H, W, 3), np.uint8); img[:] = (12, 12, 14)
            zb = np.full((H, W), 1e9, np.float32)
            x, y, z, ok = project(P, Rm, tv, fxx, W / 2, H / 2, W, H)
            zsplat(img, zb, x, y, z, C[ok], a.rad)
            im = Image.fromarray(img); dr = ImageDraw.Draw(im)
            dr.text((12, 10), title, fill=(255, 255, 255), font=f_sm)
            panels.append(im)
        out = Image.new("RGB", (W * 3 + 16, H + 34), (12, 12, 14))
        ImageDraw.Draw(out).text((12, 7), f"{lbl}   {pick.stem}   {a.view}", fill=(255, 255, 255), font=f_big)
        for i, p in enumerate(panels):
            out.paste(p, (i * (W + 8), 32))
        w, h = out.size
        return out.crop((0, 0, w - w % 2, h - h % 2))

    # voxels
    C = a.cell
    n = int(2 * R_ / C)
    def vox(P):
        m = (np.abs(P[:, 0]) < R_) & (np.abs(P[:, 1]) < R_) & (P[:, 2] > -2.5) & (P[:, 2] < 6.0)
        q = np.floor((P[m] + np.array([R_, R_, 2.5])) / C).astype(np.int32)
        return set(map(tuple, q))
    vg, vp = vox(gt), vox(pr)
    both, false_, miss = vg & vp, vp - vg, vg - vp
    def centers(S):
        if not S: return np.zeros((0, 3), np.float32)
        q = np.array(sorted(S), np.float32)
        return (q + 0.5) * C - np.array([R_, R_, 2.5], np.float32)
    cb, cf, cm = centers(both), centers(false_), centers(miss)
    iou = len(both) / max(len(vg | vp), 1)
    print(f"voxels {C} m: both {len(both):,}  false {len(false_):,}  missed {len(miss):,}  IoU {iou:.3f}")

    def frame_voxels(ang):
        Rm, tv, fxx = cam_for(ang)
        img = np.zeros((H, W, 3), np.uint8); img[:] = (12, 12, 14)
        zb = np.full((H, W), 1e9, np.float32)
        allc = np.concatenate([cb, cf, cm]) if len(cb) + len(cf) + len(cm) else np.zeros((0, 3), np.float32)
        base = np.concatenate([np.tile(np.array([60, 210, 90]), (len(cb), 1)),
                               np.tile(np.array([235, 70, 60]), (len(cf), 1)),
                               np.tile(np.array([70, 130, 255]), (len(cm), 1))])
        q, sh = cube_faces(allc, C)
        cols = np.concatenate([base, base, base])
        render_quads(img, zb, q, cols, sh, Rm, tv, fxx, W / 2, H / 2)
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.text((12, 10), f"{lbl}  voxel occupancy {C} m  IoU {iou:.3f}  [{a.view}]", fill=(255, 255, 255), font=f_sm)
        for i, (c, t) in enumerate([((60, 210, 90), f"both  {len(both):,}"),
                                    ((235, 70, 60), f"false positive  {len(false_):,}"),
                                    ((70, 130, 255), f"missed  {len(miss):,}")]):
            dr.rectangle([12, 36 + i * 22, 28, 50 + i * 22], fill=c)
            dr.text((34, 34 + i * 22), t, fill=(235, 235, 235), font=f_sm)
        w, h = im.size
        return im.crop((0, 0, w - w % 2, h - h % 2))

    def write(mk_frame, path):
        f0 = mk_frame(0.0); Wv, Hv = f0.size
        with av.open(str(path), "w") as c:
            st = c.add_stream("libx264", rate=a.fps)
            st.width, st.height, st.pix_fmt = Wv, Hv, "yuv420p"
            st.options = {"crf": "20", "preset": "fast"}
            for i in range(a.n_orbit):
                im = f0 if i == 0 else mk_frame(2 * math.pi * i / a.n_orbit)
                if im.size != (Wv, Hv): im = im.resize((Wv, Hv))
                for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
                    c.mux(pkt)
            for pkt in st.encode(): c.mux(pkt)
        print(f"  {path.name}  {path.stat().st_size/1e6:.1f} MB")
        for k, ang in enumerate((0, math.pi/2, math.pi, 3*math.pi/2)):
            mk_frame(ang).save(path.with_name(path.stem + f"_view{k}.jpg"), quality=88)

    if a.view == "ego-drive":
        seq = files if not a.drive_frames else files[: a.drive_frames]
        def drive(i):
            e = np.load(seq[i])
            g = keep(e["gt_xyz"].astype(np.float32)); p = keep(e["pred_xyz"].astype(np.float32))
            er, _ = cKDTree(e["pred_xyz"]).query(e["gt_xyz"], k=1, workers=-1)
            gg = e["gt_xyz"].astype(np.float32)
            m2 = (np.abs(gg[:, 0]) < R_) & (np.abs(gg[:, 1]) < R_) & (gg[:, 2] > -3) & (gg[:, 2] < 12)
            return frame_points(0.0, (g, p, gg[m2], er[m2]))
        f0 = drive(0); Wv, Hv = f0.size
        out = a.outdir / "ego_drive_points.mp4"
        with av.open(str(out), "w") as c:
            st = c.add_stream("libx264", rate=a.fps)
            st.width, st.height, st.pix_fmt = Wv, Hv, "yuv420p"
            st.options = {"crf": "20", "preset": "fast"}
            for i in range(len(seq)):
                im = f0 if i == 0 else drive(i)
                if im.size != (Wv, Hv): im = im.resize((Wv, Hv))
                for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
                    c.mux(pkt)
            for pkt in st.encode(): c.mux(pkt)
        print(f"  {out.name}  {len(seq)} frames  {out.stat().st_size/1e6:.1f} MB")
        for k in (0, len(seq)//3, 2*len(seq)//3):
            drive(k).save(out.with_name(out.stem + f"_f{k:03d}.jpg"), quality=88)
        print("wrote", a.outdir); return

    pre = "orbit" if a.view == "orbit" else "egospin"
    if a.mode in ("points", "both"):
        write(frame_points, a.outdir / f"{pre}_points.mp4")
    if a.mode in ("voxels", "both"):
        write(frame_voxels, a.outdir / f"{pre}_voxels.mp4")
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()
