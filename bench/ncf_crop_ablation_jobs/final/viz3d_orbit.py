#!/usr/bin/env python3
"""360-degree orbit video of a trained neurad-studio scene (SplatAD / NeuRAD / UniSim / FoggyField V1.3) — 3D sanity check.

  python viz3d_orbit.py --config <run>/config.yml --label <name> --clip NFF-02 --out <dir> [--camera camera_front_wide_120fov]

Path: start EXACTLY at the real held-out camera (middle eval timestep of --camera; frame 0 is compared with the GT image),
then orbit 360 deg around a pivot --pivot m ahead of it (horizontal, world-up from the mean camera up vector), rising
smoothly to +--lift m at the far side and looking slightly down. Time is frozen at that timestep (actors frozen).
Every frame = model RGB | model depth (turbo, log 1..--dmax m). Rolling-shutter metadata is dropped (static virtual
camera), exactly like ns-render's interpolated paths. Outputs in --out:
  orbit360_<cam>.mp4, contact_sheet_<cam>.jpg (8 angles), frame0_vs_gt_<cam>.jpg, meta.json
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

BW = Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}] [viz3d]", *a, flush=True)


def turbo(depth, valid, dmax):
    import matplotlib
    d = np.clip(depth, 1.0, dmax)
    x = (np.log(d) - 0.0) / math.log(dmax)
    rgb = matplotlib.colormaps["turbo"](1.0 - x)[..., :3]      # near = red, far = blue
    rgb[~valid] = 0.0
    return rgb


def label(img_u8, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_u8)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
    dr.text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def look_at(C, target, up):
    z = C - target; z = z / np.linalg.norm(z)                  # nerfstudio camera looks along -z
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z, C], 1)                           # 3x4


def rodrigues(v, k, th):
    return v * math.cos(th) + np.cross(k, v) * math.sin(th) + k * np.dot(k, v) * (1 - math.cos(th))


def meta_for_path(md, n):
    """Every per-camera metadata entry of the real camera, repeated n times; velocities -> 0 (static virtual camera,
    so rolling-shutter / velocity compensation is a no-op but the keys the models assert on are present)."""
    if not md:
        return None
    out = {}
    for k, v in md.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1:
            v = v.expand(n, *v.shape[1:]).clone()
            if "veloc" in k:
                v = torch.zeros_like(v)
        out[k] = v
    log("path metadata:", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in out.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution factor")
    ap.add_argument("--pivot", type=float, default=8.0, help="orbit centre, metres ahead of the camera")
    ap.add_argument("--lift", type=float, default=4.0, help="extra height (m) at the far side of the orbit")
    ap.add_argument("--drop", type=float, default=1.5, help="look-at point lowered by this (m) at the far side")
    ap.add_argument("--dmax", type=float, default=80.0)
    ap.add_argument("--chunk", type=int, default=1 << 16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for p in (BW / "code/_src/unisim", BW / "jobs"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras
    from make_videos import Mp4Writer
    from PIL import Image

    T0 = time.time()
    config, pipeline, ckpt, step = eval_setup(a.config, eval_num_rays_per_chunk=a.chunk, test_mode="val")
    model = pipeline.model; model.eval()
    ds = pipeline.datamanager.eval_dataset
    cams = ds.cameras
    names = [Path(f) for f in ds._dataparser_outputs.image_filenames]
    idx = [i for i, n in enumerate(names) if n.parent.name == a.camera]
    assert idx, f"no eval images of {a.camera}"
    ts = [int(names[i].stem.split("_")[1]) for i in idx]
    k = idx[int(np.argsort(ts)[len(ts) // 2])]
    t_mid = int(names[k].stem.split("_")[1])
    cam0 = cams[k:k + 1]
    log(f"model {type(model).__name__} step {step} ckpt {ckpt}; {a.camera} held-out timestep {t_mid} (eval idx {k}); "
        f"load {time.time() - T0:.0f}s")

    allc = cams.camera_to_worlds.detach().cpu().numpy().astype(np.float64)
    up = allc[:, :3, 1].mean(0); up /= np.linalg.norm(up)
    c2w0 = cam0.camera_to_worlds.reshape(3, 4).detach().cpu().numpy().astype(np.float64)
    C0, fwd = c2w0[:, 3], -c2w0[:, 2]
    fh = fwd - np.dot(fwd, up) * up; fh /= np.linalg.norm(fh)
    P = C0 + a.pivot * fh
    c2ws = []
    for i in range(a.frames):
        th = 2 * math.pi * i / a.frames
        s = math.sin(th / 2) ** 2
        C = P + rodrigues(C0 - P, up, th) + a.lift * s * up
        c2ws.append(look_at(C, P - a.drop * s * up, up))
    c2ws[0] = c2w0                                              # frame 0 = the real held-out camera, unmodified
    n = a.frames
    ex = lambda t: None if t is None else t.reshape(1, -1)[:, :1].expand(n, 1).clone()  # noqa: E731
    path = Cameras(camera_to_worlds=torch.from_numpy(np.stack(c2ws)).float(),
                   fx=ex(cam0.fx), fy=ex(cam0.fy), cx=ex(cam0.cx), cy=ex(cam0.cy),
                   width=ex(cam0.width), height=ex(cam0.height), camera_type=ex(cam0.camera_type),
                   times=ex(cam0.times) if cam0.times is not None else None,
                   distortion_params=None if cam0.distortion_params is None else cam0.distortion_params.reshape(1, -1).expand(n, -1).clone(),
                   metadata=meta_for_path(cam0.metadata, n))
    path.rescale_output_resolution(a.scale)

    tag = a.camera.replace("camera_", "")
    mp4 = a.out / f"orbit360_{tag}.mp4"
    writer, sheet, f0, stats = None, [], None, {"depth_median_m": [], "rgb_std": []}
    T1 = time.time()
    with torch.no_grad():
        for i in range(n):
            out = model.get_outputs_for_camera(path[i:i + 1].to(model.device))
            rgb = out["rgb"].float().clamp(0, 1).cpu().numpy()
            dk = "depth" if "depth" in out else ("expected_depth" if "expected_depth" in out else None)
            if dk is not None:
                d = out[dk].float().cpu().numpy()[..., 0]
                valid = np.isfinite(d) & (d > 0)
                drgb = turbo(d, valid, a.dmax)
                stats["depth_median_m"].append(float(np.median(d[valid])) if valid.any() else None)
            else:
                drgb = np.zeros_like(rgb)
            stats["rgb_std"].append(float(rgb.std()))
            deg = 360.0 * i / n
            left = label((rgb * 255).astype(np.uint8), f"{a.label} | {a.clip} {tag} t={t_mid} | orbit {deg:5.1f} deg | RGB")
            right = label((drgb * 255).astype(np.uint8), f"expected depth (log 1-{a.dmax:.0f} m, red = near, blue = far/empty)")
            frame = np.concatenate([left, right], 1)
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]      # even dims for libx264
            if writer is None:
                writer = Mp4Writer(mp4, frame.shape[1], frame.shape[0], a.fps)
            writer.append(frame)
            if i == 0:
                f0 = rgb
            if i % max(1, n // 8) == 0 and len(sheet) < 8:
                sheet.append(frame)
            if i % 30 == 0:
                log(f"frame {i}/{n} ({(time.time() - T1) / (i + 1):.2f} s/frame)")
    writer.close()
    # contact sheet: 8 angles, 2 columns of (RGB|depth)
    h, w = sheet[0].shape[:2]
    cs = np.zeros(((len(sheet) + 1) // 2 * h, 2 * w, 3), np.uint8)
    for j, fr in enumerate(sheet):
        cs[(j // 2) * h:(j // 2 + 1) * h, (j % 2) * w:(j % 2 + 1) * w] = fr
    Image.fromarray(cs).resize((cs.shape[1] // 2, cs.shape[0] // 2)).save(a.out / f"contact_sheet_{tag}.jpg", quality=88)
    # frame 0 vs the held-out GT image (same camera, same time) -> registration check
    gt = ds[k]["image"]                                         # dataset __getitem__ = same crop as training/eval
    gt = gt.float().numpy() / (255.0 if gt.dtype == torch.uint8 else 1.0)
    gt = np.clip(gt[..., :3], 0, 1)
    gt_s = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((f0.shape[1], f0.shape[0])))
    psnr0 = float(-10 * np.log10(np.mean((gt_s / 255.0 - f0) ** 2) + 1e-12))
    pair = np.concatenate([label(gt_s, f"GT held-out {a.clip} {tag} t={t_mid}"),
                           label((f0 * 255).astype(np.uint8), f"{a.label} render (orbit frame 0) PSNR {psnr0:.2f} dB @ {a.scale}x")], 1)
    Image.fromarray(pair).save(a.out / f"frame0_vs_gt_{tag}.jpg", quality=90)
    dm = [x for x in stats["depth_median_m"] if x is not None]
    meta = {"label": a.label, "clip": a.clip, "camera": a.camera, "heldout_timestep": t_mid, "config": str(a.config),
            "ckpt": str(ckpt), "step": int(step), "model": type(model).__name__, "frames": n, "fps": a.fps,
            "scale": a.scale, "pivot_m": a.pivot, "lift_m": a.lift, "world_up": up.tolist(),
            "frame0_psnr_vs_gt_db": psnr0, "depth_median_m_min_max": [min(dm), max(dm)] if dm else None,
            "rgb_std_min": min(stats["rgb_std"]), "render_s_per_frame": (time.time() - T1) / n,
            "note": "frame 0 = real held-out camera; other frames are novel views (no GT)."}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"DONE {mp4} ({time.time() - T0:.0f}s); frame0 PSNR {psnr0:.2f} dB")


if __name__ == "__main__":
    main()
