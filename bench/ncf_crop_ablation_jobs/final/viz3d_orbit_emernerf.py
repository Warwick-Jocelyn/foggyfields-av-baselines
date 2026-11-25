#!/usr/bin/env python3
"""EmerNeRF twin of viz3d_orbit.py: the same 360-degree orbit (same parameters, same outputs) for an EmerNeRF run.

  $EMER_PY viz3d_orbit_emernerf.py --run-dir <run>/train/run --label EmerNeRF --clip NFF-02 --out <dir>

EmerNeRF builds rays from pixel_source.cam_to_worlds[img_idx] (OpenCV axes: x right, y down, z forward); the orbit
poses are written into that slot of the middle held-out image of --camera, one frame at a time (restored afterwards),
so time / camera id / appearance embedding stay those of the real held-out image. Frame 0 = the unmodified pose.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "emernerf")); sys.path.insert(0, str(HERE.parent))
from viz3d_orbit import turbo, label, rodrigues, log  # noqa: E402


def look_at_cv(C, target, up):
    f = target - C; f = f / np.linalg.norm(f)
    r = np.cross(f, up); r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f, C], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--label", default="EmerNeRF"); ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--frames", type=int, default=180); ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--pivot", type=float, default=8.0); ap.add_argument("--lift", type=float, default=4.0)
    ap.add_argument("--drop", type=float, default=1.5); ap.add_argument("--dmax", type=float, default=80.0)
    ap.add_argument("--chunk", type=int, default=32768)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import emer_common
    from make_videos import Mp4Writer
    from PIL import Image
    T0 = time.time()
    cfg, ds, model, est, props, ckpt, step = emer_common.load_run(a.run_dir, None, a.chunk)
    import torch
    from radiance_fields.render_utils import render_rays
    ps = ds.pixel_source
    test = set(int(i) for i in ds.test_indices)
    cand = [i for i in range(ps.num_imgs) if ps.img_cam_names[i] == a.camera and i in test]
    assert cand, f"no held-out images of {a.camera}"
    cand.sort(key=lambda i: ps.img_abs_timesteps[i])
    k = cand[len(cand) // 2]; t_mid = int(ps.img_abs_timesteps[k])
    log(f"EmerNeRF step {step} ckpt {ckpt}; {a.camera} held-out t={t_mid} (img {k}); load {time.time() - T0:.0f}s")

    allc = ps.cam_to_worlds.detach().cpu().numpy().astype(np.float64)
    up = -allc[:, :3, 1].mean(0); up /= np.linalg.norm(up)              # OpenCV y = down
    c2w_orig = ps.cam_to_worlds[k].clone()
    c2w0 = c2w_orig.cpu().numpy().astype(np.float64)
    C0, fwd = c2w0[:3, 3], c2w0[:3, 2]
    fh = fwd - np.dot(fwd, up) * up; fh /= np.linalg.norm(fh)
    P = C0 + a.pivot * fh
    poses = []
    for i in range(a.frames):
        th = 2 * math.pi * i / a.frames; s = math.sin(th / 2) ** 2
        C = P + rodrigues(C0 - P, up, th) + a.lift * s * up
        poses.append(look_at_cv(C, P - a.drop * s * up, up))
    ps.downscale_factor = a.scale
    tag = a.camera.replace("camera_", "")
    writer, sheet, f0, dmed = None, [], None, []
    T1 = time.time()
    try:
        with torch.no_grad():
            for i in range(a.frames):
                M = c2w_orig.clone()
                if i > 0:
                    M[:3, :4] = torch.from_numpy(poses[i]).to(M)
                ps.cam_to_worlds[k] = M
                data = {kk: v.cuda(non_blocking=True) for kk, v in ps.get_render_rays(k).items()}
                res = render_rays(radiance_field=model, proposal_estimator=est, proposal_networks=props,
                                  data_dict=data, cfg=cfg, return_decomposition=False)
                rgb = res["rgb"].clamp(0, 1).cpu().numpy()
                d = res["depth"].cpu().numpy().reshape(rgb.shape[:2])
                valid = np.isfinite(d) & (d > 0)
                dmed.append(float(np.median(d[valid])) if valid.any() else None)
                deg = 360.0 * i / a.frames
                fr = np.concatenate([label((rgb * 255).astype(np.uint8), f"{a.label} | {a.clip} {tag} t={t_mid} | orbit {deg:5.1f} deg | RGB"),
                                     label((turbo(d, valid, a.dmax) * 255).astype(np.uint8), f"expected depth (log 1-{a.dmax:.0f} m, red = near, blue = far/empty)")], 1)
                fr = fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2]
                if writer is None:
                    writer = Mp4Writer(a.out / f"orbit360_{tag}.mp4", fr.shape[1], fr.shape[0], a.fps)
                writer.append(fr)
                if i == 0:
                    f0, gt = rgb, data["pixels"].cpu().numpy()
                if i % max(1, a.frames // 8) == 0 and len(sheet) < 8:
                    sheet.append(fr)
                if i % 30 == 0:
                    log(f"frame {i}/{a.frames} ({(time.time() - T1) / (i + 1):.2f} s/frame)")
    finally:
        ps.cam_to_worlds[k] = c2w_orig
    writer.close()
    h, w = sheet[0].shape[:2]
    cs = np.zeros(((len(sheet) + 1) // 2 * h, 2 * w, 3), np.uint8)
    for j, fr in enumerate(sheet):
        cs[(j // 2) * h:(j // 2 + 1) * h, (j % 2) * w:(j % 2 + 1) * w] = fr
    Image.fromarray(cs).resize((cs.shape[1] // 2, cs.shape[0] // 2)).save(a.out / f"contact_sheet_{tag}.jpg", quality=88)
    psnr0 = float(-10 * np.log10(np.mean((gt - f0) ** 2) + 1e-12))
    Image.fromarray(np.concatenate([label((gt * 255).astype(np.uint8), f"GT held-out {a.clip} {tag} t={t_mid}"),
                                    label((f0 * 255).astype(np.uint8), f"{a.label} render (orbit frame 0) PSNR {psnr0:.2f} dB @ {a.scale}x")], 1)
                    ).save(a.out / f"frame0_vs_gt_{tag}.jpg", quality=90)
    dm = [x for x in dmed if x is not None]
    (a.out / "meta.json").write_text(json.dumps({
        "label": a.label, "clip": a.clip, "camera": a.camera, "heldout_timestep": t_mid, "run_dir": str(a.run_dir),
        "ckpt": str(ckpt), "step": int(step), "model": "EmerNeRF", "frames": a.frames, "fps": a.fps, "scale": a.scale,
        "pivot_m": a.pivot, "lift_m": a.lift, "world_up": up.tolist(), "frame0_psnr_vs_gt_db": psnr0,
        "depth_median_m_min_max": [min(dm), max(dm)] if dm else None, "render_s_per_frame": (time.time() - T1) / a.frames,
        "note": "frame 0 = real held-out camera; other frames are novel views (no GT)."}, indent=1))
    log(f"DONE ({time.time() - T0:.0f}s); frame0 PSNR {psnr0:.2f} dB")


if __name__ == "__main__":
    main()
