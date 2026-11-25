#!/usr/bin/env python3
"""Common-evaluator adapter for a FoggyField V1.3 run (nerfstudio method neurad-fog-v13).

Writes exactly what the crop-ablation bench evaluators read:
  <out>/renders/val/rgb/<camera>/<frame>.png      model output I_out (foggy_rgb_srgb, the camera-response-corrected
                                                   final image; key selectable) for EVERY held-out image
  <out>/renders/val/gt-rgb/<camera>/<frame>.png   the held-out image as the model's dataset serves it
      -> jobs/eval_common_region.py --variant safe (the V1.3 dataparser runs with apply_safe_crop=True: the images ARE the
         safe_crop rectangle, 1 px trimmed by image_divisible_by; eval_common_region tolerates <= 8 px)
  <out>/lidar_dump/frame_XXXX.npz                 returns_only schema of jobs/emernerf/lidar_dump_emernerf.py: one predicted
                                                   point per GT-returned ray, LiDAR sensor frame
      -> jobs/emernerf/lidar_metrics_from_dump.py --eval-ray-set returns_only  (cd_returns_m, cd2_returns_m2,
         depth_median_l2_returns_m). The model has no ray-drop head: no drop is predicted ("oracle drop" = the GT return set).
      pred_range        = api 'range'       (detected range: hard-peak of the pulsed waveform, decision 100; fog echoes included)
      pred_median_range = api 'range_scene' (scene-only expected range, geometry without fog) -> cd_median_pc_returns_m
Resumable: existing PNG / npz files are kept. Needs the v13 env (jobs/v13uni_env.sh or v13mv_env.sh) inside the container.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_png(arr01, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(arr01, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    tmp = path.with_suffix(".tmp.png"); im.save(tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rgb-key", default="foggy_rgb_srgb")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-sweeps", type=int, default=0)
    ap.add_argument("--skip-rgb", action="store_true")
    ap.add_argument("--skip-lidar", action="store_true")
    a = ap.parse_args()

    from nerfstudio.models.fog_v13 import api  # noqa: E402
    t0 = time.time()
    pipeline, model, step = api.load(a.config)
    dm = pipeline.datamanager
    man = {"config": str(a.config), "step": int(step), "rgb_key": a.rgb_key, "code": os.path.dirname(os.path.dirname(api.__file__)),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "load_s": time.time() - t0}
    print(f"[v13_score] loaded step {step} from {a.config} ({man['load_s']:.0f}s)", flush=True)

    # ---------------------------------------------------------------- cameras
    names = [Path(f) for f in dm.eval_dataset._dataparser_outputs.image_filenames]
    n_img = len(names) if not a.max_images else min(a.max_images, len(names))
    rgb_root, gt_root = a.out / "renders/val/rgb", a.out / "renders/val/gt-rgb"
    if not a.skip_rgb:
        t1 = time.time(); done = 0
        for i in range(n_img):
            cam, frame = names[i].parent.name, names[i].stem
            pp, gp = rgb_root / cam / f"{frame}.png", gt_root / cam / f"{frame}.png"
            if pp.is_file() and gp.is_file():
                continue
            res = api.render_camera(model, pipeline, i, medium="foggy")
            if a.rgb_key not in res:
                raise KeyError(f"{a.rgb_key} not in render_camera output; have {sorted(k for k in res if k != 'meta')}")
            save_png(res[a.rgb_key], pp); save_png(res["observed_srgb"], gp)
            done += 1
            if done % 20 == 0 or i == n_img - 1:
                print(f"  rgb {i + 1}/{n_img} ({(time.time() - t1) / max(done, 1):.1f} s/img)", flush=True)
            del res
        n_png = sum(1 for _ in rgb_root.rglob("*.png"))
        man.update(n_eval_images=len(names), n_rendered_images=n_png, rgb_seconds=time.time() - t1,
                   render_hw=[int(x) for x in np.asarray(Image.open(next(rgb_root.rglob('*.png')))).shape[:2]])
        print(f"[v13_score] rgb done: {n_png} png ({man['rgb_seconds']:.0f}s)", flush=True)

    # ---------------------------------------------------------------- lidar
    if not a.skip_lidar:
        dl = dm.fixed_indices_eval_lidar_dataloader
        n_sw = len(dm.eval_lidar_dataset)
        if a.max_sweeps:
            n_sw = min(a.max_sweeps, n_sw)
        dump = a.out / "lidar_dump"; dump.mkdir(parents=True, exist_ok=True)
        t1 = time.time(); frames = []
        for idx in range(n_sw):
            fp = dump / f"frame_{idx:04d}.npz"
            if fp.is_file():
                frames.append(fp.name); continue
            lidar, batch = dl.get_camera(idx)
            l2w = lidar.lidar_to_worlds[0].detach().double().cpu().numpy()  # [3,4] sensor -> world
            lid = api.render_lidar(model, pipeline, idx, medium="foggy", profile_rays=0)
            did = lid["did_return"].astype(bool)
            r_obs = lid["r_obs"].astype(np.float64)
            R = did & np.isfinite(r_obs) & (r_obs > 0) & (r_obs < 1000.0)
            o, d = lid["origins"].astype(np.float64)[R], lid["directions"].astype(np.float64)[R]
            d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            rng = lid["range"].astype(np.float64)[R]
            rng_scene = lid["range_scene"].astype(np.float64)[R]
            r = r_obs[R]
            Rw, tw = l2w[:3, :3], l2w[:3, 3]
            to_s = lambda pw: (pw - tw) @ Rw           # world -> sensor (R^T (p - t))
            gt_s = to_s(o + d * r[:, None]); pr_s = to_s(o + d * rng[:, None]); pm_s = to_s(o + d * rng_scene[:, None])
            d_s = gt_s / np.maximum(np.linalg.norm(gt_s, axis=1, keepdims=True), 1e-9)
            f32 = lambda x: np.asarray(x, dtype=np.float32)
            tmp = fp.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, pred_xyz=f32(pr_s), gt_xyz=f32(gt_s), pred_median_xyz=f32(pm_s),
                                raster_shape=np.array([len(r)]),
                                azimuth=f32(np.degrees(np.arctan2(d_s[:, 1], d_s[:, 0]))),
                                elevation=f32(np.degrees(np.arcsin(np.clip(d_s[:, 2], -1, 1)))),
                                gt_range=f32(r), gt_intensity=f32(lid["intensity_obs"][R]),
                                pred_range=f32(rng), pred_median_range=f32(rng_scene),
                                pred_intensity=f32(lid["intensity"][R]) if "intensity" in lid else f32(np.zeros(len(r))),
                                lidar_to_world=f32(l2w[:3, :4]),
                                time_s=np.array(lid["meta"].get("time_s") or 0.0, dtype=np.float64))
            os.replace(tmp, fp); frames.append(fp.name)
            if (idx + 1) % 10 == 0 or idx == n_sw - 1:
                print(f"  lidar {idx + 1}/{n_sw}: {int(R.sum())} GT returns of {len(did)} beams "
                      f"({(time.time() - t1) / (idx + 1):.1f} s/sweep)", flush=True)
            del lid
        man.update(n_eval_sweeps=len(dm.eval_lidar_dataset), n_dumped_sweeps=len(frames), lidar_seconds=time.time() - t1,
                   lidar_meta_flags=lid["meta"].get("flags") if 'lid' in dir() else None,
                   eval_ray_set="returns_only", ray_drop="none predicted (no ray-drop head): oracle GT return set",
                   pred_range="api 'range' (detected hard-peak range incl. fog echoes)",
                   pred_median_range="api 'range_scene' (scene-only expected range)", gt_xyz_frame="LiDAR sensor frame (inv lidar_to_world)",
                   frame_naming="frame_<eval position>.npz (eval sweep order = odd timesteps 1..197)")
        (dump / "DUMP_MANIFEST.json").write_text(json.dumps({k: v for k, v in man.items()}, indent=2))
        print(f"[v13_score] lidar done: {len(frames)} sweeps ({man['lidar_seconds']:.0f}s)", flush=True)

    man["params"] = {k: v for k, v in api.params(model).items() if k in ("step", "stage", "beta0", "beta_field", "eta", "flags")}
    (a.out / "SCORE_MANIFEST.json").write_text(json.dumps(man, indent=2, default=str))
    print("V13_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
