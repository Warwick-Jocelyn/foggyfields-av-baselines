#!/usr/bin/env python3
"""Render a trained EmerNeRF run's HELD-OUT frames into the exact tree that
`ns-render dataset --rendered-output-names rgb gt-rgb --pose-source val` writes, so that
jobs/eval_common_region.py and jobs/make_videos.py consume it unchanged:

    <out>/rgb/<camera_name>/frame_<TTTT>.png       prediction
    <out>/gt-rgb/<camera_name>/frame_<TTTT>.png    ground truth (the training pixels)

<out> is what lane.sh passes as --renders (i.e. "<run>/renders/val"). TTTT is the ABSOLUTE
clip timestep, the same stem the source image frame_TTTT.jpg and SplatAD's renders carry.

Camera rays come from the dataset's cam_to_worlds, i.e. the training loader's own code:
F1b (pose at the camera's exposure time) applies here exactly as in training, and a run
whose config.yaml predates F1b is rendered with its pre-fix spin-centre poses
(emer_common.load_run).

Images are the camera's safe_crop rectangle at native resolution (the model is trained in
native_resolution mode), so no resampling happens anywhere between model and metric.

Checks (fail loudly):
  * the held-out timesteps come from the dataset's split, and are re-read here from
    splits/<split_json>.json and compared;
  * every render is exactly safe_crop.w x safe_crop.h, read independently from
    calib/K_rectified_v2.json;
  * the GT PNG equals the source JPEG cropped to safe_crop, byte for byte (spot-checked on
    the first timestep of every camera).
A manifest (render_manifest.json) records all of it plus wall time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emer_common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="EmerNeRF log_dir (config.yaml + checkpoint_*.pth)")
    ap.add_argument("--ckpt", type=Path, default=None, help="default: highest-step checkpoint in --run-dir")
    ap.add_argument("--out", type=Path, required=True, help="<run>/renders/val")
    ap.add_argument("--max-eval-frames", type=int, default=0, help="render only N evenly spaced held-out timesteps")
    ap.add_argument("--chunk", type=int, default=32768, help="rays per render chunk")
    a = ap.parse_args()

    t_load = time.time()
    cfg, ds, model, est, props, ckpt, step = emer_common.load_run(a.run_dir, a.ckpt, a.chunk)
    t_load = time.time() - t_load
    import torch
    from radiance_fields.render_utils import render_rays

    ps = ds.pixel_source
    if not ps.native:
        raise SystemExit("this run was not trained with pixel_source.native_resolution=true; "
                         "its renders would not be safe_crop-sized")
    clip_root = Path(ds.data_path)

    # ---- held-out timesteps: dataset split vs the split json, independently ----------
    all_held, todo = emer_common.heldout_timesteps(ds, a.max_eval_frames)
    sj = json.loads(Path(ds.split_json_path).read_text())
    want = sorted(int(t) for t in sj["eval_indices"] if ds.start_timestep <= int(t) < ds.end_timestep)
    if all_held != want:
        raise AssertionError(f"dataset held-out timesteps differ from {ds.split_json_path}")
    if set(all_held) & set(int(t) for t in sj["train_indices"]):
        raise AssertionError("held-out timesteps overlap train_indices")

    # ---- safe_crop geometry, read independently of the dataset class -----------------
    k2 = json.loads((clip_root / "calib" / "K_rectified_v2.json").read_text())
    sc = {c: k2[c]["safe_crop"] for c in k2}

    idx_of = {(ps.img_abs_timesteps[i], ps.img_cam_names[i]): i for i in range(ps.num_imgs)}
    test_set = set(ds.test_indices)
    (a.out / "rgb").mkdir(parents=True, exist_ok=True)
    (a.out / "gt-rgb").mkdir(parents=True, exist_ok=True)

    n, rays, t_render = 0, 0, 0.0
    per_cam_size, k_used = {}, {}
    gt_checked = set()
    with torch.no_grad():
        for t in todo:
            for cam in ps._cam_names:
                i = idx_of[(t, cam)]
                if i not in test_set:
                    raise AssertionError(f"image {i} (t={t}, {cam}) is not in the dataset's test indices")
                data = ps.get_render_rays(i)
                data = {k: v.cuda(non_blocking=True) for k, v in data.items()}
                torch.cuda.synchronize()
                t0 = time.time()
                res = render_rays(radiance_field=model, proposal_estimator=est, proposal_networks=props,
                                  data_dict=data, cfg=cfg, return_decomposition=False)
                torch.cuda.synchronize()
                t_render += time.time() - t0
                rgb = res["rgb"].clamp(0, 1).cpu().numpy()
                gt = data["pixels"].cpu().numpy()
                h, w = rgb.shape[:2]
                if (w, h) != (int(sc[cam]["w"]), int(sc[cam]["h"])):
                    raise AssertionError(f"{cam}: render {w}x{h} != safe_crop {sc[cam]['w']}x{sc[cam]['h']}")
                pred8 = np.round(rgb * 255.0).astype(np.uint8)
                gt8 = np.round(gt * 255.0).astype(np.uint8)
                for sub, im in (("rgb", pred8), ("gt-rgb", gt8)):
                    d = a.out / sub / cam
                    d.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(im).save(d / f"frame_{t:04d}.png")
                if cam not in gt_checked:   # GT == source JPEG cropped to safe_crop
                    src = Image.open(clip_root / ps.images_dirname / cam / f"frame_{t:04d}.jpg").convert("RGB")
                    s = sc[cam]
                    ref = np.asarray(src.crop((s["x"], s["y"], s["x"] + s["w"], s["y"] + s["h"])))
                    if not np.array_equal(ref, gt8):
                        raise AssertionError(f"{cam} t={t}: GT pixels differ from the cropped source JPEG")
                    gt_checked.add(cam)
                    K = ps.intrinsics[i].cpu().numpy()
                    k_used[cam] = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
                                   "cy": float(K[1, 2]),
                                   "calib_fx": float(k2[cam]["fx"]), "calib_fy": float(k2[cam]["fy"]),
                                   "safe_crop_cx": float(s["cx"]), "safe_crop_cy": float(s["cy"])}
                    if not (np.isclose(K[0, 0], k2[cam]["fx"]) and np.isclose(K[1, 1], k2[cam]["fy"])
                            and np.isclose(K[0, 2], s["cx"]) and np.isclose(K[1, 2], s["cy"])):
                        raise AssertionError(f"{cam}: model K {k_used[cam]} is not (calib fx/fy, safe_crop cx/cy)")
                per_cam_size[cam] = [w, h]
                n += 1
                rays += h * w
            print(f"  t={t:4d}  {n} images  {rays / max(t_render, 1e-9) / 1e6:.2f} Mrays/s", flush=True)

    man = {
        "method": "emernerf", "run_dir": str(a.run_dir), "checkpoint": str(ckpt), "step": step,
        "clip_root": str(clip_root), "split_json": ds.split_json_path,
        "heldout_timesteps_split": all_held, "rendered_timesteps": todo,
        "truncated": len(todo) != len(all_held), "n_images": n, "cameras": list(ps._cam_names),
        "render_wh": per_cam_size, "intrinsics_used": k_used,
        "gt_equals_cropped_source_jpeg": sorted(gt_checked),
        "seconds_model_load": t_load, "seconds_render": t_render, "mrays_per_s": rays / max(t_render, 1e-9) / 1e6,
        "render_chunk_size": int(cfg.render.render_chunk_size),
        # F1b: get_render_rays uses ps.cam_to_worlds, built by the same loader code as in
        # training (camera pose at the camera's exposure time unless the run says "lidar")
        "camera_pose_time": ps.camera_pose_time,
    }
    (a.out.parent / "render_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"rendered {n} images ({len(todo)} of {len(all_held)} held-out timesteps x {len(ps._cam_names)} cams) "
          f"in {t_render:.0f} s render + {t_load:.0f} s load -> {a.out}")
    print("RENDER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
