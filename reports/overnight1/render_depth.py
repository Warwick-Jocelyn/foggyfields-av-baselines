#!/usr/bin/env python3
"""Task 0 smoke + A4 input: load a NeuRAD checkpoint, render RGB + expected depth for N held-out
frames of one camera, save npz (depth, metres along the ray per neurad's DepthRenderer) + pngs.

    python render_depth.py --config <run>/config.yml --out DIR [--camera camera_front_wide_120fov]
                           [--n-frames 12]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--n-frames", type=int, default=12)
    a = ap.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup
    t0 = time.time()
    config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode="val")
    print(f"[smoke] loaded {ckpt_path} (step {step}) in {time.time()-t0:.1f}s", flush=True)
    model, dm = pipeline.model, pipeline.datamanager
    ds = dm.eval_dataset
    names = [str(f) for f in ds.image_filenames]
    idxs = [i for i, n in enumerate(names) if a.camera in n][: a.n_frames]
    if not idxs:
        print(f"[smoke] !! no eval frames match camera {a.camera}; have e.g. {names[:3]}")
        sys.exit(1)
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None
    meta = []
    for j, i in enumerate(idxs):
        cam = ds.cameras[i : i + 1].to(model.device)
        with torch.no_grad():
            out = model.get_outputs_for_camera(cam)
        if j == 0:
            print("[smoke] output keys:", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}, flush=True)
        rgb = out["rgb"].clamp(0, 1).cpu().numpy()
        depth = out["depth"][..., 0].cpu().numpy() if "depth" in out else None
        acc = out["accumulation"][..., 0].cpu().numpy() if "accumulation" in out else None
        stem = Path(names[i]).stem
        np.savez_compressed(a.out / f"{stem}.npz", rgb=rgb.astype(np.float16),
                            depth=depth.astype(np.float32) if depth is not None else np.zeros(0),
                            accumulation=acc.astype(np.float16) if acc is not None else np.zeros(0),
                            image_filename=names[i])
        if Image is not None:
            Image.fromarray((rgb * 255).astype(np.uint8)).save(a.out / f"{stem}_rgb.png")
            if depth is not None:
                dn = np.clip(depth / 120.0, 0, 1)
                Image.fromarray((dn * 255).astype(np.uint8)).save(a.out / f"{stem}_depth.png")
        meta.append({"i": int(i), "file": names[i],
                     "depth_min": float(np.nanmin(depth)) if depth is not None else None,
                     "depth_med": float(np.nanmedian(depth)) if depth is not None else None,
                     "depth_max": float(np.nanmax(depth)) if depth is not None else None})
        print(f"[smoke] frame {j+1}/{len(idxs)} {stem} depth med "
              f"{meta[-1]['depth_med']}", flush=True)
    (a.out / "render_meta.json").write_text(json.dumps(
        {"config": str(a.config), "ckpt": str(ckpt_path), "step": int(step),
         "camera": a.camera, "frames": meta}, indent=1))
    print(f"[smoke] DONE -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
