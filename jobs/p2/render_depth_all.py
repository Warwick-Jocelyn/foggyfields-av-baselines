#!/usr/bin/env python3
"""T8: expected depth + accumulation at ray res for ALL train+eval frames (one camera subset
via --cam-mod/--cam-rem for multi-GPU splitting). Saves per frame:
<out>/<split>_<idx>.npz { depth f16 [H,W], acc f16 [H,W], crop i32 [4] (x,y,w,h), image_filename }.
Input for T9 synthetic fog (Koschmieder needs per-pixel depth inside the safe crop)."""
import argparse
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--config', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--cam-mod', type=int, default=1)
ap.add_argument('--cam-rem', type=int, default=0)
a = ap.parse_args()
from nerfstudio.utils.eval_utils import eval_setup
config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val')
model, dm = pipeline.model, pipeline.datamanager
model.eval()
a.out.mkdir(parents=True, exist_ok=True)
for split, ds in [('train', dm.train_dataset), ('eval', dm.eval_dataset)]:
    crops = ds.metadata.get('image_crops') if hasattr(ds, 'metadata') else None
    if crops is None:
        crops = ds._dataparser_outputs.metadata.get('image_crops')
    names = [str(f) for f in ds.image_filenames]
    cams = sorted({n.split('/')[-2] for n in names})
    sel_cams = {c for j, c in enumerate(cams) if j % a.cam_mod == a.cam_rem}
    done = 0
    with torch.no_grad():
        for i, n in enumerate(names):
            if n.split('/')[-2] not in sel_cams:
                continue
            f = a.out / f'{split}_{i:05d}.npz'
            if f.exists():
                continue
            out = model.get_outputs_for_camera(ds.cameras[i: i + 1].to(model.device))
            np.savez_compressed(f, depth=out['depth'][..., 0].cpu().numpy().astype(np.float16),
                                acc=out['accumulation'][..., 0].cpu().numpy().astype(np.float16),
                                crop=np.asarray(crops[i], dtype=np.int32),
                                image_filename=n)
            done += 1
            if done % 50 == 0:
                print(f'[t8] {split} {done}', flush=True)
    print(f'[t8] {split}: {done} frames rendered (cams {sorted(sel_cams)})', flush=True)
print('[t8] DONE', flush=True)
