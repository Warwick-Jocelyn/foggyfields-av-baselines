#!/usr/bin/env python3
"""Save safe-cropped GT at ray resolution next to the depth_render npz files (validate-agent
fix: PIL full-canvas loads were misaligned with the cropped ray grid). For each
analysis/depth_renders/<clip>/frame_*.npz, writes frame_*_gtcrop.npy (H, W, 3 float32 at the
eval ray grid = gt[1::3, 1::3])."""
import argparse, glob
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--config', type=Path, required=True)
ap.add_argument('--render-dir', type=Path, required=True)
a = ap.parse_args()
from nerfstudio.utils.eval_utils import eval_setup
config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val')
eds = pipeline.datamanager.eval_dataset
names = [str(f) for f in eds.image_filenames]
step3 = pipeline.model.config.rgb_upsample_factor
n = 0
for f in sorted(glob.glob(str(a.render_dir / 'frame_*.npz'))):
    if f.endswith('_gtcrop.npz'):
        continue
    d = np.load(f, allow_pickle=True)
    fn = str(d['image_filename'])
    i = names.index(fn)
    gt = eds.get_image_float32(i)[..., :3].numpy()[step3 // 2::step3, step3 // 2::step3]
    np.save(f.replace('.npz', '_gtcrop.npy'), gt.astype(np.float32))
    n += 1
print(f'[gtcrop] {n} frames -> {a.render_dir}', flush=True)
