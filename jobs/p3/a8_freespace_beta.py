#!/usr/bin/env python3
"""A8 (advisor insight): the fog-trained NeRF already CONTAINS beta - read it out of the
free-space transmittance. NeuRAD is SDF-alpha (per-sample alpha, spacing-independent), so the
per-metre extinction is not a direct density read; instead, per camera ray with a far surface:

    beta_hat = -ln(T_pre_surface) / s_surface

where T_pre_surface = 1 - sum(w_i before the surface peak window) from the model's own
compositing weights, and s_surface = the surface sample depth. The CLEAR checkpoint gives the
method floor; fog ckpt minus clear floor = the baked-in fog extinction, comparable to the
dark-channel anchor (0.0321 on NFF-15) WITHOUT any GT.

Usage: a8_freespace_beta.py --config <B0 config.yml> --out DIR [--n-frames 12]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(0)

ap = argparse.ArgumentParser()
ap.add_argument('--config', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--tag', default='')
ap.add_argument('--n-frames', type=int, default=12)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
from nerfstudio.utils.eval_utils import eval_setup
config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val')
model, dm = pipeline.model, pipeline.datamanager
model.eval(); dev = model.device
eds = dm.eval_dataset
names = [str(f) for f in eds.image_filenames]
eidx = [i for i, n in enumerate(names) if 'front_wide' in n]
eidx = eidx[:: max(len(eidx) // a.n_frames, 1)][: a.n_frames]

betas = []
with torch.no_grad():
    for i in eidx:
        ws, sm = [], []
        h1 = model.renderer_feat.register_forward_hook(
            lambda m_, ar, kw, out, ws=ws: ws.append(kw['weights'].detach()), with_kwargs=True)
        h2 = model.field.register_forward_hook(
            lambda m_, inp, out, sm=sm: sm.append(((inp[0].frustums.starts + inp[0].frustums.ends) * 0.5)[..., 0].detach()))
        _ = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
        h1.remove(); h2.remove()
        w = torch.cat(ws)[..., 0]           # [R, S] sky-absorbed
        s = torch.cat(sm)                   # [R, S]
        # surface = argmax weight (excluding the sky column); window +-1 sample
        w_nosky = w[:, :-1]
        k = w_nosky.argmax(dim=1)
        R = w.shape[0]
        ar = torch.arange(w_nosky.shape[1], device=dev)[None, :]
        pre = (ar < (k[:, None] - 1)).float()
        mass_pre = (w_nosky * pre).sum(1)
        s_surf = s.gather(1, k[:, None])[:, 0]
        surf_w = w_nosky.gather(1, k[:, None])[:, 0]
        # usable rays: definite far surface, enough path, surface dominant
        ok = (s_surf > 15) & (s_surf < 80) & (surf_w > 0.3) & (mass_pre < 0.9)
        bh = -torch.log((1 - mass_pre[ok]).clamp(min=1e-3)) / s_surf[ok]
        betas.append(bh.cpu())
b = torch.cat(betas).numpy()
res = dict(config=str(a.config), step=int(step), tag=a.tag, n_rays=int(len(b)),
           beta_freespace_median=float(np.median(b)), beta_freespace_mean=float(b.mean()),
           beta_p25=float(np.percentile(b, 25)), beta_p75=float(np.percentile(b, 75)))
(a.out / f'a8_{a.tag}.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1), flush=True)
