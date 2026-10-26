#!/usr/bin/env python3
"""A4 (camera half): per-clip Koschmieder beta_C + airlight A from the GT foggy frames and
NeuRAD rendered depth (12 held-out front-wide frames per clip).

Dark-channel prior: for haze-free dark pixels, I_dark(x) ~= A * (1 - exp(-beta * d(x))).
A = mean RGB over the brightest 0.1% dark-channel pixels. Fit y = -ln(1 - I_dark / A) vs d
(2 m depth bins, median per bin, line fit WITH intercept; slope = beta_C). Pixels: central
crop, accumulation > 0.98, 8 < d < 100 m. Clear clip NCF-02 = control (expect ~0).
NFF-05 is a NIGHT clip: no daylight airlight; its value is recorded but flagged unusable.
"""
import csv, glob, json
import numpy as np
from scipy.ndimage import minimum_filter

FF = '/capstor/scratch/cscs/yitiwang/foggyfields'
rows = []
for clip in ['NCF-02', 'NFF-05', 'NFF-13', 'NFF-15']:
    betas, As = [], []
    for f in sorted(glob.glob(f'{FF}/analysis/depth_renders/{clip}/frame_*.npz')):
        d = np.load(f, allow_pickle=True)
        depth = d['depth'].astype(np.float32)
        acc = d['accumulation'].astype(np.float32)
        H, W = depth.shape
        gtp = f.replace('.npz', '_gtcrop.npy')  # safe-cropped GT at the ray grid (validate fix)
        if not __import__('os').path.exists(gtp):
            continue
        gt = np.load(gtp)
        if gt.shape[:2] != (H, W):   # render grid vs eval-trimmed grid off-by-one guard
            m0, m1 = min(H, gt.shape[0]), min(W, gt.shape[1])
            gt, depth, acc = gt[:m0, :m1], depth[:m0, :m1], acc[:m0, :m1]
            H, W = m0, m1
        dark = minimum_filter(gt.min(-1), size=7)
        hb, wb = int(H * 0.08), int(W * 0.08)
        sl = np.s_[hb:H - hb, wb:W - wb]
        dark_c, depth_c, acc_c = dark[sl], depth[sl], acc[sl]
        A = float(np.mean(np.sort(dark_c.ravel())[-max(dark_c.size // 1000, 50):]))
        A = max(A, 0.05)
        ok = (acc_c > 0.98) & (depth_c > 8) & (depth_c < 100) & (dark_c < 0.95 * A)
        if ok.sum() < 5000:
            continue
        y = -np.log(np.clip(1.0 - dark_c[ok] / A, 1e-3, None))
        x = depth_c[ok]
        bins = np.arange(8, 100, 2.0)
        idx = np.digitize(x, bins) - 1
        bx, by = [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() >= 100:
                bx.append(0.5 * (bins[b] + bins[b + 1])); by.append(np.median(y[m]))
        if len(bx) < 10:
            continue
        slope, c0 = np.polyfit(bx, by, 1)
        betas.append(slope); As.append(A)
    b = np.array(betas)
    rows.append(dict(clip=clip, n_frames=len(b), beta_C=round(float(b.mean()), 5),
                     beta_C_std=round(float(b.std()), 5), airlight_A=round(float(np.mean(As)), 3),
                     mor_m=round(3 / b.mean(), 0) if b.mean() > 1e-4 else '',
                     note='NIGHT - unusable airlight' if clip == 'NFF-05' else
                          ('clear control' if clip == 'NCF-02' else 'day fog')))
    print(rows[-1], flush=True)
with open(f'{FF}/analysis/A4/A4_beta_camera.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('saved')
