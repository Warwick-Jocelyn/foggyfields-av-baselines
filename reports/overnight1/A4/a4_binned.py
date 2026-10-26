#!/usr/bin/env python3
"""A4 LiDAR anchors, binned common-support version: mean ln-intensity per 1 m range bin over
10-40 m (all clips have support there), equal-weight line fit on bin means. The fog-minus-clear
slope difference is then model-independent (any fixed function of r cancels exactly).
"""
import csv, glob
import numpy as np

W = '/capstor/scratch/cscs/yitiwang/ncf-crop-ablation'
OUT = '/capstor/scratch/cscs/yitiwang/foggyfields-baseline/analysis/A4'
ZG = -2.16
BINS = np.arange(10, 41, 1.0)
CENT = 0.5 * (BINS[:-1] + BINS[1:])

res = {}
for c in ['NCF-02', 'NFF-05', 'NFF-13', 'NFF-15']:
    files = sorted(glob.glob(f'{W}/outputs_fix/_lidar_dump/neurad_safe__{c}/frame_*.npz'))
    slopes = []
    for f in files:
        d = np.load(f)
        ret = d['gt_did_return'] & (d['gt_range'] < 1000) & (d['gt_intensity'] > 0)
        r = d['gt_range'][ret]; el = np.deg2rad(d['elevation'][ret])  # dump stores degrees
        z = r * np.sin(el); i = d['gt_intensity'][ret]
        road = (np.abs(z - ZG) < 0.3) & (r >= 10) & (r < 40)
        if road.sum() < 1000:
            continue
        rr, ii = r[road], np.log(i[road])
        idx = np.digitize(rr, BINS) - 1
        ym = np.full(len(CENT), np.nan)
        for b in range(len(CENT)):
            m = idx == b
            if m.sum() >= 30:
                ym[b] = ii[m].mean()
        ok = np.isfinite(ym)
        if ok.sum() < 15:
            continue
        slopes.append(np.polyfit(CENT[ok], ym[ok], 1)[0])
    s = np.array(slopes)
    res[c] = s
    print(c, 'n_sweeps', len(s), 'mean slope', round(s.mean(), 5), 'std', round(s.std(), 5))

rows = []
base = res['NCF-02']
for c, s in res.items():
    dslope = s.mean() - base.mean()
    beta = -dslope / 2
    se = np.sqrt(s.var() / len(s) + base.var() / len(base))
    rows.append(dict(clip=c, n_sweeps=len(s), slope=round(float(s.mean()), 5),
                     beta_anchor=round(float(beta), 5),
                     ci95=f'[{beta - se:.5f},{beta + se:.5f}]',
                     mor_m=round(3 / beta, 0) if beta > 1e-4 else ''))
    print(rows[-1])
with open(f'{OUT}/A4_beta_anchors.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
