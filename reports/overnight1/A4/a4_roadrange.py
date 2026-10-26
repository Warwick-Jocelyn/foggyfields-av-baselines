#!/usr/bin/env python3
"""A4 anchor via road visible range (censoring-robust): per sweep, the 95th/99th percentile of
road-return range. Detection-limit model (raw power ~ r^-2 e^{-2 beta r} at threshold, clear as
reference): beta = ln(r_clear / r_fog) / r_fog. CI by bootstrap over sweeps.
"""
import csv, glob
import numpy as np

W = '/capstor/scratch/cscs/yitiwang/ncf-crop-ablation'
OUT = '/capstor/scratch/cscs/yitiwang/foggyfields-baseline/analysis/A4'
ZG = -2.16

per = {}
for c in ['NCF-02', 'NFF-05', 'NFF-13', 'NFF-15']:
    p95s, p99s = [], []
    for f in sorted(glob.glob(f'{W}/outputs_fix/_lidar_dump/neurad_safe__{c}/frame_*.npz')):
        d = np.load(f)
        ret = d['gt_did_return'] & (d['gt_range'] < 1000)
        r = d['gt_range'][ret]; el = np.deg2rad(d['elevation'][ret])
        z = r * np.sin(el)
        road = (np.abs(z - ZG) < 0.3) & (r > 5)
        if road.sum() < 500:
            continue
        p95s.append(np.percentile(r[road], 95)); p99s.append(np.percentile(r[road], 99))
    per[c] = (np.array(p95s), np.array(p99s))
    print(c, 'road range p95', round(per[c][0].mean(), 1), 'p99', round(per[c][1].mean(), 1))

rc = per['NCF-02'][0].mean()
rows = []
for c, (p95, p99) in per.items():
    rf = p95.mean()
    beta = max(np.log(rc / rf) / rf, 0.0)
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(500):
        rfb = rng.choice(p95, len(p95)).mean(); rcb = rng.choice(per['NCF-02'][0], len(per['NCF-02'][0])).mean()
        boot.append(max(np.log(rcb / rfb) / rfb, 0.0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append(dict(clip=c, road_r95_m=round(float(rf), 1), road_r99_m=round(float(p99.mean()), 1),
                     beta_anchor_roadrange=round(float(beta), 5), ci95=f'[{lo:.5f},{hi:.5f}]',
                     mor_m=round(3 / beta, 0) if beta > 1e-4 else ''))
    print(rows[-1])
with open(f'{OUT}/A4_roadrange.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
