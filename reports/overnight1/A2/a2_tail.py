#!/usr/bin/env python3
"""A2 tail decomposition of the NeuRAD baselines on fog clips (plan v2 §4.2).

Per GT-returned ray (gt range < 1000 m): err = |pred - gt| with pred = the benchmark's
rendered expected depth (pred_range; pred_median_range reported alongside). Decomposes the
>5 m tail into
  (a) fog-return-as-surface : GT in the near halo (< HALO m), predicted far behind it
  (b) surface-at-fog-range  : GT a real surface (>= HALO m), predicted into the halo (< HALO m)
  (c) other
HALO = 10 m (A1: the fog clips' near-range excess lives below ~10 m).
Outputs A2_tail.csv (per clip x range-bin stats + tail classes) + A2_tail.png.
"""
import csv, glob, os
import numpy as np

W = '/capstor/scratch/cscs/yitiwang/ncf-crop-ablation'
OUT = os.path.dirname(os.path.abspath(__file__))
CLIPS = ['NCF-02', 'NFF-05', 'NFF-13', 'NFF-15']
HALO = 10.0
BINS = [(0, 10), (10, 30), (30, 60), (60, 1000)]

rows, cdfs = [], {}
for c in CLIPS:
    files = sorted(glob.glob(f'{W}/outputs_fix/_lidar_dump/neurad_safe__{c}/frame_*.npz'))
    assert files, c
    gt_all, pred_all, predmed_all, inten_all = [], [], [], []
    for f in files:
        d = np.load(f)
        ret = d['gt_did_return'] & (d['gt_range'] < 1000)
        gt_all.append(d['gt_range'][ret]); pred_all.append(d['pred_range'][ret])
        predmed_all.append(d['pred_median_range'][ret]); inten_all.append(d['gt_intensity'][ret])
    gt = np.concatenate(gt_all); pred = np.concatenate(pred_all)
    pmed = np.concatenate(predmed_all); inten = np.concatenate(inten_all)
    err = np.abs(pred - gt); errm = np.abs(pmed - gt)
    tail5 = err > 5
    a = tail5 & (gt < HALO) & (pred > gt + 5)           # fog return predicted as surface
    b = tail5 & (gt >= HALO) & (pred < HALO)            # surface predicted at fog range
    o = tail5 & ~a & ~b
    base = dict(clip=c, n_rays=len(gt),
                med_err=round(float(np.median(err)), 3), med_err_median_depth=round(float(np.median(errm)), 3),
                frac_gt1m=round(float((err > 1).mean()), 4), frac_gt5m=round(float(tail5.mean()), 4),
                tail_a_fog_as_surface=round(float(a.sum() / max(tail5.sum(), 1)), 4),
                tail_b_surface_at_fog=round(float(b.sum() / max(tail5.sum(), 1)), 4),
                tail_c_other=round(float(o.sum() / max(tail5.sum(), 1)), 4),
                tail_gt_mean_intensity=round(float(inten[tail5].mean()) if tail5.any() else 0, 4),
                all_mean_intensity=round(float(inten.mean()), 4))
    for lo, hi in BINS:
        m = (gt >= lo) & (gt < hi)
        base[f'frac_gt1m_{lo}-{hi}'] = round(float((err[m] > 1).mean()) if m.any() else 0, 4)
        base[f'frac_gt5m_{lo}-{hi}'] = round(float((err[m] > 5).mean()) if m.any() else 0, 4)
        base[f'n_{lo}-{hi}'] = int(m.sum())
    rows.append(base); print(base, flush=True)
    q = np.quantile(err, np.linspace(0, 0.999, 200))
    cdfs[c] = (q, err, gt)

with open(f'{OUT}/A2_tail.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
colors = {'NCF-02': 'tab:green', 'NFF-05': 'tab:blue', 'NFF-13': 'tab:orange', 'NFF-15': 'tab:red'}
for c, (q, err, gt) in cdfs.items():
    ax[0].plot(q, np.linspace(0, 0.999, 200), color=colors[c], label=c)
    bc, fr1, fr5 = [], [], []
    for lo, hi in [(0,5),(5,10),(10,20),(20,30),(30,45),(45,60),(60,90),(90,150)]:
        m = (gt >= lo) & (gt < hi)
        if m.sum() > 500:
            bc.append((lo+hi)/2); fr1.append((err[m] > 1).mean()); fr5.append((err[m] > 5).mean())
    ax[1].plot(bc, fr1, 'o-', color=colors[c], label=c)
    ax[2].plot(bc, fr5, 'o-', color=colors[c], label=c)
ax[0].set(xlabel='abs range error (m)', ylabel='CDF', xlim=(0, 8), title='A2: error CDF (expected depth)')
ax[1].set(xlabel='GT range (m)', ylabel='frac err > 1 m', title='tail >1 m vs GT range')
ax[2].set(xlabel='GT range (m)', ylabel='frac err > 5 m', title='tail >5 m vs GT range')
for a_ in ax: a_.legend(); a_.grid(alpha=0.3)
fig.suptitle('A2 NeuRAD LiDAR tail decomposition (held-out sweeps, GT-returned rays)')
fig.tight_layout(); fig.savefig(f'{OUT}/A2_tail.png', dpi=130)
print('saved')
