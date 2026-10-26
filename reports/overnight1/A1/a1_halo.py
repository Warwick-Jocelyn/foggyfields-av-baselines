#!/usr/bin/env python3
"""A1 fog-halo statistics (plan v2 §4.2): return-range + intensity histograms, near-range return
fractions, per-ring stats, no-return fraction — fog clips vs the clear reference, all from
lidar/range_image/ (sensor-origin range, true no-return mask).

Outputs: A1_halo.csv (per clip x ring-group stats + overall), A1_hist.csv (histograms),
A1_halo.png. Every number comes from the range images; the point lists are not needed.
"""
import glob, json, os
import numpy as np

W = '/capstor/scratch/cscs/yitiwang/ncf-crop-ablation'
OUT = os.path.dirname(os.path.abspath(__file__))
CLIPS = {  # id -> clip dir
    'NCF-02': glob.glob(f'{W}/data_all/clips/NCF-02__*')[0],
    'NFF-05': glob.glob(f'{W}/data_nff/clips/NFF-05__*')[0],
    'NFF-13': glob.glob(f'{W}/data_nff/clips/NFF-13__*')[0],
    'NFF-15': glob.glob(f'{W}/data_nff/clips/NFF-15__*')[0],
}
RBINS = np.concatenate([np.arange(0, 30, 0.5), np.arange(30, 150, 2.0), [150, 200, 300]])
IBINS = np.linspace(0, 1, 51)
STRIDE = 2  # every 2nd sweep

rows, hists = [], {}
for cid, cdir in CLIPS.items():
    files = sorted(glob.glob(f'{cdir}/lidar/range_image/sweep_*.npz'))[::STRIDE]
    assert files, cid
    rhist = np.zeros(len(RBINS) - 1); ihist = np.zeros(len(IBINS) - 1)
    ihist_near = np.zeros(len(IBINS) - 1)  # intensity of returns < 10 m
    n_cells = n_ret = 0
    ring_ret = np.zeros(128); ring_lt5 = np.zeros(128); ring_lt10 = np.zeros(128)
    ring_cells = np.zeros(128); ring_int = np.zeros(128)
    frac_sweep_lt5 = []
    for f in files:
        d = np.load(f)
        rng = d['range_m'].astype(np.float32)
        inten = d['intensity'].astype(np.float32) / 255.0
        mask = np.unpackbits(d['mask'], axis=-1, count=3600).astype(bool)
        ret = mask & (rng > 0)
        r = rng[ret]; i = inten[ret]
        rhist += np.histogram(r, RBINS)[0]
        ihist += np.histogram(i, IBINS)[0]
        ihist_near += np.histogram(i[r < 10], IBINS)[0]
        n_cells += mask.size; n_ret += ret.sum()
        ring_ret += ret.sum(1); ring_cells += mask.shape[1]
        ring_lt5 += (ret & (rng < 5)).sum(1); ring_lt10 += (ret & (rng < 10)).sum(1)
        ring_int += np.where(ret, inten, 0).sum(1)
        frac_sweep_lt5.append(float((r < 5).sum()) / max(len(r), 1))
    tot = rhist.sum()
    centers = 0.5 * (RBINS[:-1] + RBINS[1:])
    med = float(np.interp(0.5, np.cumsum(rhist) / tot, centers))
    row = dict(clip=cid, n_sweeps=len(files), n_rays_grid=int(n_cells), n_returns=int(n_ret),
               occupancy=round(n_ret / n_cells, 4),
               frac_lt5=round(float(rhist[centers < 5].sum() / tot), 4),
               frac_lt10=round(float(rhist[centers < 10].sum() / tot), 4),
               frac_lt20=round(float(rhist[centers < 20].sum() / tot), 4),
               frac_gt60=round(float(rhist[centers > 60].sum() / tot), 4),
               median_range_m=round(med, 2),
               frac_lt5_sweep_std=round(float(np.std(frac_sweep_lt5)), 4),
               mean_intensity=round(float(ring_int.sum() / n_ret), 4))
    rows.append(row)
    hists[cid] = dict(rhist=rhist, ihist=ihist, ihist_near=ihist_near,
                      ring_ret=ring_ret, ring_cells=ring_cells, ring_lt5=ring_lt5,
                      ring_lt10=ring_lt10, ring_int=ring_int)
    print(row, flush=True)

import csv
with open(f'{OUT}/A1_halo.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]));  w.writeheader(); w.writerows(rows)
# per-ring csv
with open(f'{OUT}/A1_rings.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['clip', 'ring', 'occupancy', 'frac_lt5', 'frac_lt10', 'mean_intensity'])
    for cid, h in hists.items():
        for r in range(128):
            n = h['ring_ret'][r]
            w.writerow([cid, r, round(h['ring_ret'][r] / max(h['ring_cells'][r], 1), 4),
                        round(h['ring_lt5'][r] / max(n, 1), 4), round(h['ring_lt10'][r] / max(n, 1), 4),
                        round(h['ring_int'][r] / max(n, 1), 4)])
# histogram csv (density per metre)
with open(f'{OUT}/A1_hist.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['clip', 'bin_lo_m', 'bin_hi_m', 'density_per_m'])
    for cid, h in hists.items():
        dens = h['rhist'] / h['rhist'].sum() / np.diff(RBINS)
        for lo, hi, dv in zip(RBINS[:-1], RBINS[1:], dens):
            w.writerow([cid, lo, hi, f'{dv:.6g}'])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
colors = {'NCF-02': 'tab:green', 'NFF-05': 'tab:blue', 'NFF-13': 'tab:orange', 'NFF-15': 'tab:red'}
for cid, h in hists.items():
    dens = h['rhist'] / h['rhist'].sum() / np.diff(RBINS)
    ax[0, 0].semilogy(0.5 * (RBINS[:-1] + RBINS[1:]), dens, color=colors[cid], label=cid)
    ax[0, 1].semilogy(0.5 * (IBINS[:-1] + IBINS[1:]), h['ihist'] / h['ihist'].sum(), color=colors[cid], label=cid)
    ax[0, 1].semilogy(0.5 * (IBINS[:-1] + IBINS[1:]), np.where(h['ihist_near'] > 0, h['ihist_near'], np.nan) / max(h['ihist_near'].sum(), 1),
                      color=colors[cid], ls=':', alpha=0.7)
    ax[1, 0].plot(np.arange(128), h['ring_ret'] / np.maximum(h['ring_cells'], 1), color=colors[cid], label=cid)
    ax[1, 1].plot(np.arange(128), h['ring_lt10'] / np.maximum(h['ring_ret'], 1), color=colors[cid], label=cid)
ax[0, 0].set(xlabel='return range from sensor (m)', ylabel='density / m', title='A1: return-range distribution', xlim=(0, 150))
ax[0, 0].axvspan(0, 10, color='gray', alpha=0.1)
ax[0, 1].set(xlabel='intensity (0-1)', ylabel='fraction', title='intensity (solid: all; dotted: returns < 10 m)')
ax[1, 0].set(xlabel='ring', ylabel='occupancy', title='grid occupancy per ring (raw grid incl. structural non-fires)')
ax[1, 1].set(xlabel='ring', ylabel='frac of returns < 10 m', title='near-range (halo) fraction per ring')
for a in ax.flat: a.legend(fontsize=8); a.grid(alpha=0.3)
fig.suptitle('A1 fog-halo statistics - range images (sensor origin), every 2nd sweep')
fig.tight_layout(); fig.savefig(f'{OUT}/A1_halo.png', dpi=130)
print('saved', OUT)
