#!/usr/bin/env python3
"""A4 (LiDAR half): physics-only beta_L anchors per fog clip (plan v2 §4.2).

Road returns from the NeuRAD dump rasters (sensor frame): z = r*sin(elevation), road =
|z - z_ground_sensor| < 0.3 m, 5 < r < 60 m, intensity > 0. Fit ln(I * r^2) = c - 2*beta*r.
Two forms: plain, and with the Lambertian grazing-incidence term (ln(I * r^2) = c' - ln r - 2*beta*r);
the spread between them is systematic uncertainty. 95% CI by bootstrap over sweeps.
NCF-02 is the clear-sky control: its fitted beta is the calibration/incidence floor, and the
anchor for the fog medium is beta_L_fog - beta_L_clear (same sensor, similar road geometry).
"""
import csv, glob, os
import numpy as np

W = '/capstor/scratch/cscs/yitiwang/ncf-crop-ablation'
OUT = os.path.dirname(os.path.abspath(__file__))
ZG = {'NCF-02': -2.2, 'NFF-05': -2.16, 'NFF-13': -2.16, 'NFF-15': -2.16}  # ground_z_sensor_m (same fleet mount)


def fit(r, y, w=None):
    X = np.stack([np.ones_like(r), r], 1)
    sol, *_ = np.linalg.lstsq(X if w is None else X * w[:, None], y if w is None else y * w, rcond=None)
    return sol  # [c, slope]; beta = -slope/2


rows = []
for c, zg in ZG.items():
    files = sorted(glob.glob(f'{W}/outputs_fix/_lidar_dump/neurad_safe__{c}/frame_*.npz'))
    per_sweep = {'plain': [], 'incid': []}
    nsel = 0
    zs = []
    for f in files:
        d = np.load(f)
        ret = d['gt_did_return'] & (d['gt_range'] < 1000) & (d['gt_intensity'] > 0)
        r = d['gt_range'][ret]; el = d['elevation'][ret] * np.pi / 180 if d['elevation'].max() > 3.2 else d['elevation'][ret]
        z = r * np.sin(el); i = d['gt_intensity'][ret]
        if zg is None:
            lowm = (r > 3) & (r < 25)
            h, e = np.histogram(z[lowm], bins=np.arange(-4, 1, 0.1))
            zg_est = e[np.argmax(h)] + 0.05
            zs.append(zg_est)
            zgl = zg_est
        else:
            zgl = zg
        road = (np.abs(z - zgl) < 0.3) & (r > 5) & (r < 60)
        if road.sum() < 2000:
            continue
        rr, ii = r[road], i[road]
        y = np.log(ii * rr ** 2)
        per_sweep['plain'].append(-fit(rr, y)[1] / 2)
        per_sweep['incid'].append(-fit(rr, y + np.log(rr))[1] / 2)
        per_sweep.setdefault('refl', []).append(-fit(rr, np.log(ii))[1] / 2)  # intensity already range-compensated
        nsel += road.sum()
    bp = np.array(per_sweep['plain']); bi = np.array(per_sweep['incid']); br = np.array(per_sweep['refl'])
    boot = [np.mean(np.random.default_rng(k).choice(bp, len(bp))) for k in range(500)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    row = dict(clip=c, n_sweeps=len(bp), n_road_returns=int(nsel),
               z_ground_used=round(float(np.median(zs)) if zs else zg, 2),
               beta_L_plain=round(float(bp.mean()), 5), beta_L_plain_ci=f'[{lo:.5f},{hi:.5f}]',
               beta_L_incidence=round(float(bi.mean()), 5),
               beta_L_refl=round(float(br.mean()), 5),
               sweep_std=round(float(bp.std()), 5))
    rows.append(row); print(row, flush=True)

clear = next(r for r in rows if r['clip'] == 'NCF-02')
for r in rows:
    # the r^2 / incidence terms cancel in the fog-minus-clear slope difference:
    r['beta_L_anchor_minus_clear'] = round(r['beta_L_plain'] - clear['beta_L_plain'], 5)
    r['mor_m_koschmieder'] = round(3.0 / r['beta_L_anchor_minus_clear'], 0) if r['beta_L_anchor_minus_clear'] > 1e-4 else ''
with open(f'{OUT}/A4_beta_lidar.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('saved A4_beta_lidar.csv')
