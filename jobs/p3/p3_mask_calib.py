#!/usr/bin/env python3
"""T-P3-1 calibration: choose s_halo/ground_tol so that
  (1) retention of CLEAR NCF-02 returns >= 98%  (A1: 24% of clear returns are < 10 m - road!)
  (2) capture of NFF-15 A2 class-(a) rays (fog echo read as surface: gt<10, pred>gt+5) is maximal.
Uses the existing NeuRAD eval dumps (gt ranges + elevations, degrees)."""
import glob
import numpy as np
import torch
import sys
sys.path.insert(0, '/capstor/scratch/cscs/yitiwang/foggyfields/medium')
from foggyfields.trusted import trusted_returns

W = '/capstor/scratch/cscs/yitiwang/ncf-crop-ablation'

def load(clip, n=25):
    zs, els, preds = [], [], []
    for f in sorted(glob.glob(f'{W}/outputs_fix/_lidar_dump/neurad_safe__{clip}/frame_*.npz'))[:n]:
        d = np.load(f)
        ret = d['gt_did_return'] & (d['gt_range'] < 1000)
        zs.append(d['gt_range'][ret]); els.append(np.deg2rad(d['elevation'][ret]))
        preds.append(d['pred_range'][ret])
    return (torch.from_numpy(np.concatenate(zs)), torch.from_numpy(np.concatenate(els)),
            torch.from_numpy(np.concatenate(preds)))

zc, elc, _ = load('NCF-02')
zf, elf, pf = load('NFF-15')
cls_a = (zf < 10) & (pf > zf + 5)          # A2 class (a): fog echo read as surface
print(f'clear rays {len(zc)}, fog rays {len(zf)}, class-a {int(cls_a.sum())} ({cls_a.float().mean():.4f})')
print(f'{"s_halo":>7} {"g_tol":>6} | {"clear retention":>15} | {"class-a capture":>15} | {"fog untrusted frac":>18}')
best = None
for s_halo in (4.0, 5.0, 6.0, 8.0, 10.0):
    for gt_ in (0.6, 1.0, 1.5):
        tc = trusted_returns(zc, torch.sin(elc), s_halo, gt_)
        tf = trusted_returns(zf, torch.sin(elf), s_halo, gt_)
        ret_c = tc.float().mean().item()
        cap = (~tf & cls_a).float().sum().item() / max(cls_a.sum().item(), 1)
        row = (s_halo, gt_, ret_c, cap, (~tf).float().mean().item())
        print(f'{s_halo:7.1f} {gt_:6.1f} | {ret_c:15.4f} | {cap:15.4f} | {row[4]:18.4f}')
        if ret_c >= 0.98 and (best is None or cap > best[3]):
            best = row
print('\nCHOSEN (max class-a capture with clear retention >= 98%):', best)
