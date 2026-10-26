#!/usr/bin/env python3
"""T-P3-4: checkpoint surgery for the fog-ckpt init (V1/V3). Takes the bench NeuRAD fog
checkpoint, injects the calibrated rgb head + medium/mixture params at their P3 inits, and
DELETES optimizer/scheduler states (trainer tolerates absence) so ns-train strict-load works
on the flag-extended model with --load-checkpoint.

  python p3_seed_ckpt.py --ckpt <step-29999.ckpt> --head rgb_head_NFF-15.pt --out seeded.ckpt
                         [--beta 0.0321] [--airlight 0.544] [--eta 0.0154] [--theta 1e-6]
"""
import argparse, math
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', type=Path, required=True)
ap.add_argument('--head', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--beta', type=float, default=0.0321)
ap.add_argument('--airlight', type=float, default=0.544)
ap.add_argument('--eta', type=float, default=0.0154)
ap.add_argument('--theta', type=float, default=1e-6)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location='cpu')
sd = ck['pipeline']
pfx = None
for k in sd:
    if '_model.' in k:
        pfx = k.split('_model.')[0] + '_model.'
        break
assert pfx, f'no _model. prefix found; sample keys: {list(sd)[:3]}'
print(f'[seed] prefix {pfx!r}, {len(sd)} keys, step {ck.get("step")}')

head = torch.load(a.head, map_location='cpu')
for k, v in head.items():
    sd[f'{pfx}sample_rgb_head.{k}'] = v
print(f'[seed] injected head: {list(head)}')

def inv_softplus(y):
    return y + math.log(-math.expm1(-y))

sd[f'{pfx}medium_fog.raw_beta0'] = torch.tensor(inv_softplus(a.beta))
# MediumIntegrator.field IS medium_fog (shared module) -> state_dict holds the key twice
sd[f'{pfx}medium_integrator.field.raw_beta0'] = torch.tensor(inv_softplus(a.beta))
sd[f'{pfx}medium_airlight_raw'] = torch.logit(torch.tensor([a.airlight] * 3).clamp(1e-3, 1 - 1e-3))
sd[f'{pfx}medium_gain_raw'] = torch.tensor(0.0)
sd[f'{pfx}fog_mixture.log_eta'] = torch.tensor(math.log(a.eta))
sd[f'{pfx}fog_mixture.log_theta'] = torch.tensor(math.log(a.theta))
for key in ('optimizers', 'schedulers'):
    if key in ck:
        del ck[key]
        print(f'[seed] deleted {key}')
ck['step'] = 0
a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, a.out)
print(f'[seed] wrote {a.out} (beta={a.beta}, A={a.airlight}, eta={a.eta}, theta={a.theta})')
