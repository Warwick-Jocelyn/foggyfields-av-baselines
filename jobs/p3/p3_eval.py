#!/usr/bin/env python3
"""T-P3-7: P3 gate evaluation for one trained variant (NFF-15, held-out split).

Produces per-variant JSON with:
  G2  RGB at ray resolution: rgb_ray vs center-pixel GT (PSNR/SSIM/LPIPS), medium ON;
      plus the same with the medium OFF (defogged render quality is NOT a gate - counterfactual)
  G1/G5  LiDAR: per-range-bin >1 m / >5 m tail fractions + median depth err + intensity RMSE,
      from a fresh lidar dump of this model (reuses the ray adapter conventions)
  G3  fitted beta0 / A / eta / theta read from the checkpoint
  G4  fog-in-geometry diagnostics with the MEDIUM OFF:
      (i) false-near mass: frac of rays with pred depth < 10 m where trusted GT > 20 m
      (ii) dark-channel residual beta of the medium-off renders (Koschmieder fit vs own depth)
Usage: p3_eval.py --config <run config.yml> --out DIR [--n-frames 24] [--n-sweeps 25]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/capstor/scratch/cscs/yitiwang/foggyfields/medium')
torch.manual_seed(0)

BINS = [(0, 10), (10, 30), (30, 60), (60, 120)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-frames', type=int, default=24)
    ap.add_argument('--n-sweeps', type=int, default=25)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup
    config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val')
    model, dm = pipeline.model, pipeline.datamanager
    model.eval(); dev = model.device
    mc = model.config.medium
    med_on = bool(getattr(mc, 'enabled', False)) and getattr(model, 'medium_fog', None) is not None
    fitted = {}
    if med_on:
        fitted = dict(beta0=float(model.medium_fog.beta0),
                      airlight=[float(x) for x in torch.sigmoid(model.medium_airlight_raw)],
                      gain=float(torch.exp(model.medium_gain_raw)))
        if getattr(model, 'fog_mixture', None) is not None:
            fitted.update(eta=float(model.fog_mixture.eta), theta=float(model.fog_mixture.theta))
    print(f'[p3eval] {a.config} step {step} medium={med_on} fitted={fitted}', flush=True)

    import torchmetrics
    lp = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(dev)
    ssim_f = torchmetrics.functional.structural_similarity_index_measure
    step3 = model.config.rgb_upsample_factor
    eds = dm.eval_dataset
    enames = [str(f) for f in eds.image_filenames]
    eidx = [i for i, n in enumerate(enames) if 'front_wide' in n]
    eidx = eidx[:: max(len(eidx) // a.n_frames, 1)][: a.n_frames]

    def rgb_pass(enable):
        if med_on:
            model.config.medium.enabled = enable
        rows, dark_betas = [], []
        with torch.no_grad():
            for j, i in enumerate(eidx):
                out = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
                gt = torch.from_numpy(eds.get_image_float32(i)[..., :3].numpy()
                                      [step3 // 2::step3, step3 // 2::step3]).to(dev)
                im = out['rgb_ray'].clamp(0, 1)
                x = im.permute(2, 0, 1)[None]; y = gt.permute(2, 0, 1)[None].clamp(0, 1)
                rows.append(dict(psnr=-10 * math.log10(float(torch.mean((im - gt) ** 2)) + 1e-12),
                                 ssim=float(ssim_f(x, y, data_range=1.0)), lpips=float(lp(x, y))))
                if not enable:
                    # G4(ii): dark-channel residual beta of the medium-off render vs its own depth
                    dn = (out['depth'][..., 0] / out['accumulation'][..., 0].clamp(min=0.3)).clamp(0, 120)
                    dark = im.min(-1).values
                    A = float(dark.flatten().sort().values[-max(dark.numel() // 1000, 50):].mean().clamp(min=0.05))
                    ok = (out['accumulation'][..., 0] > 0.98) & (dn > 8) & (dn < 100) & (dark < 0.95 * A)
                    if ok.sum() > 5000:
                        yv = -torch.log((1 - dark[ok] / A).clamp(min=1e-3))
                        xv = dn[ok]
                        X = torch.stack([torch.ones_like(xv), xv], 1)
                        sol = torch.linalg.lstsq(X, yv[:, None]).solution
                        dark_betas.append(float(sol[1]))
                if j < 3:
                    tag = 'on' if enable else 'off'
                    trio = torch.cat([gt, im], dim=1).cpu().numpy()
                    Image.fromarray((trio * 255).astype(np.uint8)).save(a.out / f'qual_{j}_gt_medium{tag}.png')
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
        return agg, (float(np.median(dark_betas)) if dark_betas else None)

    rgb_on, _ = rgb_pass(True) if med_on else (None, None)
    rgb_off, dark_beta_off = rgb_pass(False)
    if med_on:
        model.config.medium.enabled = True

    # ---------- LiDAR: fresh dump over held-out sweeps ----------
    lds = dm.eval_lidar_dataset
    sel = list(np.linspace(0, len(lds) - 1, min(a.n_sweeps, len(lds))).astype(int))
    errs, gts, false_near = [], [], [0, 0]
    int_se, int_n = 0.0, 0
    from foggyfields.trusted import trusted_returns
    with torch.no_grad():
        for i in sel:
            batch = dict(lds[int(i)]); batch['lidar_idx'] = int(i)
            batch['lidar'] = batch['lidar'].to(dev)
            out, batch = model.get_outputs_for_lidar(lds.lidars[int(i): int(i) + 1].to(dev), batch)
            ret = batch['did_return'].flatten()
            z = batch['distance'].flatten()[ret]
            D = out['depth'].flatten()[ret]
            errs.append((D - z).abs().cpu()); gts.append(z.cpu())
            I_obs = batch['lidar'][:, 3].flatten()[ret]
            I_pred = out['intensity'].flatten()[ret]
            int_se += float(((I_pred - I_obs) ** 2).sum()); int_n += int(ret.sum())
            # G4(i) false-near with medium off is the geometry; depth is medium-independent here
            far_gt = z > 20
            false_near[0] += int(((D < 10) & far_gt).sum()); false_near[1] += int(far_gt.sum())
    err = torch.cat(errs).numpy(); gt = torch.cat(gts).numpy()
    lidar = dict(n_rays=int(len(err)), median_err=float(np.median(err)),
                 frac_gt1m=float((err > 1).mean()), frac_gt5m=float((err > 5).mean()),
                 intensity_rmse=math.sqrt(int_se / max(int_n, 1)),
                 false_near_mass=false_near[0] / max(false_near[1], 1))
    for lo, hi in BINS:
        m = (gt >= lo) & (gt < hi)
        lidar[f'gt1m_{lo}-{hi}'] = float((err[m] > 1).mean()) if m.any() else None
        lidar[f'gt5m_{lo}-{hi}'] = float((err[m] > 5).mean()) if m.any() else None

    res = dict(config=str(a.config), step=int(step), medium=med_on, fitted=fitted,
               rgb_ray_medium_on=rgb_on, rgb_ray_medium_off=rgb_off,
               dark_channel_beta_medium_off=dark_beta_off, lidar=lidar,
               notes='rgb at ray res (center-pixel GT); lidar depth is medium-independent '
                     '(geometry); dark-channel residual beta: clear-method floor is 0.0077')
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
