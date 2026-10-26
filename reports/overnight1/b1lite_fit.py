#!/usr/bin/env python3
"""B1-lite (overnight1 Task 4): homogeneous fog + airlight + gain on a FROZEN NeuRAD fog
checkpoint. Fits only (beta_0, A, g). Camera transport = expected-depth overlay (P1 prototype,
labelled; audit Q3: no per-sample RGB exists). LiDAR = surface-echo attenuation E_S * T^2 only.

Structural note reported with the results: with the scene frozen, predicted LiDAR RANGE is the
backbone's rendered depth unchanged — attenuation cannot move the >5 m range tail. The
measurements here are: held-out RGB (overlay vs backbone), held-out intensity (attenuated vs
backbone), fitted beta vs the A4 anchor, and cross-sensor agreement (camera-only vs lidar-only
fits). For homogeneous fog tau(s) = beta*s is used directly; MediumIntegrator reproduces it
exactly (tests/test_medium.py::test_analytic_homogeneous_tau).
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/capstor/scratch/cscs/yitiwang/foggyfields/medium')
from foggyfields.medium import HomogeneousFog  # noqa: E402


def psnr(a, b):
    return -10 * math.log10(torch.mean((a - b) ** 2).item() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--beta-init', type=float, required=True)
    ap.add_argument('--a-init', type=float, required=True)
    ap.add_argument('--beta-anchor', type=float, required=True)
    ap.add_argument('--n-train-cams', type=int, default=21)
    ap.add_argument('--n-train-sweeps', type=int, default=20)
    ap.add_argument('--n-eval-cams', type=int, default=33)
    ap.add_argument('--iters', type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from nerfstudio.utils.eval_utils import eval_setup
    config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val')
    model, dm = pipeline.model, pipeline.datamanager
    model.eval()
    dev = model.device
    print(f'[b1] loaded {ckpt_path} step {step}', flush=True)

    # ---------- cache TRAIN camera renders ----------
    tds = dm.train_dataset
    names = [str(f) for f in tds.image_filenames]
    cams = ['camera_front_wide_120fov', 'camera_cross_left_120fov', 'camera_cross_right_120fov']
    per = max(a.n_train_cams // len(cams), 1)
    idxs = []
    for cn in cams:
        ci = [i for i, n in enumerate(names) if cn in n]
        idxs += ci[:: max(len(ci) // per, 1)][:per]
    from PIL import Image
    cam_cache = []
    with torch.no_grad():
        for i in idxs:
            out = model.get_outputs_for_camera(tds.cameras[i: i + 1].to(dev))
            dpt = out['depth'][..., 0]
            acc = out['accumulation'][..., 0]
            H, W = dpt.shape
            rgb = out['rgb'].permute(2, 0, 1)[None]
            rgb = torch.nn.functional.interpolate(rgb, size=(H, W), mode='area')[0].permute(1, 2, 0)
            gt = torch.from_numpy(np.asarray(Image.open(names[i]).resize((W, H), Image.BILINEAR),
                                             dtype=np.float32) / 255.0).to(dev)
            cam_cache.append((rgb.float(), dpt.float(), acc.float(), gt))
    print(f'[b1] cached {len(cam_cache)} train camera frames', flush=True)

    # ---------- zero-fog bit-for-bit check on a real frame (GPU) ----------
    rgb0, dpt0, acc0, _ = cam_cache[0]
    T0 = torch.exp(-0.0 * dpt0)[..., None]
    A0 = torch.full((3,), a.a_init, device=dev)
    ov0 = T0 * rgb0 + (1 - T0) * A0
    assert torch.equal(ov0, rgb0 * 1.0) or (ov0 - rgb0).abs().max() == 0, 'zero-fog reduction violated'
    print('[b1] zero-fog reduction on real frame: EXACT', flush=True)

    # ---------- cache TRAIN lidar renders ----------
    lds = dm.train_lidar_dataset
    n_sw = len(lds)
    step_sw = max(n_sw // a.n_train_sweeps, 1)
    lid_cache = []
    with torch.no_grad():
        for i in range(0, n_sw, step_sw):
            batch = dict(lds[i])
            batch['lidar_idx'] = i
            pts = batch['lidar'].to(dev)
            batch['lidar'] = pts
            out, batch = model.get_outputs_for_lidar(lds.lidars[i: i + 1].to(dev), batch)
            ret = batch['did_return'].flatten()
            D = out['depth'].flatten()[ret]
            I_bb = out['intensity'].flatten()[ret]
            I_gt = pts[:, 3].flatten()[ret]
            keep = torch.randperm(len(D), device=dev)[:80000]
            lid_cache.append((D[keep].float(), I_bb[keep].float(), I_gt[keep].float()))
            if len(lid_cache) >= a.n_train_sweeps:
                break
    print(f'[b1] cached {len(lid_cache)} train sweeps', flush=True)
    Dl = torch.cat([x[0] for x in lid_cache]); Ib = torch.cat([x[1] for x in lid_cache])
    Ig = torch.cat([x[2] for x in lid_cache])

    # ---------- fit variants ----------
    def fit(use_cam, use_lid, tag):
        fog = HomogeneousFog(a.beta_init).to(dev)
        A_raw = torch.nn.Parameter(torch.logit(torch.tensor([a.a_init] * 3, device=dev)))
        g_raw = torch.nn.Parameter(torch.zeros((), device=dev))
        opt = torch.optim.Adam([fog.raw_beta0, A_raw, g_raw], lr=0.02)
        hist = []
        for it in range(a.iters):
            opt.zero_grad()
            beta = fog.beta0
            A = torch.sigmoid(A_raw)
            loss = torch.zeros((), device=dev)
            lc = ll = float('nan')
            if use_cam:
                lcam = torch.zeros((), device=dev)
                for rgb, dpt, acc, gt in cam_cache:
                    T = torch.exp(-beta * dpt)[..., None]
                    ov = T * rgb + (1 - T) * A[None, None, :]
                    T_far = torch.exp(-beta * 120.0)
                    sky = T_far * rgb + (1 - T_far) * A[None, None, :]
                    pred = acc[..., None] * ov + (1 - acc[..., None]) * sky
                    lcam = lcam + torch.mean((pred - gt) ** 2)
                lcam = lcam / len(cam_cache)
                loss = loss + lcam; lc = lcam.item()
            if use_lid:
                g = torch.nn.functional.softplus(g_raw) + 0.5
                I_hat = g * Ib * torch.exp(-2 * beta * Dl)
                llid = torch.mean(torch.abs(I_hat - Ig))
                loss = loss + llid; ll = llid.item()
            loss.backward(); opt.step()
            if it % 100 == 0 or it == a.iters - 1:
                hist.append(dict(it=it, loss=loss.item(), cam=lc, lid=ll,
                                 beta=fog.beta0.item(), A=torch.sigmoid(A_raw).mean().item(),
                                 g=(torch.nn.functional.softplus(g_raw) + 0.5).item()))
                print(f'[b1:{tag}] {hist[-1]}', flush=True)
        return dict(tag=tag, beta=fog.beta0.item(), A=[float(x) for x in torch.sigmoid(A_raw)],
                    g=(torch.nn.functional.softplus(g_raw) + 0.5).item(), hist=hist)

    t0 = time.time()
    fits = {v['tag']: v for v in [fit(True, True, 'joint'), fit(True, False, 'camera_only'),
                                  fit(False, True, 'lidar_only')]}
    fit_s = time.time() - t0

    # ---------- held-out RGB eval ----------
    eds = dm.eval_dataset
    enames = [str(f) for f in eds.image_filenames]
    eidx = [i for i, n in enumerate(enames) if 'front_wide' in n]
    eidx = eidx[:: max(len(eidx) // a.n_eval_cams, 1)][: a.n_eval_cams]
    import torchmetrics
    lp = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(dev)
    ssim_f = torchmetrics.functional.structural_similarity_index_measure
    bj, Aj = fits['joint']['beta'], torch.tensor(fits['joint']['A'], device=dev)
    rows = []
    with torch.no_grad():
        for i in eidx:
            out = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
            dpt = out['depth'][..., 0]; acc = out['accumulation'][..., 0]
            H, W = dpt.shape
            rgb = torch.nn.functional.interpolate(out['rgb'].permute(2, 0, 1)[None], size=(H, W),
                                                  mode='area')[0].permute(1, 2, 0)
            gt = torch.from_numpy(np.asarray(Image.open(enames[i]).resize((W, H), Image.BILINEAR),
                                             dtype=np.float32) / 255.0).to(dev)
            T = torch.exp(-bj * dpt)[..., None]
            ov = T * rgb + (1 - T) * Aj[None, None, :]
            T_far = math.exp(-bj * 120.0)
            pred = acc[..., None] * ov + (1 - acc[..., None]) * (T_far * rgb + (1 - T_far) * Aj[None, None, :])
            hb, wb = int(H * .08), int(W * .08)
            sl = np.s_[hb:H - hb, wb:W - wb]
            r = {}
            for tag, img in [('backbone', rgb[sl]), ('b1lite', pred[sl])]:
                g_ = gt[sl]
                x = img.permute(2, 0, 1)[None].clamp(0, 1); y = g_.permute(2, 0, 1)[None].clamp(0, 1)
                r[tag] = dict(psnr=psnr(img, g_), ssim=float(ssim_f(x, y, data_range=1.0)),
                              lpips=float(lp(x, y)))
            rows.append(r)
    agg = {t: {k: float(np.mean([r[t][k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
           for t in ('backbone', 'b1lite')}

    # qualitative frames
    from PIL import Image as PImage
    for j, i in enumerate(eidx[:3]):
        with torch.no_grad():
            out = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
            dpt = out['depth'][..., 0]; acc = out['accumulation'][..., 0]
            H, W = dpt.shape
            rgb = torch.nn.functional.interpolate(out['rgb'].permute(2, 0, 1)[None], size=(H, W),
                                                  mode='area')[0].permute(1, 2, 0)
            gt = torch.from_numpy(np.asarray(PImage.open(enames[i]).resize((W, H)),
                                             dtype=np.float32) / 255.0).to(dev)
            T = torch.exp(-bj * dpt)[..., None]
            ov = acc[..., None] * (T * rgb + (1 - T) * Aj[None, None, :]) + \
                 (1 - acc[..., None]) * (math.exp(-bj * 120.0) * rgb + (1 - math.exp(-bj * 120.0)) * Aj[None, None, :])
            trio = torch.cat([gt, rgb, ov.clamp(0, 1)], dim=1).cpu().numpy()
            PImage.fromarray((trio * 255).astype(np.uint8)).save(a.out / f'qual_{j}_gt_backbone_b1lite.png')

    # ---------- held-out LiDAR intensity from the existing dump ----------
    import glob as _g
    dump = sorted(_g.glob(f'/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/outputs_fix/_lidar_dump/neurad_safe__{a.clip}/frame_*.npz'))
    bl, gl = fits['joint']['beta'], fits['joint']['g']
    se_b = se_a = cnt = 0
    for f in dump:
        d = np.load(f)
        ret = d['gt_did_return'] & (d['gt_range'] < 1000)
        Ibb = d['pred_intensity'][ret]; Igt = d['gt_intensity'][ret]; Dd = d['pred_range'][ret]
        se_b += float(((Ibb - Igt) ** 2).sum()); cnt += ret.sum()
        se_a += float(((gl * Ibb * np.exp(-2 * bl * Dd) - Igt) ** 2).sum())
    lid_eval = dict(n_rays=int(cnt), intensity_rmse_backbone=math.sqrt(se_b / cnt),
                    intensity_rmse_b1lite=math.sqrt(se_a / cnt),
                    note='range metrics are structurally UNCHANGED under attenuation-only with a frozen scene')

    res = dict(clip=a.clip, config=str(a.config), ckpt=str(ckpt_path), step=int(step),
               variant='B1-lite (frozen scene, expected-depth overlay [P1 prototype], surface-echo attenuation only)',
               beta_anchor_a4=a.beta_anchor, beta_init=a.beta_init, A_init=a.a_init,
               fits={k: {kk: v[kk] for kk in ('beta', 'A', 'g')} for k, v in fits.items()},
               fit_history={k: v['hist'] for k, v in fits.items()}, fit_seconds=fit_s,
               rgb_eval=dict(n_frames=len(rows), frames='held-out front_wide subset, central 84% crop, render res', **agg),
               lidar_eval=lid_eval,
               separation=dict(beta_within_x2_of_anchor=bool(a.beta_anchor / 2 <= fits['joint']['beta'] <= a.beta_anchor * 2),
                               camera_only_beta=fits['camera_only']['beta'], lidar_only_beta=fits['lidar_only']['beta'],
                               medium_off_render='== backbone by construction (scene frozen); fog remains in the '
                                                 'frozen geometry — separation requires scene unfreezing (P3)'))
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'fit_history'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
