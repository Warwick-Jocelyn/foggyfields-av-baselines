#!/usr/bin/env python3
"""T10 (P2): controlled beta recovery on a synthetic-fog clone of NCF-02.

Frozen CLEAR NeuRAD checkpoint + calibrated per-sample RGB head; the datamanager reads the
FOGGED clip (data_fogsim). Free params: beta_0, airlight A, lidar gain g. Camera transport =
the in-loop formulation (T*c compositing + ordered airlight) computed on hook-cached per-sample
tensors with the SAME foggyfields functions the model uses; LiDAR = FogReturnMixture NLL on
observed fogged returns + drop NLL on fired non-returns (true labels from the regenerated
range images). Gate (plan §6 P2): fitted beta within 20% of beta_true at moderate fog.

Outputs: results.json (fits per mode, beta errors, tau MAE, held-out RGB), qual PNGs
(fogged GT | clear-scene render | fitted-medium render).
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/capstor/scratch/cscs/yitiwang/foggyfields/medium')
from foggyfields.medium import HomogeneousFog, MediumIntegrator  # noqa: E402
from foggyfields.transport import ordered_airlight  # noqa: E402
from foggyfields.mixture import FogReturnMixture, MixtureConfig  # noqa: E402

torch.manual_seed(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--head', type=Path, required=True)
    ap.add_argument('--fog-root', type=Path, required=True)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--beta-true', type=float, required=True)  # -1 for the ramp variant
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--beta-init', type=float, default=0.01)
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--n-frames', type=int, default=12)
    ap.add_argument('--n-sweeps', type=int, default=15)
    ap.add_argument('--temporal-knots', type=int, default=0,
                    help='>0: fit beta(t) piecewise-linear over clip time instead of a scalar (T11)')
    ap.add_argument('--eta-fixed', type=float, default=None,
                    help="pin the backscatter-to-extinction ratio (T10c: Hahner's 0.046/ln20=0.01536)")
    ap.add_argument('--intensity-rt-factor', type=float, default=2.0,
                    help='exponent factor in the intensity attenuation e^{-F*beta*r}: 2 = physical '
                         'round trip (Hahner); 1 = LISA\'s vendored one-way convention')
    ap.add_argument('--intensity-weight', type=float, default=0.0,
                    help='T10c: L1 weight on g*a_S*T^2 vs observed fogged intensity (returns only)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup

    def upd(cfg):
        dp = cfg.pipeline.datamanager.dataparser
        dp.data = a.fog_root
        dp.sequence = a.variant
        if hasattr(dp, 'use_range_image_raydrop'):
            dp.use_range_image_raydrop = True
        m = cfg.pipeline.model
        m.rgb_head = 'sample_linear'
        from nerfstudio.models.neurad import MediumConfig
        if getattr(m, 'medium', None) is None:   # configs written before the medium existed
            m.medium = MediumConfig()
        m.medium.enabled = True   # creates the modules; fitting runs on cached tensors
        return cfg

    config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val',
                                                   update_config_callback=upd, strict_load=False)
    model, dm = pipeline.model, pipeline.datamanager
    model.eval(); dev = model.device
    model.sample_rgb_head.load_state_dict(torch.load(a.head, map_location=dev))
    model.config.medium.enabled = False          # cache CLEAR-scene renders (no transport)
    model.config.medium.attenuate_lidar = False
    step3 = model.config.rgb_upsample_factor
    print(f'[p2] clear ckpt {ckpt_path} step {step}; fogged clip {a.variant}', flush=True)

    # ---------- camera cache: per-sample c,w,s_mid + per-ray o,d + fogged GT ----------
    tds = dm.train_dataset
    names = [str(f) for f in tds.image_filenames]
    cams = sorted({n.split('/')[-2] for n in names})
    per = max(a.n_frames // len(cams), 1)
    sel = []
    for cn in cams:
        ci = [i for i, n in enumerate(names) if f'/{cn}/' in n]
        sel += [ci[j] for j in np.linspace(0, len(ci) - 1, per).astype(int)]  # cover clip END too
    CC, WW, SS, OO, DD, GG, TT = [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in sel:
            fs, ws, sm, oo, dd = [], [], [], [], []
            def fh(mod, inp, out, fs=fs):
                fs.append(out.detach().half().cpu())          # c = head output per sample
            def wh(mod, args, kwargs, out, ws=ws):
                ws.append(kwargs['weights'].detach().half().cpu())
            def fieldh(mod, inp, out, sm=sm, oo=oo, dd=dd):
                fr = inp[0].frustums
                sm.append(((fr.starts + fr.ends) * 0.5)[..., 0].detach().half().cpu())
                oo.append(fr.origins[..., 0, :].detach().cpu())
                dd.append(fr.directions[..., 0, :].detach().cpu())
            h = [model.sample_rgb_head.register_forward_hook(fh),
                 model.renderer_feat.register_forward_hook(wh, with_kwargs=True),
                 model.field.register_forward_hook(fieldh)]
            _ = model.get_outputs_for_camera(tds.cameras[i: i + 1].to(dev))
            for x in h:
                x.remove()
            c = torch.cat(fs); w = torch.cat(ws)[..., 0]; s = torch.cat(sm)
            o = torch.cat(oo); d = torch.cat(dd)
            gt = tds.get_image_float32(i)[..., :3].numpy()[step3 // 2::step3, step3 // 2::step3]
            gt = torch.from_numpy(gt).reshape(-1, 3)
            assert len(gt) == len(c), (len(gt), len(c))
            keep = torch.from_numpy(rng.choice(len(c), size=min(30000, len(c)), replace=False))
            CC.append(c[keep]); WW.append(w[keep]); SS.append(s[keep])
            OO.append(o[keep]); DD.append(d[keep]); GG.append(gt[keep])
            fi = int(Path(names[i]).stem.split('_')[-1])   # frame_XXXX -> clip time
            TT.append(torch.full((len(keep),), fi / 198.0))
    C = torch.cat(CC).to(dev).float(); Wt = torch.cat(WW).to(dev).float()
    S = torch.cat(SS).to(dev).float().clamp(max=model.config.medium.max_range)
    O = torch.cat(OO).to(dev); Dir = torch.cat(DD).to(dev)
    G = torch.cat(GG).to(dev)
    Tc = torch.cat(TT).to(dev)  # per-ray clip time in [0, 1]
    assert S.shape[1] == Wt.shape[1], (S.shape, Wt.shape)  # sky column last in both
    print(f'[p2] camera cache {C.shape} over {len(sel)} frames', flush=True)

    # ---------- lidar cache ----------
    lds = dm.train_lidar_dataset
    sweep_sel = list(np.linspace(0, len(lds) - 1, a.n_sweeps).astype(int))
    LD, LA, LZ, LR, LO, LDir, LI, LT, sweep_ids = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i in sweep_sel:
            i = int(i)  # np.int64 fails ad_model's isinstance(..., int) assert
            batch = dict(lds[i]); batch['lidar_idx'] = i
            batch['lidar'] = batch['lidar'].to(dev)
            lidar = lds.lidars[i: i + 1].to(dev)
            out, batch = model.get_outputs_for_lidar(lidar, batch)
            pts = batch['lidar']
            li = torch.zeros_like(pts[:, 0:1]).long()
            rb = lidar.generate_rays(lidar_indices=li, points=pts, keep_shape=True)
            ret = batch['did_return'].flatten()
            D = out['depth'].flatten(); A_s = out['intensity'].flatten()
            z = batch['distance'].flatten()
            o = rb.origins.reshape(-1, 3); dr = rb.directions.reshape(-1, 3)
            kr = torch.randperm(int(ret.sum()), device=dev)[:40000]
            kn = torch.randperm(int((~ret).sum()), device=dev)[:15000]
            idx_r = torch.nonzero(ret)[:, 0][kr]; idx_n = torch.nonzero(~ret)[:, 0][kn]
            idx = torch.cat([idx_r, idx_n])
            LD.append(D[idx].cpu()); LA.append(A_s[idx].cpu()); LZ.append(z[idx].cpu())
            LR.append(ret[idx].cpu()); LO.append(o[idx].cpu()); LDir.append(dr[idx].cpu())
            LI.append(pts[idx, 3].cpu())                       # observed (fogged) intensity
            LT.append(torch.full((len(idx),), i / max(len(lds) - 1, 1)))
            sweep_ids.append(i)
            if len(sweep_ids) >= a.n_sweeps:
                break
    LD = torch.cat(LD).to(dev); LA = torch.cat(LA).to(dev); LZ = torch.cat(LZ).to(dev)
    LR = torch.cat(LR).to(dev); LO = torch.cat(LO).to(dev); LDir = torch.cat(LDir).to(dev)
    LI = torch.cat(LI).to(dev); LT = torch.cat(LT).to(dev)
    LR = LR & (LZ <= model.config.medium.max_range)  # observed ranges beyond the LUT: drop from returns
    LD_c = LD.clamp(max=model.config.medium.max_range)
    # drop_nll only on BELOW-HORIZON non-returns (sky rays never return in clear air either;
    # world z is up and the lidar is roughly level, so dir_z < 0 ~ not sky)
    LNR = (~LR) & (LDir[:, 2] < -0.01)
    print(f'[p2] lidar cache {LD.shape} rays ({int(LR.sum())} returns) over {len(sweep_ids)} sweeps', flush=True)

    # ---------- fits ----------
    edges = MediumIntegrator(HomogeneousFog(0.01), max_range=model.config.medium.max_range).edges.to(dev)

    class BetaKnots(torch.nn.Module):
        """T11: beta(t) piecewise-linear over clip time (softplus-positive knots)."""
        def __init__(self, k, init):
            super().__init__()
            self.raw = torch.nn.Parameter(HomogeneousFog._inv_softplus(
                torch.full((k,), float(max(init, 1e-6)))).to(dev))
            self.k = k

        def at(self, t):  # t [R] in [0,1] -> beta [R]
            x = t.clamp(0, 1) * (self.k - 1)
            i0 = x.floor().long().clamp(0, self.k - 2)
            w = x - i0.float()
            b = torch.nn.functional.softplus(self.raw)
            return b[i0] * (1 - w) + b[i0 + 1] * w

    from foggyfields.medium import MediumLUT as _ML

    def make_lut(fog_or_knots, Opts, Dpts, tvec):
        if isinstance(fog_or_knots, BetaKnots):
            return _ML(edges, fog_or_knots.at(tvec)[:, None] * edges[None, :],
                       beta_mid=fog_or_knots.at(tvec)[:, None].expand(-1, len(edges) - 1))
        return MediumIntegrator(fog_or_knots, max_range=model.config.medium.max_range).to(dev)(Opts, Dpts)

    def fit(use_cam, use_lid, tag, iters=None, k_lidar=1.0):
        eta0 = a.eta_fixed if a.eta_fixed is not None else 0.03
        mixcfg = MixtureConfig(k_lidar=k_lidar, eta=eta0, learn_eta=a.eta_fixed is None,
                               s0_near=2.0, theta=1e-6, learn_theta=True, min_range=2.5)
        if a.temporal_knots > 0:
            fog = BetaKnots(a.temporal_knots, a.beta_init)
            fog_params = [fog.raw]
        else:
            fog = HomogeneousFog(a.beta_init).to(dev)
            fog_params = [fog.raw_beta0]
        A_raw = torch.nn.Parameter(torch.logit(torch.tensor([0.5] * 3, device=dev)))
        # gain FIXED at 1 (T10b): a_S is the clear checkpoint's calibrated intensity.
        g_raw = torch.zeros((), device=dev)
        mix = FogReturnMixture(mixcfg).to(dev)
        opt = torch.optim.Adam([*fog_params, A_raw, *mix.parameters()], lr=0.03)
        hist = []
        for it in range(iters or a.iters):
            opt.zero_grad()
            loss = torch.zeros((), device=dev)
            if use_cam:
                lut = make_lut(fog, O, Dir, Tc)
                T = lut.transmittance(S)
                A = torch.sigmoid(A_raw)
                rgb = (Wt[..., None] * T[..., None] * C).sum(-2) \
                    + ordered_airlight(lut, A, scene_w=Wt, scene_s=S)
                loss = loss + torch.mean((rgb - G) ** 2) * 20.0
            if use_lid:
                lut_l = make_lut(fog, LO, LDir, LT)
                g = torch.exp(g_raw)
                E_S, E_F, s_mid = mix.echo_strengths(g * LA, LD_c, lut_l)
                pi = mix.selection_probs(E_S, E_F)
                nll = mix.nll(LZ[LR], LD_c[LR], pi[LR], s_mid).mean()
                dnll = mix.drop_nll(pi[LNR]).mean() if LNR.any() else torch.zeros((), device=dev)
                loss = loss + nll + 0.5 * dnll
                if a.intensity_weight > 0:
                    # T10d: LOG-ratio residual on confident SURFACE returns only (z ~= clear depth,
                    # bright enough to be above the uint8 floor). A direct numpy regression of this
                    # quantity recovers beta to 2% on the synthetic data; linear-space L1 over all
                    # returns (T10c) buried that signal under scattered halo points.
                    m_int = LR & ((LZ - LD).abs() < 1.0) & (LA > 0.02) & (LI > 1.5 / 255)
                    if m_int.any():
                        tau_z = lut_l.tau_at(LZ.clamp(max=120.0))
                        resid = torch.log(LI[m_int] / (g * LA[m_int]).clamp(min=1e-6)) \
                            + a.intensity_rt_factor * k_lidar * tau_z[m_int]
                        loss = loss + a.intensity_weight * (resid ** 2).mean()
            loss.backward(); opt.step()
            if it % 150 == 0 or it == (iters or a.iters) - 1:
                bval = float(torch.nn.functional.softplus(fog.raw).mean()) if a.temporal_knots > 0 \
                    else float(fog.beta0)
                hist.append(dict(it=it, loss=float(loss), beta=bval,
                                 A=float(torch.sigmoid(A_raw).mean()), g=float(torch.exp(g_raw)),
                                 eta=float(mix.eta), theta=float(mix.theta)))
                print(f'[p2:{tag}] {hist[-1]}', flush=True)
        out = dict(tag=tag, A=[float(x) for x in torch.sigmoid(A_raw)],
                   g=float(torch.exp(g_raw)), eta=float(mix.eta), theta=float(mix.theta), hist=hist)
        if a.temporal_knots > 0:
            out['beta_knots'] = [float(x) for x in torch.nn.functional.softplus(fog.raw)]
            out['beta'] = float(np.mean(out['beta_knots']))
        else:
            out['beta'] = float(fog.beta0)
        return out

    # code-correctness control: GT constructed with the MODEL's own formation at beta_true
    # -> camera fit must recover beta_true almost exactly (isolates formation mismatch of the
    # stored Koschmieder JPEGs from fit-code errors)
    G_real = G
    ctrl = None
    if a.beta_true > 0:
        with torch.no_grad():
            fog_t = HomogeneousFog(a.beta_true, learnable=False).to(dev)
            integ_t = MediumIntegrator(fog_t, max_range=model.config.medium.max_range).to(dev)
            lut_t = integ_t(O, Dir)
            T_t = lut_t.transmittance(S)
            A_t = torch.full((3,), 0.75, device=dev)
            G = ((Wt[..., None] * T_t[..., None] * C).sum(-2)
                 + ordered_airlight(lut_t, A_t, scene_w=Wt, scene_s=S)).clamp(0, 1)
        ctrl = fit(True, False, 'control_selfconsistent', iters=500)
        G = G_real
    t0 = time.time()
    fits = {t: fit(c_, l_, t) for t, c_, l_ in [('joint', True, True),
                                                ('camera_only', True, False),
                                                ('lidar_only', False, True)]}
    if ctrl is not None:
        fits['control_selfconsistent'] = ctrl
    # sensitivity (lidar-only, +-30% k_lidar)
    sens = {}
    for kl in (0.7, 1.3):
        sens[f'k_lidar={kl}'] = fit(False, True, f'sens_k{kl}', iters=400, k_lidar=kl)['beta']
    fit_s = time.time() - t0

    # ---------- held-out RGB + quals with the fitted medium ----------
    eds = dm.eval_dataset
    enames = [str(f) for f in eds.image_filenames]
    eidx = [i for i, n in enumerate(enames) if 'front_wide' in n][::8][:8]
    bj = fits['joint']
    model.medium_fog = HomogeneousFog(max(bj['beta'], 1e-6), learnable=False).to(dev)
    model.medium_integrator.field = model.medium_fog
    with torch.no_grad():
        model.medium_airlight_raw.copy_(torch.logit(torch.tensor(bj['A'], device=dev).clamp(1e-3, 1 - 1e-3)))
    import torchmetrics
    lp = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(dev)
    ssim_f = torchmetrics.functional.structural_similarity_index_measure
    rows = []
    with torch.no_grad():
        for j, i in enumerate(eidx):
            model.config.medium.enabled = False
            off = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
            model.config.medium.enabled = True
            on = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
            gt = torch.from_numpy(eds.get_image_float32(i)[..., :3].numpy()
                                  [step3 // 2::step3, step3 // 2::step3]).to(dev)
            r = {}
            for tag, im in [('clear_scene', off['rgb_ray']), ('fitted_medium', on['rgb_ray'])]:
                x = im.permute(2, 0, 1)[None].clamp(0, 1); y = gt.permute(2, 0, 1)[None].clamp(0, 1)
                r[tag] = dict(psnr=-10 * math.log10(float(torch.mean((im - gt) ** 2)) + 1e-12),
                              ssim=float(ssim_f(x, y, data_range=1.0)), lpips=float(lp(x, y)))
            rows.append(r)
            if j < 3:
                trio = torch.cat([gt, off['rgb_ray'].clamp(0, 1), on['rgb_ray'].clamp(0, 1)], 1)
                Image.fromarray((trio.cpu().numpy() * 255).astype(np.uint8)).save(
                    a.out / f'qual_{j}_foggt_clearscene_medium.png')
    agg = {t: {k: float(np.mean([r[t][k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')} for t in rows[0]}

    bt = a.beta_true
    ramp_eval = None
    if a.temporal_knots > 0:
        RAMP = (0.005, 0.06)  # must match make_fog_clip.py
        tk = np.linspace(0, 1, a.temporal_knots)
        true_k = RAMP[0] + (RAMP[1] - RAMP[0]) * tk
        ramp_eval = {}
        for mname, f in fits.items():
            if 'beta_knots' in f:
                bk = np.array(f['beta_knots'])
                rel = np.abs(bk - true_k) / true_k
                ramp_eval[mname] = dict(knots_fit=[round(float(x), 4) for x in bk],
                                        knots_true=[round(float(x), 4) for x in true_k],
                                        rel_err=[round(float(x), 3) for x in rel],
                                        max_rel_err_t_ge_03=round(float(rel[tk >= 0.3].max()), 3),
                                        gate_20pct_t_ge_03=bool(rel[tk >= 0.3].max() <= 0.2))
    res = dict(variant=a.variant, beta_true=bt, ckpt=str(ckpt_path), ramp_eval=ramp_eval,
               fits={k: {kk: v[kk] for kk in ('beta', 'A', 'g')} for k, v in fits.items()},
               fit_history={k: v['hist'] for k, v in fits.items()},
               beta_rel_err={k: (abs(v['beta'] - bt) / bt if bt > 0 else None) for k, v in fits.items()},
               gate_20pct={k: (abs(v['beta'] - bt) / bt <= 0.2 if bt > 0 else None) for k, v in fits.items()},
               tau_mae_60m={k: (abs(v['beta'] - bt) * 30.0 if bt > 0 else None) for k, v in fits.items()},
               sensitivity_beta=sens, airlight_true=0.75, fit_seconds=fit_s,
               gate_note='camera_only is the P2 gate number; lidar_only/joint are diagnostic until '
                         'eta/theta calibration is validated (validate-agent 2026-10-26); '
                         'control_selfconsistent isolates fit-code errors from GT formation mismatch',
               rgb_heldout=dict(n=len(rows), **agg))
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'fit_history'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
