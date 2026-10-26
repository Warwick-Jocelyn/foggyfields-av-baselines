#!/usr/bin/env python3
"""T2 (plan-agent P1a): calibrate the per-sample linear RGB head on a FROZEN NeuRAD checkpoint.

Everything frozen except the 147-param head. Per-sample features + sky-absorbed weights are
captured with forward hooks during normal eval renders of TRAIN frames (exact consistency with
the model forward), cached (subsampled), and the head is fitted to GT train pixels at ray
resolution (GT[1::3, 1::3], the same grid the compensated eval rays shoot). Then B0' numbers on
held-out frames: head rgb_ray vs GT, next to the CNN rgb (area-downsampled) on the same pixels.

Outputs in --out: rgb_head_<clip>.pt, results.json, manifest.yaml, qual PNGs (GT | B0 CNN | B0' head).
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(0)


def psnr(a, b):
    return -10 * math.log10(torch.mean((a - b) ** 2).item() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--frames-per-cam', type=int, default=2)
    ap.add_argument('--rays-per-frame', type=int, default=40000)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--n-eval-frames', type=int, default=14)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from nerfstudio.utils.eval_utils import eval_setup

    def upd(cfg):
        cfg.pipeline.model.rgb_head = "sample_linear"
        return cfg

    config, pipeline, ckpt_path, step = eval_setup(a.config, test_mode='val',
                                                   update_config_callback=upd, strict_load=False)
    model, dm = pipeline.model, pipeline.datamanager
    model.eval()
    dev = model.device
    assert hasattr(model, 'sample_rgb_head'), 'flag did not take'
    # strict_load=False fallback must only have tolerated the new head's keys
    import torch as _t
    ck = _t.load(ckpt_path, map_location='cpu')['pipeline']
    ck_model_keys = {k.split('module.', 1)[-1].replace('_model.', '', 1) for k in ck if '_model.' in k}
    missing = {k for k, _ in model.named_parameters()} - ck_model_keys
    assert all(k.startswith('sample_rgb_head') for k in missing), f'unexpected missing keys: {sorted(missing)[:5]}'
    print(f'[t2] loaded {ckpt_path} step {step}; head params '
          f'{sum(p.numel() for p in model.sample_rgb_head.parameters())}', flush=True)

    tds = dm.train_dataset
    names = [str(f) for f in tds.image_filenames]
    cams = sorted({n.split('/')[-2] for n in names})
    sel = []
    for cn in cams:
        ci = [i for i, n in enumerate(names) if f'/{cn}/' in n]
        sel += ci[:: max(len(ci) // a.frames_per_cam, 1)][: a.frames_per_cam]
    print(f'[t2] caching {len(sel)} train frames over {len(cams)} cameras', flush=True)

    F_, W_, G_ = [], [], []
    rng = np.random.default_rng(0)
    step3 = model.config.rgb_upsample_factor
    with torch.no_grad():
        for i in sel:
            # accumulate per-chunk hook captures during one full-frame render
            fs, ws = [], []
            def f_hook2(mod, inp, out, fs=fs):
                fs.append(inp[0].detach().half().cpu())
            def w_hook2(mod, args, kwargs, out, ws=ws):
                ws.append(kwargs['weights'].detach().half().cpu())
            hh1 = model.renderer_feat.register_forward_hook(w_hook2, with_kwargs=True)
            hh2 = model.sample_rgb_head.register_forward_hook(f_hook2)
            _ = model.get_outputs_for_camera(tds.cameras[i: i + 1].to(dev))
            hh1.remove(); hh2.remove()
            f = torch.cat(fs); w = torch.cat(ws)[..., 0]
            img = tds.get_image_float32(i)[..., :3].numpy()  # safe-crop applied (validate-agent fix)
            gt = torch.from_numpy(img[step3 // 2::step3, step3 // 2::step3]).reshape(-1, 3)
            assert len(gt) == len(f), (len(gt), len(f))
            keep = torch.from_numpy(rng.choice(len(f), size=min(a.rays_per_frame, len(f)), replace=False))
            F_.append(f[keep]); W_.append(w[keep]); G_.append(gt[keep])
    F = torch.cat(F_).to(dev); W = torch.cat(W_).to(dev); G = torch.cat(G_).to(dev)
    print(f'[t2] cache: {F.shape} feats, {W.shape} weights, {G.shape} gt', flush=True)

    # ---- fit the head on the cache ----
    head = model.sample_rgb_head
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
    n = len(F)
    t0 = time.time()
    for it in range(a.iters):
        idx = torch.randint(0, n, (65536,), device=dev)
        f = F[idx].float(); w = W[idx].float()
        pred = (w[..., None] * head(f)).sum(-2)
        loss = torch.mean((pred - G[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == a.iters - 1:
            print(f'[t2] it {it} mse {loss.item():.5f}', flush=True)
    fit_s = time.time() - t0

    # ---- held-out B0' vs B0 ----
    eds = dm.eval_dataset
    enames = [str(f) for f in eds.image_filenames]
    eidx = list(range(0, len(enames), max(len(enames) // a.n_eval_frames, 1)))[: a.n_eval_frames]
    import torchmetrics
    lp = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(dev)
    ssim_f = torchmetrics.functional.structural_similarity_index_measure
    rows = []
    with torch.no_grad():
        for j, i in enumerate(eidx):
            out = model.get_outputs_for_camera(eds.cameras[i: i + 1].to(dev))
            img = eds.get_image_float32(i)[..., :3].numpy()  # safe-crop applied
            gt = torch.from_numpy(img[step3 // 2::step3, step3 // 2::step3]).to(dev)
            H, W2 = gt.shape[:2]
            cnn = torch.nn.functional.interpolate(out['rgb'].permute(2, 0, 1)[None],
                                                  size=(H, W2), mode='area')[0].permute(1, 2, 0)
            headr = out['rgb_ray'].clamp(0, 1)
            r = {}
            for tag, im in [('b0_cnn_at_ray_res', cnn), ('b0p_head', headr)]:
                x = im.permute(2, 0, 1)[None].clamp(0, 1); y = gt.permute(2, 0, 1)[None].clamp(0, 1)
                r[tag] = dict(psnr=psnr(im, gt), ssim=float(ssim_f(x, y, data_range=1.0)), lpips=float(lp(x, y)))
            rows.append(r)
            if j < 3:
                trio = torch.cat([gt, cnn.clamp(0, 1), headr], 1).cpu().numpy()
                Image.fromarray((trio * 255).astype(np.uint8)).save(a.out / f'qual_{j}_gt_cnn_head.png')
    agg = {t: {k: float(np.mean([r[t][k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
           for t in rows[0]}
    torch.save(head.state_dict(), a.out / f'rgb_head_{a.clip}.pt')
    res = dict(clip=a.clip, ckpt=str(ckpt_path), step=int(step), head_params=147,
               train_frames=len(sel), train_rays=int(n), fit_seconds=fit_s,
               eval_frames=len(rows), eval=agg,
               note='B0 CNN area-downsampled to the ray-res grid for like-for-like; '
                    'all scene params frozen, head only')
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    (a.out / 'manifest.yaml').write_text(
        f"clip_id: {a.clip}\nsplit_id: nvs_50_50 (overnight1.json)\ncommit: nff-v2-ablation+T1\n"
        f"init_checkpoint: {ckpt_path}\nbackbone: neurad\nrgb_head_variant: sample_linear(147p)\n"
        f"medium: none\nfixed_params: all-but-head\nfree_params: sample_rgb_head\n"
        f"seed: 0\nmetrics_file: results.json\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
