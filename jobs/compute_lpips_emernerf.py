"""LPIPS for EmerNeRF: per-frame alex-LPIPS between rendered and GT test videos.

EmerNeRF's built-in eval reports PSNR/SSIM only (lpips: -1 sentinel). To fill
the LPIPS column in the comparison table, we read its final-step test videos
(25000_rgbs.mp4 + 25000_gt_rgbs.mp4 under test_videos/) and compute LPIPS
per frame with torchmetrics (net='alex'), matching what compute_rgb_metrics.py
does for SplatAD/NeuRAD's PNG trees.

    python compute_lpips_emernerf.py <rendered_mp4> <gt_mp4> --output metrics_lpips.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import imageio.v2 as imageio
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rendered", type=Path, help="rendered mp4")
    ap.add_argument("gt", type=Path, help="ground-truth mp4")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    pred_r = imageio.get_reader(str(args.rendered))
    gt_r = imageio.get_reader(str(args.gt))

    pn, sn, ln, n = [], [], [], 0
    for pred_np, gt_np in zip(pred_r, gt_r):
        pred = torch.from_numpy(pred_np[..., :3]).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        gt = torch.from_numpy(gt_np[..., :3]).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        h = min(pred.shape[2], gt.shape[2]); w = min(pred.shape[3], gt.shape[3])
        pred, gt = pred[:, :, :h, :w], gt[:, :, :h, :w]
        with torch.no_grad():
            pn.append(psnr(pred, gt).item())
            sn.append(ssim(pred, gt).item())
            ln.append(lpips(pred.clamp(0, 1), gt.clamp(0, 1)).item())
        n += 1

    out = {
        "num_frames": n,
        "psnr": sum(pn) / len(pn) if pn else None,
        "ssim": sum(sn) / len(sn) if sn else None,
        "lpips": sum(ln) / len(ln) if ln else None,
        "source": f"final-step test videos ({args.rendered.name}, {args.gt.name})",
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
