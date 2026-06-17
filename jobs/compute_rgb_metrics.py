"""Compute mean PSNR/SSIM/LPIPS between rendered and GT images from an
`ns-render dataset` PNG tree.

Used to obtain SplatAD's held-out (val) RGB metrics: ns-eval is infeasible for
SplatAD here because its ~93GB resident model leaves no room for the LiDAR
chamfer metric (GPU OOM) and the CPU fallback exceeds host RAM. The rendered val
frames (rgb vs gt-rgb) give the same RGB metrics directly.

    python compute_rgb_metrics.py SPLIT_DIR --output metrics.json

SPLIT_DIR contains rgb/<cam>/*.png and gt-rgb/<cam>/*.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import imageio.v2 as imageio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def _load(p: Path, device) -> torch.Tensor:
    img = imageio.imread(p)
    if img.ndim == 2:
        img = img[..., None].repeat(3, -1)
    img = img[..., :3]
    t = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", type=Path, help="dir with rgb/ and gt-rgb/ subdirs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    rgb_root = args.split_dir / "rgb"
    gt_root = args.split_dir / "gt-rgb"
    cams = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())

    ps, ss, ls, n = [], [], [], 0
    for cam in cams:
        preds = sorted((rgb_root / cam).glob("*.png"))
        for pred in preds:
            gt = gt_root / cam / pred.name
            if not gt.exists():
                continue
            pi, gi = _load(pred, device), _load(gt, device)
            h = min(pi.shape[2], gi.shape[2]); w = min(pi.shape[3], gi.shape[3])
            pi, gi = pi[:, :, :h, :w], gi[:, :, :h, :w]
            with torch.no_grad():
                ps.append(psnr(pi, gi).item())
                ss.append(ssim(pi, gi).item())
                ls.append(lpips(pi.clamp(0, 1), gi.clamp(0, 1)).item())
            n += 1

    out = {
        "num_images": n,
        "psnr": sum(ps) / len(ps) if ps else None,
        "ssim": sum(ss) / len(ss) if ss else None,
        "lpips": sum(ls) / len(ls) if ls else None,
        "source": "rendered val PNGs (rgb vs gt-rgb)",
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
