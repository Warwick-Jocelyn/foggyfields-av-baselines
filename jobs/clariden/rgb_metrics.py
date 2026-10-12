"""PSNR/SSIM/LPIPS over an `ns-render dataset` PNG tree, unmasked AND masked by the per-camera valid_mask.
Unmasked = identical to the original compute_rgb_metrics.py (torchmetrics, LPIPS-AlexNet).
Masked PSNR = MSE over valid pixels only (NFF-bench definition); masked SSIM/LPIPS = both images zeroed
outside the valid region (the SplatAD in-model convention)."""
import argparse, json
from pathlib import Path
import numpy as np, torch, imageio.v2 as imageio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def load(p, dev):
    im = imageio.imread(p)
    if im.ndim == 2: im = im[..., None].repeat(3, -1)
    return torch.from_numpy(np.ascontiguousarray(im[..., :3])).float().div(255).permute(2, 0, 1)[None].to(dev)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split_dir", type=Path); ap.add_argument("--masks-root", type=Path)
    ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev); ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    cams = sorted(p.name for p in (a.split_dir / "rgb").iterdir() if p.is_dir())
    per_cam, agg = {}, {k: [] for k in ("psnr", "ssim", "lpips", "psnr_masked", "ssim_masked", "lpips_masked")}
    n = 0
    for cam in cams:
        mask = None
        if a.masks_root and (a.masks_root / cam / "valid_mask.png").exists():
            m = imageio.imread(a.masks_root / cam / "valid_mask.png"); m = m[..., 0] if m.ndim == 3 else m
            mask = torch.from_numpy(m > 128).to(dev)[None, None]  # 1,1,H,W
        rows = {k: [] for k in agg}
        for pred in sorted((a.split_dir / "rgb" / cam).glob("*.png")):
            gt = a.split_dir / "gt-rgb" / cam / pred.name
            if not gt.exists(): continue
            pi, gi = load(pred, dev), load(gt, dev)
            h, w = min(pi.shape[2], gi.shape[2]), min(pi.shape[3], gi.shape[3]); pi, gi = pi[..., :h, :w], gi[..., :h, :w]
            with torch.no_grad():
                rows["psnr"].append(psnr(pi, gi).item()); rows["ssim"].append(ssim(pi, gi).item())
                rows["lpips"].append(lpips(pi.clamp(0, 1), gi.clamp(0, 1)).item())
                if mask is not None:
                    mk = mask[..., :h, :w]
                    mse = ((pi - gi) ** 2)[mk.expand_as(pi)].mean()
                    rows["psnr_masked"].append((-10 * torch.log10(mse.clamp_min(1e-12))).item())
                    pm, gm = pi * mk, gi * mk
                    rows["ssim_masked"].append(ssim(pm, gm).item()); rows["lpips_masked"].append(lpips(pm.clamp(0, 1), gm.clamp(0, 1)).item())
            n += 1
        per_cam[cam] = {k: (float(np.mean(v)) if v else None) for k, v in rows.items()}; per_cam[cam]["n"] = len(rows["psnr"])
        for k in agg: agg[k] += rows[k]
    out = {"num_images": n, **{k: (float(np.mean(v)) if v else None) for k, v in agg.items()}, "per_camera": per_cam,
           "source": "rendered val PNGs (rgb vs gt-rgb); *_masked uses masks/<cam>/valid_mask.png"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_camera"}, indent=2))

if __name__ == "__main__": main()
