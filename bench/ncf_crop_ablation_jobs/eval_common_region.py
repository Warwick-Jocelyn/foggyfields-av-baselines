#!/usr/bin/env python3
"""Score a crop/mask ablation run on the ONE pixel set both variants share.

Why this exists
---------------
The two variants are not comparable on their native pixels:

  * ``safe`` renders the safe_crop rectangle (no black corners, narrower FOV).
  * ``mask`` renders the full v2 canvas; the black lens-cone corners were excluded from
    the RGB loss, so the model is *unconstrained* there. Its unmasked PSNR is ~17-18 dB
    while its masked PSNR is ~28 -- comparing unmasked numbers inverts the conclusion.
    And scoring a full canvas *including* the black pixels inflates PSNR, because black
    is a zero-cost prediction wherever the GT is also black.

``safe_crop`` is by construction the largest rectangle with zero invalid pixels, so
``safe_crop n valid_mask == safe_crop``. That rectangle is therefore a fair common
region, and it is the ONLY region the primary numbers are computed on:

  safe : the render already *is* that rectangle -> use as is
  mask : the render is the full canvas      -> crop to the safe_crop box first

Both paths assert the extracted region is exactly (safe_crop.h, safe_crop.w), so a
silent size/offset mistake cannot turn into a plausible number.

Secondary, clearly-labelled, NOT comparable to the primary number (mask variant only):
  wide_masked          full canvas n valid_mask  -- what the extra valid pixels look like
  outside_safe_masked  valid_mask AND NOT safe_crop -- what the extra region alone buys
  native_unmasked      full canvas incl. black corners -- DIAGNOSTIC ONLY, the trap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_paths import CAMERAS, load_geometry, resolve_clip_root  # noqa: E402

# Tolerated shrink (px) of a render vs its expected size: `image_divisible_by > 1`
# crops a few pixels off the right/bottom edge. Anything larger is a real mismatch.
SIZE_SLACK = 8


def load_rgb(path: Path, dev: str) -> torch.Tensor:
    """-> [1, 3, H, W] float in [0, 1]."""
    im = np.asarray(Image.open(path).convert("RGB"))
    return torch.from_numpy(np.ascontiguousarray(im)).float().div_(255.0).permute(2, 0, 1)[None].to(dev)


def psnr_of(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """PSNR with data_range=1. ``mask`` is [1,1,H,W] bool; None = every pixel."""
    se = (pred - gt) ** 2
    mse = se.mean() if mask is None else se[mask.expand_as(se)].mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", type=Path, required=True,
                    help="the split dir written by `ns-render dataset`, i.e. <output-path>/val (holds rgb/ and gt-rgb/)")
    ap.add_argument("--data", type=Path, required=True, help="dataset root (holds clips/)")
    ap.add_argument("--sequence", required=True, help="clip selector, e.g. NCF-02")
    ap.add_argument("--variant", required=True, choices=["safe", "mask"])
    ap.add_argument("--method", default="unknown")
    ap.add_argument("--cameras", default="", help="space-separated subset; default = all 7, and all 7 are required")
    ap.add_argument("--no-wide-perceptual", action="store_true",
                    help="skip wide SSIM/LPIPS (they are the expensive full-canvas ones)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--output", type=Path, required=True)
    a = ap.parse_args()

    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    dev = a.device
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)

    clip_root = resolve_clip_root(a.data, a.sequence)
    geo = load_geometry(clip_root)
    mask_dir = clip_root / "masks_rectv2"

    want = a.cameras.split() if a.cameras else list(CAMERAS)
    rgb_root, gt_root = a.renders / "rgb", a.renders / "gt-rgb"
    for d in (rgb_root, gt_root):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} missing -- did `ns-render dataset --rendered-output-names rgb gt-rgb` run?")
    present = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())
    missing = [c for c in want if c not in present]
    if missing:
        raise FileNotFoundError(
            f"renders are missing cameras {missing} (found {present}). Every one of the 7 cameras must be scored, "
            "otherwise the two variants are averaged over different sensors."
        )

    per_cam: dict[str, dict] = {}
    pool: dict[str, list[float]] = {}
    notes: list[str] = []

    def push(key: str, val: float | None) -> None:
        if val is not None and np.isfinite(val):
            pool.setdefault(key, []).append(val)

    for cam in want:
        g = geo[cam]
        safe_box = (g["x"], g["y"], g["w"], g["h"])
        preds = sorted(rgb_root.joinpath(cam).glob("*.png")) + sorted(rgb_root.joinpath(cam).glob("*.jpg"))
        if not preds:
            raise FileNotFoundError(f"no rendered frames under {rgb_root / cam}")

        # ---- establish the geometry from the first frame, loudly -------------------
        first = np.asarray(Image.open(preds[0]))
        rh, rw = first.shape[:2]
        exp_w, exp_h = (g["w"], g["h"]) if a.variant == "safe" else (g["canvas_w"], g["canvas_h"])
        if not (0 <= exp_w - rw <= SIZE_SLACK and 0 <= exp_h - rh <= SIZE_SLACK):
            raise AssertionError(
                f"{cam}: variant '{a.variant}' render is {rw}x{rh} but expected {exp_w}x{exp_h} "
                f"(safe_crop {g['w']}x{g['h']}, canvas {g['canvas_w']}x{g['canvas_h']}). "
                "Either the variant label is wrong, --apply-safe-crop did not take effect, or the "
                "images were rescaled -- in any of those cases the common region would be wrong."
            )
        if (rw, rh) != (exp_w, exp_h):
            notes.append(f"{cam}: render {rw}x{rh} is {exp_w - rw}x{exp_h - rh} px smaller than expected "
                         f"{exp_w}x{exp_h} (image_divisible_by cropping); region trimmed to match")

        if a.variant == "safe":
            cx0, cy0 = 0, 0
            cw, ch = rw, rh
        else:
            cx0, cy0 = g["x"], g["y"]
            cw, ch = min(g["w"], rw - cx0), min(g["h"], rh - cy0)
            if cw <= 0 or ch <= 0:
                raise AssertionError(f"{cam}: safe_crop box {safe_box} does not fit inside a {rw}x{rh} render")

        # valid mask (canvas resolution), needed for the secondary metrics only
        vmask = None
        if a.variant == "mask":
            mp_ = mask_dir / cam / "valid_mask.png"
            if not mp_.is_file():
                raise FileNotFoundError(f"{mp_} missing -- needed for the secondary 'wide' metric")
            mm = np.asarray(Image.open(mp_))
            mm = mm[..., 0] if mm.ndim == 3 else mm
            if mm.shape[:2] != (g["canvas_h"], g["canvas_w"]):
                raise AssertionError(f"{cam}: mask is {mm.shape[1]}x{mm.shape[0]}, canvas is {g['canvas_w']}x{g['canvas_h']}")
            vmask = torch.from_numpy(mm > 128)[None, None].to(dev)[..., :rh, :rw]
            outside = vmask.clone()
            outside[..., cy0 : cy0 + ch, cx0 : cx0 + cw] = False
            if int(vmask[..., cy0 : cy0 + ch, cx0 : cx0 + cw].logical_not().sum()) != 0:
                raise AssertionError(
                    f"{cam}: valid_mask has invalid pixels inside safe_crop -- the common-region premise is broken"
                )
        else:
            outside = None

        rows: list[dict] = []
        for pred_path in preds:
            gt_path = gt_root / cam / pred_path.name
            if not gt_path.is_file():
                raise FileNotFoundError(f"no ground truth for {pred_path} (expected {gt_path})")
            p = load_rgb(pred_path, dev)[..., :rh, :rw]
            t = load_rgb(gt_path, dev)[..., :rh, :rw]
            if p.shape != t.shape:
                raise AssertionError(f"{pred_path.name}: render {tuple(p.shape)} vs gt {tuple(t.shape)}")

            pc = p[..., cy0 : cy0 + ch, cx0 : cx0 + cw]
            tc = t[..., cy0 : cy0 + ch, cx0 : cx0 + cw]
            assert pc.shape[-2:] == (ch, cw)
            row: dict = {"frame": pred_path.stem}
            with torch.no_grad():
                row["psnr"] = psnr_of(pc, tc)
                row["ssim"] = float(ssim(pc, tc))
                row["lpips"] = float(lpips(pc.clamp(0, 1), tc.clamp(0, 1)))
                if vmask is not None:
                    row["wide_psnr"] = psnr_of(p, t, vmask)
                    row["native_unmasked_psnr"] = psnr_of(p, t)
                    row["outside_safe_psnr"] = (
                        psnr_of(p, t, outside) if int(outside.sum()) > 0 else None  # type: ignore[union-attr]
                    )
                    if not a.no_wide_perceptual:
                        try:
                            pm, tm = p * vmask, t * vmask
                            row["wide_ssim"] = float(ssim(pm, tm))
                            row["wide_lpips"] = float(lpips(pm.clamp(0, 1), tm.clamp(0, 1)))
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            row["wide_ssim"] = row["wide_lpips"] = None
                            if "wide perceptual metrics skipped (CUDA OOM)" not in notes:
                                notes.append("wide perceptual metrics skipped (CUDA OOM) -- wide PSNR is unaffected")
            rows.append(row)
            for k, v in row.items():
                if k != "frame":
                    push(k, v)

        agg = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
               for k in rows[0] if k != "frame" and any(r.get(k) is not None for r in rows)}
        per_cam[cam] = {
            "n": len(rows),
            "common_region_hw": [ch, cw],
            "render_wh": [rw, rh],
            "safe_crop_xywh": list(safe_box),
            "canvas_wh": [g["canvas_w"], g["canvas_h"]],
            "metrics": agg,
            "per_frame": rows,
        }
        print(f"  {cam:28s} n={len(rows):4d} region {cw}x{ch}  PSNR {agg['psnr']:6.3f}  "
              f"SSIM {agg['ssim']:.4f}  LPIPS {agg['lpips']:.4f}"
              + (f"  | wide PSNR {agg['wide_psnr']:6.3f}  native(unmasked) {agg['native_unmasked_psnr']:6.3f}"
                 if "wide_psnr" in agg else ""))

    n_images = sum(v["n"] for v in per_cam.values())
    if n_images == 0:
        raise RuntimeError("scored 0 images")
    region_px = {c: v["common_region_hw"][0] * v["common_region_hw"][1] for c, v in per_cam.items()}

    out = {
        "method": a.method,
        "variant": a.variant,
        "clip": a.sequence,
        "clip_root": str(clip_root),
        "renders": str(a.renders),
        "n_images": n_images,
        "cameras": want,
        "primary_region": "safe_crop rectangle (identical pixel set for both variants; "
                          "safe_crop n valid_mask == safe_crop by construction)",
        "psnr_convention": "mean over images of (-10 log10 MSE) with data_range=1; "
                           "SSIM/LPIPS(alex) per image then averaged",
        "common_region_pixels_per_timestep": int(sum(region_px.values())),
        "primary": {k: float(np.mean(v)) for k, v in pool.items() if k in ("psnr", "ssim", "lpips")},
        "secondary_NOT_COMPARABLE": {
            "wide_masked": {k.replace("wide_", ""): float(np.mean(pool[k]))
                            for k in ("wide_psnr", "wide_ssim", "wide_lpips") if k in pool} or None,
            "outside_safe_masked": ({"psnr": float(np.mean(pool["outside_safe_psnr"]))}
                                    if "outside_safe_psnr" in pool else None),
            "native_unmasked_DIAGNOSTIC_ONLY": ({"psnr": float(np.mean(pool["native_unmasked_psnr"]))}
                                                if "native_unmasked_psnr" in pool else None),
        },
        "notes": notes,
        "per_camera": per_cam,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    pr = out["primary"]
    print(f"[{a.method}/{a.variant}/{a.sequence}] COMMON REGION (safe_crop) n={n_images}  "
          f"PSNR {pr['psnr']:.3f}  SSIM {pr['ssim']:.4f}  LPIPS {pr['lpips']:.4f}")
    sec = out["secondary_NOT_COMPARABLE"]
    if sec["wide_masked"]:
        print(f"      secondary (NOT comparable): full canvas n valid_mask PSNR {sec['wide_masked']['psnr']:.3f}; "
              f"valid-outside-safe_crop PSNR "
              f"{sec['outside_safe_masked']['psnr'] if sec['outside_safe_masked'] else float('nan'):.3f}; "
              f"native unmasked (the trap) {sec['native_unmasked_DIAGNOSTIC_ONLY']['psnr']:.3f}")
    for n in notes:
        print(f"      note: {n}")
    print(f"wrote {a.output}")
    print("EVAL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
