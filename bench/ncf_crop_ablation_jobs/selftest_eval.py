#!/usr/bin/env python3
"""End-to-end self-test of the common-region metric. CPU-only, ~1 min, no GPU needed.

It fabricates two render trees from the SAME synthetic pixels:

  tree_mask : full v2 canvas  (what the `mask` variant produces)
  tree_safe : the same pixels cropped to safe_crop (what the `safe` variant produces)

and asserts that ``eval_common_region.py`` returns the SAME primary PSNR/SSIM/LPIPS for
both. That is the one property the entire study rests on -- if the crop offset, the
variant branch, or the size assertion were wrong, these two numbers would differ.

It also asserts that:
  * a canvas-sized render scored as `safe` is REJECTED (size assertion works),
  * the mask variant's native unmasked PSNR is far below its common-region PSNR when the
    render is garbage outside the valid region (the trap this study exists to avoid).

Run with --camera to pick a cheaper camera (default: the smallest one).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clip_paths import CAMERAS, load_geometry, resolve_clip_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--camera", default="")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--tol-psnr", type=float, default=1e-3)
    a = ap.parse_args()

    clip_root = resolve_clip_root(a.data, a.sequence)
    geo = load_geometry(clip_root)
    cam = a.camera or min(CAMERAS, key=lambda c: geo[c]["canvas_w"] * geo[c]["canvas_h"])
    g = geo[cam]
    print(f"self-test on {a.sequence}/{cam}: canvas {g['canvas_w']}x{g['canvas_h']} "
          f"safe_crop ({g['x']},{g['y']}) {g['w']}x{g['h']}")

    vm = np.asarray(Image.open(clip_root / "masks_rectv2" / cam / "valid_mask.png"))
    vm = (vm[..., 0] if vm.ndim == 3 else vm) > 128

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix="ncf_selftest_") as td:
        tmp = Path(td)
        trees = {}
        for variant in ("mask", "safe"):
            root = tmp / f"tree_{variant}" / "val"
            (root / "rgb" / cam).mkdir(parents=True)
            (root / "gt-rgb" / cam).mkdir(parents=True)
            trees[variant] = root
        for i in range(a.frames):
            base = rng.integers(0, 256, (g["canvas_h"], g["canvas_w"], 3), dtype=np.uint8)
            gt = np.where(vm[..., None], base, 0).astype(np.uint8)          # black outside the lens cone
            noise = rng.normal(0, 12, base.shape)
            pred = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            # garbage where the mask variant was never supervised
            pred = np.where(vm[..., None], pred, rng.integers(0, 256, base.shape, dtype=np.uint8))
            name = f"frame_{i:04d}.png"
            Image.fromarray(gt).save(trees["mask"] / "gt-rgb" / cam / name)
            Image.fromarray(pred).save(trees["mask"] / "rgb" / cam / name)
            y, x, h, w = g["y"], g["x"], g["h"], g["w"]
            Image.fromarray(gt[y : y + h, x : x + w]).save(trees["safe"] / "gt-rgb" / cam / name)
            Image.fromarray(pred[y : y + h, x : x + w]).save(trees["safe"] / "rgb" / cam / name)

        results = {}
        for variant in ("mask", "safe"):
            out = tmp / f"m_{variant}.json"
            cmd = [sys.executable, str(HERE / "eval_common_region.py"),
                   "--renders", str(trees[variant]), "--data", str(a.data), "--sequence", a.sequence,
                   "--variant", variant, "--cameras", cam, "--device", "cpu", "-o", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout); print(r.stderr, file=sys.stderr)
                raise SystemExit(f"SELFTEST FAILED: eval_common_region.py rc={r.returncode} for variant={variant}")
            results[variant] = json.loads(out.read_text())

        pm, ps = results["mask"]["primary"], results["safe"]["primary"]
        print(f"  mask-variant common region: PSNR {pm['psnr']:.6f} SSIM {pm['ssim']:.6f} LPIPS {pm['lpips']:.6f}")
        print(f"  safe-variant common region: PSNR {ps['psnr']:.6f} SSIM {ps['ssim']:.6f} LPIPS {ps['lpips']:.6f}")
        bad = []
        if abs(pm["psnr"] - ps["psnr"]) > a.tol_psnr:
            bad.append(f"PSNR differs by {abs(pm['psnr'] - ps['psnr']):.6f} (> {a.tol_psnr})")
        if abs(pm["ssim"] - ps["ssim"]) > 1e-4:
            bad.append(f"SSIM differs by {abs(pm['ssim'] - ps['ssim']):.6f}")
        if abs(pm["lpips"] - ps["lpips"]) > 1e-3:
            bad.append(f"LPIPS differs by {abs(pm['lpips'] - ps['lpips']):.6f}")
        if bad:
            raise SystemExit("SELFTEST FAILED: the two variants do not agree on identical pixels: " + "; ".join(bad))
        print("  ok  identical pixels -> identical primary metrics")

        nat = results["mask"]["secondary_NOT_COMPARABLE"]["native_unmasked_DIAGNOSTIC_ONLY"]["psnr"]
        print(f"  mask-variant native unmasked PSNR {nat:.3f} vs common {pm['psnr']:.3f}")
        if nat >= pm["psnr"]:
            raise SystemExit("SELFTEST FAILED: unmasked PSNR should collapse when the corners are unsupervised")
        print("  ok  the unmasked-PSNR trap is reproduced (and excluded from the primary number)")

        # the size assertion must reject a canvas-sized render labelled `safe`
        out = tmp / "m_bad.json"
        r = subprocess.run([sys.executable, str(HERE / "eval_common_region.py"),
                            "--renders", str(trees["mask"]), "--data", str(a.data), "--sequence", a.sequence,
                            "--variant", "safe", "--cameras", cam, "--device", "cpu", "-o", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            raise SystemExit("SELFTEST FAILED: a canvas-sized render was accepted as variant 'safe'")
        print("  ok  mislabelled render size is rejected")

    print("SELFTEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
