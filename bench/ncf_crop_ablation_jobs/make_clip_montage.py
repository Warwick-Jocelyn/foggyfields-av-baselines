#!/usr/bin/env python3
"""One image per clip for quick eyeballing: rows = method/variant, cols = held-out frames.

Each row shows the SAME camera at the SAME frames, cropped to the safe_crop rectangle --
i.e. exactly the pixels the primary metric is computed on -- so the four rows are visually
comparable in the same way the numbers are. Rightmost column is the ground truth.

Reads the PNG renders under outputs/, so run it before deleting them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_paths import load_geometry, resolve_clip_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_videos import frame_paths, get_font  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default="splatad neurad")
    ap.add_argument("--variants", default="safe mask")
    ap.add_argument("--camera", default="camera_front_wide_120fov")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--cell-width", type=int, default=380)
    a = ap.parse_args()

    g = load_geometry(resolve_clip_root(a.data, a.sequence))[a.camera]
    font = get_font(14)
    rows: list[tuple[str, list[Image.Image]]] = []
    gt_row: list[Image.Image] | None = None

    def crop_scale(p: Path, is_canvas: bool) -> Image.Image:
        im = Image.open(p).convert("RGB")
        if is_canvas:
            im = im.crop((g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]))
        h = max(2, int(round(im.height * a.cell_width / im.width)))
        return im.resize((a.cell_width, h), Image.Resampling.BILINEAR)

    for m in a.methods.split():
        for v in a.variants.split():
            d = a.outputs / m / v / a.sequence / "renders" / "val"
            cam_dir = d / "rgb" / a.camera
            if not cam_dir.is_dir():
                print(f"  (skip {m}/{v}: no renders at {cam_dir})")
                continue
            fr = frame_paths(cam_dir)
            if not fr:
                continue
            idx = np.linspace(0, len(fr) - 1, min(a.frames, len(fr))).round().astype(int)
            is_canvas = v == "mask"
            rows.append((f"{m} / {v}", [crop_scale(fr[i], is_canvas) for i in idx]))
            if gt_row is None:
                gt_row = [crop_scale(d / "gt-rgb" / a.camera / fr[i].name, is_canvas) for i in idx]
    if not rows:
        print(f"no renders found for {a.sequence} -- nothing to montage")
        return 0
    if gt_row:
        rows.append(("ground truth", gt_row))

    cw = a.cell_width
    rh = max(im.height for _, ims in rows for im in ims)
    ncol = max(len(ims) for _, ims in rows)
    lab_w = 150
    mont = Image.new("RGB", (lab_w + cw * ncol, rh * len(rows)), (24, 24, 24))
    d = ImageDraw.Draw(mont)
    for r, (label, ims) in enumerate(rows):
        d.text((6, r * rh + rh // 2 - 8), label, fill=(255, 235, 0), font=font)
        for c, im in enumerate(ims):
            mont.paste(im, (lab_w + c * cw, r * rh))
    d.text((6, 4), f"{a.sequence}  {a.camera.replace('camera_', '')}  (safe_crop region = scored pixels)",
           fill=(200, 200, 200), font=font)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    mont.save(a.out, quality=85)
    print(f"wrote {a.out} ({mont.size[0]}x{mont.size[1]}, {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
