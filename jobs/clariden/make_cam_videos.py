"""Per-camera held-out videos: [ rendered | ground truth ] side by side, cropped to the valid-pixel bounding
box (removes the black ftheta->pinhole corners), 10 fps.  Also [ rendered RGB | rendered depth ] for the front
camera.  Usage: make_cam_videos.py <renders/val> <masks_root> <out_dir> [--runs-label TEXT]"""
import sys, argparse, numpy as np, imageio.v2 as iio
from pathlib import Path
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("val"); ap.add_argument("masks"); ap.add_argument("out"); ap.add_argument("--label", default=""); ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args(); V, M, O = Path(a.val), Path(a.masks), Path(a.out); O.mkdir(parents=True, exist_ok=True)
cams = sorted(p.name for p in (V / "rgb").iterdir() if p.is_dir())
def bbox(mask):  # tight box around the valid region, then shrink to the largest axis-aligned rectangle that is fully valid
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # shrink rows/cols until >99% of the pixels inside the box are valid
    while mask[y0:y1 + 1, x0:x1 + 1].mean() < 0.99 and (y1 - y0) > 40 and (x1 - x0) > 40:
        cand = [mask[y0 + 4:y1 + 1, x0:x1 + 1].mean(), mask[y0:y1 - 3, x0:x1 + 1].mean(), mask[y0:y1 + 1, x0 + 4:x1 + 1].mean(), mask[y0:y1 + 1, x0:x1 - 3].mean()]
        k = int(np.argmax(cand)); y0, y1, x0, x1 = [(y0 + 4, y1, x0, x1), (y0, y1 - 4, x0, x1), (y0, y1, x0 + 4, x1), (y0, y1, x0, x1 - 4)][k]
    y1, x1 = y1 + 1, x1 + 1
    return int(y0), int(y1 - (y1 - y0) % 2), int(x0), int(x1 - (x1 - x0) % 2)  # even crop size for libx264
def label(img, txt):
    d = ImageDraw.Draw(img); d.rectangle([0, 0, 8 + 7 * len(txt), 18], fill=(0, 0, 0)); d.text((4, 3), txt, fill=(255, 255, 0)); return img
for cam in cams:
    m = iio.imread(M / cam / "valid_mask.png"); m = (m[..., 0] if m.ndim == 3 else m) > 128; y0, y1, x0, x1 = bbox(m)
    frames = sorted((V / "rgb" / cam).glob("*.png"))
    w = iio.get_writer(O / f"{cam}__pred_vs_gt.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=2)
    for f in frames:
        p = Image.open(f).convert("RGB").crop((x0, y0, x1, y1)); g = Image.open(V / "gt-rgb" / cam / f.name).convert("RGB").crop((x0, y0, x1, y1))
        c = Image.new("RGB", (p.width * 2 + 6, p.height), (40, 40, 40)); c.paste(label(p, f"SplatAD {a.label} {f.stem}"), (0, 0)); c.paste(label(g, "ground truth"), (p.width + 6, 0))
        w.append_data(np.asarray(c))
    w.close()
    if cam == "camera_front_wide_120fov" and (V / "depth" / cam).exists():
        w = iio.get_writer(O / f"{cam}__pred_rgb_vs_pred_depth.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=2)
        for f in frames:
            p = Image.open(f).convert("RGB").crop((x0, y0, x1, y1)); d = Image.open(V / "depth" / cam / f.name).convert("RGB").crop((x0, y0, x1, y1))
            c = Image.new("RGB", (p.width * 2 + 6, p.height), (40, 40, 40)); c.paste(label(p, f"rendered RGB {f.stem}"), (0, 0)); c.paste(label(d, "rendered depth (turbo, near=blue)"), (p.width + 6, 0)); w.append_data(np.asarray(c))
        w.close()
    print("wrote", cam, "crop", (x0, y0, x1, y1))
