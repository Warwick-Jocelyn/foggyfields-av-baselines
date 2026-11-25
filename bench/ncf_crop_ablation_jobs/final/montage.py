#!/usr/bin/env python3
"""Side-by-side comparison montage (CPU): GT | every finished method, one held-out frame, selected cameras.

  python3 montage.py --clip NFF-08 --frame 0099 --out <dir>
Writes <out>/<clip>__frame_<f>__<cam>.jpg (rows = cameras, columns = GT + methods). Methods without renders are skipped.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BW = Path("/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/outputs_final")
CAMS = ["camera_front_wide_120fov", "camera_front_tele_30fov"]


def methods(clip):
    m = [("SplatAD allfix", BW / f"allfix/splatad/safe/{clip}/renders/val"),
         ("NeuRAD std", BW / f"std/neurad/safe/{clip}/renders/val"),
         ("UniSim", BW / f"std/unisim/safe/{clip}/renders/val"),
         ("EmerNeRF", BW / f"std/emernerf/safe/{clip}/renders/val")]
    for lb, nm in (("V13-UcnnR", "V1.3 Colour"), ("V13-U0", "V1.3 BRDF"), ("V13-C7gR", "V1.3 BRDF"), ("V13-Fc", "V1.3 Colour+L"), ("V13-F", "V1.3 BRDF+L")):
        m.append((nm, BW / f"v13/{lb}__{clip}/renders/val"))
    seen, res = set(), []
    for nm, d in m:
        if nm in seen:
            continue
        if (d / "rgb").is_dir():
            res.append((nm, d)); seen.add(nm)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--frame", default="0099")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args()
    ms = methods(a.clip)
    if not ms:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for cam in CAMS:
        gt = ms[0][1] / "gt-rgb" / cam / f"frame_{a.frame}.png"
        tiles = [("GT (held-out)", gt)] + [(nm, d / "rgb" / cam / f"frame_{a.frame}.png") for nm, d in ms]
        tiles = [(n, p) for n, p in tiles if p.is_file()]
        if len(tiles) < 2:
            continue
        ims = []
        for n, p in tiles:
            im = Image.open(p).convert("RGB")
            h = round(im.height * a.width / im.width)
            im = im.resize((a.width, h))
            ImageDraw.Draw(im).rectangle([0, 0, a.width, 18], fill=(0, 0, 0))
            ImageDraw.Draw(im).text((4, 3), n, fill=(255, 255, 255), font=font)
            ims.append(im)
        cols = 3
        rows = (len(ims) + cols - 1) // cols
        H = max(i.height for i in ims)
        canvas = Image.new("RGB", (cols * a.width, rows * H), (255, 255, 255))
        for k, im in enumerate(ims):
            canvas.paste(im, ((k % cols) * a.width, (k // cols) * H))
        dst = a.out / f"{a.clip}__frame_{a.frame}__{cam.replace('camera_', '')}.jpg"
        canvas.save(dst, quality=88)
        print(dst)


if __name__ == "__main__":
    main()
