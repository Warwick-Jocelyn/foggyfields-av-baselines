#!/usr/bin/env python3
"""One video showing ALL 7 cameras at once, in a surround layout.

Each cell is that camera's RENDER above its GT, so every view can be compared without
opening seven files. Layout mirrors the rig:

      cross_left      front_wide      cross_right
                front_tele  rear_tele
      rear_left                        rear_right

Usage:
  make_surround_video.py --renders <run>/renders/val -o out.mp4 [--cell-w 480] [--fps 10]
                         [--max-frames N] [--label "safe / NCF-15"]
  make_surround_video.py --renders-a <A>/renders/val --renders-b <B>/renders/val \
                         --label-a safe --label-b mask -o cmp.mp4     # two variants, render only
"""
import argparse, sys
from pathlib import Path
import numpy as np, av
from PIL import Image, ImageDraw, ImageFont

ROWS = [
    ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"],
    ["camera_front_tele_30fov", "camera_rear_tele_30fov"],
    ["camera_rear_left_70fov", "camera_rear_right_70fov"],
]

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    try:
        import glob as g
        c = g.glob(str(Path(sys.executable).parent.parent) + "/**/DejaVuSans*.ttf", recursive=True)
        if c: return ImageFont.truetype(c[0], sz)
    except Exception: pass
    return ImageFont.load_default()

def tag(im, text, f, bg=(0, 0, 0)):
    d = ImageDraw.Draw(im)
    b = d.textbbox((0, 0), text, font=f)
    d.rectangle([0, 0, b[2] + 10, b[3] + 8], fill=bg)
    d.text((5, 3), text, fill=(255, 255, 255), font=f)

def fit(p, w):
    im = Image.open(p).convert("RGB")
    return im.resize((w, max(2, round(im.height * w / im.width))), Image.LANCZOS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path)
    ap.add_argument("--renders-a", type=Path); ap.add_argument("--renders-b", type=Path)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--label", default="")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--cell-w", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    two = a.renders_a is not None
    if two:
        top_root, bot_root, top_sub, bot_sub = a.renders_a, a.renders_b, "rgb", "rgb"
        top_lbl, bot_lbl = a.label_a, a.label_b
    else:
        top_root = bot_root = a.renders
        top_sub, bot_sub = "rgb", "gt-rgb"
        top_lbl, bot_lbl = "RENDER", "GT"

    cams = [c for r in ROWS for c in r if (top_root / top_sub / c).is_dir()]
    if not cams:
        print(f"no camera dirs under {top_root/top_sub}", file=sys.stderr); return 2
    names = sorted(p.name for p in (top_root / top_sub / cams[0]).glob("*.png"))
    if a.max_frames: names = names[: a.max_frames]
    if not names:
        print("no frames", file=sys.stderr); return 2

    f_small, f_big = font(max(12, a.cell_w // 26)), font(max(16, a.cell_w // 18))
    GAP, PAD = 6, 10

    def build(name):
        row_imgs = []
        for row in ROWS:
            cells = []
            for cam in row:
                pa, pb = top_root/top_sub/cam/name, bot_root/bot_sub/cam/name
                if not (pa.exists() and pb.exists()):
                    continue
                ia, ib = fit(pa, a.cell_w), fit(pb, a.cell_w)
                tag(ia, f"{cam.replace('camera_','')}  {top_lbl}", f_small)
                tag(ib, bot_lbl, f_small, bg=(40, 40, 40))
                cell = Image.new("RGB", (a.cell_w, ia.height + 3 + ib.height), (18, 18, 18))
                cell.paste(ia, (0, 0)); cell.paste(ib, (0, ia.height + 3))
                cells.append(cell)
            if not cells: continue
            h = max(c.height for c in cells)
            w = len(cells) * a.cell_w + (len(cells) - 1) * GAP
            strip = Image.new("RGB", (w, h), (18, 18, 18))
            for i, c in enumerate(cells):
                strip.paste(c, (i * (a.cell_w + GAP), 0))
            row_imgs.append(strip)
        W = max(r.width for r in row_imgs)
        hdr = 0 if not a.label else f_big.size + 12
        H = hdr + sum(r.height for r in row_imgs) + GAP * (len(row_imgs) - 1)
        canvas = Image.new("RGB", (W + 2*PAD, H + 2*PAD), (18, 18, 18))
        if a.label:
            ImageDraw.Draw(canvas).text((PAD, PAD), a.label, fill=(255, 255, 255), font=f_big)
        y = PAD + hdr
        for r in row_imgs:
            canvas.paste(r, (PAD + (W - r.width) // 2, y)); y += r.height + GAP
        w, h = canvas.size
        return canvas.crop((0, 0, w - w % 2, h - h % 2))

    first = build(names[0])
    W, H = first.size
    print(f"{len(names)} frames, {len(cams)} cameras, canvas {W}x{H} -> {a.output}", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(a.output), "w") as c:
        st = c.add_stream("libx264", rate=a.fps)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "20", "preset": "medium"}
        for i, n in enumerate(names):
            im = first if i == 0 else build(n)
            if im.size != (W, H): im = im.resize((W, H), Image.LANCZOS)
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode(): c.mux(pkt)
    print(f"OK {a.output}  {a.output.stat().st_size/1e6:.1f} MB")
    return 0

if __name__ == "__main__":
    sys.exit(main())
