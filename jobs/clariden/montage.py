"""Side-by-side (pred | GT) montage of held-out renders: rows = cameras, cols = frames. Usage: montage.py <renders/val> <out.jpg>"""
import sys
from PIL import Image
V, out = sys.argv[1], sys.argv[2]
cams = ["camera_front_wide_120fov", "camera_cross_left_120fov", "camera_rear_left_70fov", "camera_front_tele_30fov"]
frames = [1, 49, 99, 149, 197]; tw, th = 320, 240
im = Image.new("RGB", (tw * len(frames) * 2, th * len(cams)))
for r, cam in enumerate(cams):
    for c, t in enumerate(frames):
        for k, kind in enumerate(["rgb", "gt-rgb"]):
            try: tile = Image.open(f"{V}/{kind}/{cam}/frame_{t:04d}.png").resize((tw, th))
            except FileNotFoundError: continue
            im.paste(tile, ((c * 2 + k) * tw, r * th))
im.save(out, quality=80); print("wrote", out)
