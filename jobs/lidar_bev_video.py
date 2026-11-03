"""BEV GT-vs-pred video from per-frame npz dumps (PyAV, no ffmpeg binary needed)."""
import argparse, io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import av
from PIL import Image

def bev(ax, xyz, title, rng):
    m = (np.abs(xyz[:, 0]) < rng) & (np.abs(xyz[:, 1]) < rng)
    ax.scatter(xyz[m, 0], xyz[m, 1], s=0.05, c=np.clip(xyz[m, 2], -2, 6),
               cmap="viridis", linewidths=0)
    ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_aspect("equal")
    ax.set_facecolor("#0d0d0d"); ax.set_title(title, fontsize=9, color="w")
    ax.tick_params(labelsize=6, colors="#888")
    for s in ax.spines.values(): s.set_color("#555")

ap = argparse.ArgumentParser()
ap.add_argument("--dump", type=Path, required=True)
ap.add_argument("-o", "--out", type=Path, required=True)
ap.add_argument("--range", type=float, default=60.0)
ap.add_argument("--fps", type=int, default=10)
a = ap.parse_args()
files = sorted(a.dump.glob("frame_*.npz"))
assert files, f"no npz under {a.dump}"
a.out.parent.mkdir(parents=True, exist_ok=True)
cont = av.open(str(a.out), "w")
stream = None
for i, fp in enumerate(files):
    d = np.load(fp)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4), facecolor="#0d0d0d")
    bev(axs[0], d["gt_xyz"], f"GT  {fp.stem}", a.range)
    bev(axs[1], d["pred_xyz"], "PREDICTED", a.range)
    fig.suptitle(a.dump.name, color="w", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, facecolor="#0d0d0d"); plt.close(fig)
    img = Image.open(buf).convert("RGB")
    W, H = (img.width // 2) * 2, (img.height // 2) * 2
    img = img.crop((0, 0, W, H))
    if stream is None:
        stream = cont.add_stream("h264", rate=a.fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
    frame = av.VideoFrame.from_image(img)
    for pkt in stream.encode(frame): cont.mux(pkt)
    if i % 20 == 0: print(f"{i}/{len(files)}", flush=True)
for pkt in stream.encode(): cont.mux(pkt)
cont.close()
print("wrote", a.out)
