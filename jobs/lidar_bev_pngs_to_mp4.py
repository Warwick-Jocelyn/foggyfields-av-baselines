"""Stitch the per-scan LiDAR PNGs that `ns-render dataset --rendered-output-names lidar gt-lidar`
writes into per-split MP4 videos.

Layout the renderer produces (per `nerfstudio/scripts/render.py:1153-1170`):

    RENDER_DIR/<split>/lidar/lidar_<N>.png
    RENDER_DIR/<split>/lidar/gt-lidar_<N>.png

Output:
    OUT_DIR/<split>/lidar_pred.mp4         predicted point cloud per scan
    OUT_DIR/<split>/lidar_gt.mp4           ground-truth point cloud per scan
    OUT_DIR/<split>/lidar_compare.mp4      GT | pred side-by-side
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np


_SCAN_RE = re.compile(r"(?:gt-)?lidar_(\d+)\.png$")


def _scan_idx(p: Path) -> int:
    m = _SCAN_RE.search(p.name)
    return int(m.group(1)) if m else -1


def _encode(out: Path, frames_iter, fps: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out), mode="w")
    stream = None
    n = 0
    try:
        for img in frames_iter:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            img = np.ascontiguousarray(img, dtype=np.uint8)
            h, w = img.shape[:2]
            if h % 2: img = img[:-1]
            if w % 2: img = img[:, :-1]
            h, w = img.shape[:2]
            if stream is None:
                stream = container.add_stream("libx264", rate=fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            n += 1
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    split_dir = args.render_dir / args.split / "lidar"
    if not split_dir.exists():
        print(f"  [skip] no lidar dir at {split_dir}")
        return
    pred = sorted([p for p in split_dir.iterdir() if p.name.startswith("lidar_")], key=_scan_idx)
    gt   = sorted([p for p in split_dir.iterdir() if p.name.startswith("gt-lidar_")], key=_scan_idx)
    print(f"  found {len(pred)} pred / {len(gt)} gt PNGs in {split_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if pred:
        n = _encode(args.out_dir / "lidar_pred.mp4", (imageio.imread(p) for p in pred), args.fps)
        print(f"  wrote lidar_pred.mp4 ({n} frames)")
    if gt:
        n = _encode(args.out_dir / "lidar_gt.mp4", (imageio.imread(p) for p in gt), args.fps)
        print(f"  wrote lidar_gt.mp4 ({n} frames)")
    if pred and gt:
        nshared = min(len(pred), len(gt))

        def pair_gen():
            for g_, p_ in zip(gt[:nshared], pred[:nshared]):
                gi = imageio.imread(g_)
                pi = imageio.imread(p_)
                h = min(gi.shape[0], pi.shape[0])
                yield np.concatenate([gi[:h], pi[:h]], axis=1)

        n = _encode(args.out_dir / "lidar_compare.mp4", pair_gen(), args.fps)
        print(f"  wrote lidar_compare.mp4 ({n} frames, gt | pred)")


if __name__ == "__main__":
    main()
