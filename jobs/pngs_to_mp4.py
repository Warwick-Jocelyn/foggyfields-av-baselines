"""Build per-camera MP4 videos from a `ns-render dataset` PNG output tree.

Uses pyav directly (imageio's PyAV plugin chokes on RGB→x264 with our images).

Layout it expects:
    RENDER_DIR/<split>/<kind>/<cam>/<idx>.png
        kind ∈ {rgb, gt-rgb, depth, rgb_bbox (optional)}

Produces, for each camera:
  - {cam}_rgb.mp4            model output
  - {cam}_gt.mp4             ground truth
  - {cam}_depth.mp4          depth colourised
  - {cam}_compare.mp4        gt | model | depth         (3-panel)
  - {cam}_rgb_bbox.mp4       model output with projected 3D cuboids overlaid
                             (only if rgb_bbox/ exists)
  - {cam}_compare_bbox.mp4   gt | model | rgb_bbox | depth   (4-panel)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np


def _encode(out: Path, frames_iter, fps: int) -> int:
    """Write H.264/yuv420p MP4 from an iterator of HxWx3 uint8 RGB arrays."""
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


def write_mp4(out: Path, frames: list[Path], fps: int) -> None:
    if not frames:
        print(f"  [skip] no frames -> {out.name}")
        return
    n = _encode(out, (imageio.imread(f) for f in frames), fps)
    print(f"  wrote {out.name} ({n} frames)")


def write_compare_mp4(out: Path, gt: list[Path], rgb: list[Path], depth: list[Path], fps: int) -> None:
    n_in = min(len(gt), len(rgb), len(depth))
    if n_in == 0:
        print(f"  [skip] missing tracks -> {out.name}")
        return

    def gen():
        for g_, r_, d_ in zip(gt[:n_in], rgb[:n_in], depth[:n_in]):
            gi = imageio.imread(g_)
            ri = imageio.imread(r_)
            di = imageio.imread(d_)
            h = min(gi.shape[0], ri.shape[0], di.shape[0])
            yield np.concatenate([gi[:h], ri[:h], di[:h]], axis=1)

    n = _encode(out, gen(), fps)
    print(f"  wrote {out.name} ({n} frames, triptych)")


def write_compare_bbox_mp4(out: Path, gt: list[Path], rgb: list[Path],
                           rgb_bbox: list[Path], depth: list[Path], fps: int) -> None:
    n_in = min(len(gt), len(rgb), len(rgb_bbox), len(depth))
    if n_in == 0:
        print(f"  [skip] missing tracks -> {out.name}")
        return

    def gen():
        for g_, r_, b_, d_ in zip(gt[:n_in], rgb[:n_in], rgb_bbox[:n_in], depth[:n_in]):
            gi = imageio.imread(g_)
            ri = imageio.imread(r_)
            bi = imageio.imread(b_)
            di = imageio.imread(d_)
            h = min(gi.shape[0], ri.shape[0], bi.shape[0], di.shape[0])
            yield np.concatenate([gi[:h], ri[:h], bi[:h], di[:h]], axis=1)

    n = _encode(out, gen(), fps)
    print(f"  wrote {out.name} ({n} frames, 4-panel)")


PANORAMA_LAYOUT = (
    ("front_left_camera", "front_camera", "front_right_camera"),
    ("left_camera",       "back_camera",  "right_camera"),
)


def write_panorama_mp4(out: Path, split_dir: Path, kind: str, fps: int) -> None:
    """Write a 3x2 mosaic MP4 (front-left | front | front-right on top, left |
    back | right on bottom) for a given rendered output kind (rgb, gt-rgb, depth,
    rgb_bbox)."""
    kind_dir = split_dir / kind
    if not kind_dir.exists():
        return
    cam_files = {}
    for row in PANORAMA_LAYOUT:
        for cam in row:
            files = sorted((kind_dir / cam).glob("*.png"))
            if not files:
                print(f"  [skip panorama {kind}] no frames for {cam}")
                return
            cam_files[cam] = files
    n = min(len(v) for v in cam_files.values())

    def gen():
        for i in range(n):
            rows = []
            row_h = None
            for row in PANORAMA_LAYOUT:
                tiles = [imageio.imread(cam_files[c][i]) for c in row]
                # unify each row to the same height (smallest in the row)
                rh = min(t.shape[0] for t in tiles)
                tiles = [t[:rh] for t in tiles]
                # unify each row to the same width per tile (smallest in the row),
                # cropped centrally so columns line up
                rw = min(t.shape[1] for t in tiles)
                tiles = [t[:, (t.shape[1] - rw) // 2 : (t.shape[1] - rw) // 2 + rw] for t in tiles]
                rows.append(np.concatenate(tiles, axis=1))
            # equalise width across rows by cropping to the smaller
            min_w = min(r.shape[1] for r in rows)
            rows = [r[:, :min_w] for r in rows]
            yield np.concatenate(rows, axis=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = _encode(out, gen(), fps)
    print(f"  wrote {out.name} ({written} frames, 3x2 panorama, kind={kind})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir", type=Path, help="Directory ns-render wrote to (contains a split sub-dir)")
    ap.add_argument("out_dir", type=Path, help="Directory to write MP4 files")
    ap.add_argument("--split", default="val")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--panorama-only", action="store_true",
                    help="Skip per-camera files; only write 3x2 panorama MP4s for each kind.")
    args = ap.parse_args()

    split_dir = args.render_dir / args.split
    if not split_dir.exists():
        split_dir = args.render_dir
    rgb_root = split_dir / "rgb"
    gt_root = split_dir / "gt-rgb"
    depth_root = split_dir / "depth"
    bbox_root = split_dir / "rgb_bbox"
    has_bbox = bbox_root.exists()
    cams = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.panorama_only:
        for cam in cams:
            print(f"-- {cam} --")
            rgb = sorted((rgb_root / cam).glob("*.png"))
            gt = sorted((gt_root / cam).glob("*.png"))
            depth = sorted((depth_root / cam).glob("*.png"))
            write_mp4(args.out_dir / f"{cam}_rgb.mp4", rgb, args.fps)
            write_mp4(args.out_dir / f"{cam}_gt.mp4", gt, args.fps)
            write_mp4(args.out_dir / f"{cam}_depth.mp4", depth, args.fps)
            write_compare_mp4(args.out_dir / f"{cam}_compare.mp4", gt, rgb, depth, args.fps)
            if has_bbox:
                rgb_bbox = sorted((bbox_root / cam).glob("*.png"))
                if rgb_bbox:
                    write_mp4(args.out_dir / f"{cam}_rgb_bbox.mp4", rgb_bbox, args.fps)
                    write_compare_bbox_mp4(args.out_dir / f"{cam}_compare_bbox.mp4",
                                           gt, rgb, rgb_bbox, depth, args.fps)

    # 3x2 panorama mosaic (front-left | front | front-right / left | back | right)
    print("-- panorama (3x2) --")
    for kind in ("gt-rgb", "rgb", "depth", "rgb_bbox"):
        write_panorama_mp4(args.out_dir / f"panorama_{kind.replace('-', '_')}.mp4",
                           split_dir, kind, args.fps)


if __name__ == "__main__":
    main()
