"""Build per-camera + 3x3 7-cam panorama MP4s from `ns-render dataset` output for
the NVIDIA_AV_Fog dataset.

This is a fork of `pngs_to_mp4.py` specialized for the 7-camera NVIDIA_AV_Fog
layout. Differences vs the PandaSet 3x2 version:

  - PANORAMA_LAYOUT has 3 rows, with empty slots in the middle row's outer cells
    (only the front_tele_30fov sits between the wide cams' bottom edge and the
    rear cams). Empty slots are filled with black tiles sized to match the rest
    of the row.
  - Per-row tile size unification: each row's tiles are resized to a common
    (h, w) by central-crop (matches existing convention in pngs_to_mp4.py) but
    rows are then padded to the same total width so the mosaic is rectangular.

Layout (matches the user's chosen 3x3 with teles on middle row):

    cross_left_120 | front_wide_120 | cross_right_120
    (empty)        | front_tele_30  | (empty)
    rear_left_70   | rear_tele_30   | rear_right_70
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import av
import imageio.v2 as imageio
import numpy as np


# Use None to indicate empty (black) tile.
PANORAMA_LAYOUT: tuple[tuple[Optional[str], ...], ...] = (
    ("camera_cross_left_120fov",  "camera_front_wide_120fov", "camera_cross_right_120fov"),
    (None,                         "camera_front_tele_30fov", None),
    ("camera_rear_left_70fov",    "camera_rear_tele_30fov",   "camera_rear_right_70fov"),
)


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
            if h % 2:
                img = img[:-1]
            if w % 2:
                img = img[:, :-1]
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


def _crop_or_pad_center(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih > h:
        top = (ih - h) // 2
        img = img[top:top + h]
    elif ih < h:
        pad = h - ih
        img = np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)))
    iw = img.shape[1]
    if iw > w:
        left = (iw - w) // 2
        img = img[:, left:left + w]
    elif iw < w:
        pad = w - iw
        img = np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)))
    return img


def write_panorama_mp4(out: Path, split_dir: Path, kind: str, fps: int) -> None:
    """3x3 mosaic for the NVIDIA_AV_Fog 7-camera setup. Empty cells are black tiles."""
    kind_dir = split_dir / kind
    if not kind_dir.exists():
        return
    cam_files: dict[str, list[Path]] = {}
    for row in PANORAMA_LAYOUT:
        for cam in row:
            if cam is None or cam in cam_files:
                continue
            files = sorted((kind_dir / cam).glob("*.png"))
            if not files:
                print(f"  [skip panorama {kind}] no frames for {cam}")
                return
            cam_files[cam] = files
    n = min(len(v) for v in cam_files.values())
    if n == 0:
        return

    # Probe shapes once so each row's tile size is consistent.
    sample = {cam: imageio.imread(files[0]).shape[:2] for cam, files in cam_files.items()}
    row_dims: list[tuple[int, int]] = []
    for row in PANORAMA_LAYOUT:
        real = [c for c in row if c is not None]
        rh = min(sample[c][0] for c in real)
        rw = min(sample[c][1] for c in real)
        row_dims.append((rh, rw))

    panel_h = [rh for rh, _ in row_dims]
    panel_w_per_col = [max(rw for rh, rw in row_dims)] * 3
    # Use single uniform tile width across rows = max across rows' rw; this keeps
    # all 3 columns aligned. Tiles will be center-cropped/padded to (rh, panel_w).
    panel_w = max(rw for _, rw in row_dims)

    def gen():
        for i in range(n):
            rows_out = []
            for r, row in enumerate(PANORAMA_LAYOUT):
                rh = panel_h[r]
                tiles = []
                for cam in row:
                    if cam is None:
                        tiles.append(np.zeros((rh, panel_w, 3), dtype=np.uint8))
                    else:
                        img = imageio.imread(cam_files[cam][i])
                        if img.ndim == 2:
                            img = np.stack([img] * 3, axis=-1)
                        tiles.append(_crop_or_pad_center(img, rh, panel_w))
                rows_out.append(np.concatenate(tiles, axis=1))
            yield np.concatenate(rows_out, axis=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = _encode(out, gen(), fps)
    print(f"  wrote {out.name} ({written} frames, 3x3 panorama, kind={kind})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir", type=Path, help="Directory ns-render wrote to (contains a split sub-dir)")
    ap.add_argument("out_dir", type=Path, help="Directory to write MP4 files")
    ap.add_argument("--split", default="val")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--panorama-only", action="store_true",
                    help="Skip per-camera files; only write 3x3 panorama MP4s.")
    args = ap.parse_args()

    split_dir = args.render_dir / args.split
    if not split_dir.exists():
        split_dir = args.render_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.panorama_only:
        rgb_root = split_dir / "rgb"
        if rgb_root.exists():
            cams = sorted(p.name for p in rgb_root.iterdir() if p.is_dir())
            for cam in cams:
                print(f"-- {cam} --")
                rgb = sorted((rgb_root / cam).glob("*.png"))
                gt = sorted((split_dir / "gt-rgb" / cam).glob("*.png"))
                depth = sorted((split_dir / "depth" / cam).glob("*.png"))
                write_mp4(args.out_dir / f"{cam}_rgb.mp4", rgb, args.fps)
                write_mp4(args.out_dir / f"{cam}_gt.mp4", gt, args.fps)
                write_mp4(args.out_dir / f"{cam}_depth.mp4", depth, args.fps)

    print("-- panorama (3x3) --")
    for kind in ("gt-rgb", "rgb", "depth", "rgb_bbox"):
        write_panorama_mp4(args.out_dir / f"panorama_{kind.replace('-', '_')}.mp4",
                           split_dir, kind, args.fps)


if __name__ == "__main__":
    main()
