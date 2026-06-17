"""Project trained SplatAD `DynamicActors` cuboids onto already-rendered RGB PNGs.

Reads a SplatAD `config.yml` (so it can load the optimized actor poses from the
checkpoint), iterates the same (split, camera_idx) pairs that `ns-render dataset
--pose-source train+val` writes, and for each frame:

  1. asks `pipeline.model.dynamic_actors.get_boxes2world(time)` for the per-actor
     box->world pose (post-optimization),
  2. builds the 8 cuboid corners from `actor_sizes` (wlh) in the actor-local frame,
  3. transforms world -> camera (OpenGL convention: cam looks down -Z),
  4. projects with the camera's pinhole intrinsics,
  5. clips against the image rect and draws the 12 cuboid edges on a *copy* of the
     already-rendered model RGB PNG.

Output layout mirrors the existing `ns-render dataset` tree:

    <render_dir>/<split>/rgb_bbox/<cam>/<frame>.png

Usage:
    python draw_bbox_overlay.py <config.yml> <render_dir> [--splits train val]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


# 12 edges of a cuboid given the 8 corners ordered as:
#   0:(-x,-y,-z) 1:(+x,-y,-z) 2:(+x,+y,-z) 3:(-x,+y,-z)
#   4:(-x,-y,+z) 5:(+x,-y,+z) 6:(+x,+y,+z) 7:(-x,+y,+z)
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
)
# Highlight the heading face. PandaSet stores `yaw` as rotation about Z and the
# cuboid local axes are (x=width, y=length, z=height). The vehicle heading in
# PandaSet's convention is along the local +y axis, so the +y face is the front.
# Edges that bound the +y face in our 0..7 corner ordering:
#   bottom +y face: 2-3; top +y face: 6-7; verticals: 2-6, 3-7
FRONT_FACE_IDX = (2, 6, 10, 11)  # indices into EDGES list for the +y face

# DynamicActors doesn't keep per-actor class labels at inference time, so use a
# single colour for the cuboid body and a contrasting colour for the heading
# face (+y face in PandaSet cuboid convention).
DEFAULT_COLOR = (60, 220, 60)      # green body
FRONT_COLOR = (40, 80, 240)        # red-ish heading face


def project_corners(corners_world: np.ndarray, c2w_4x4: np.ndarray,
                    fx: float, fy: float, cx: float, cy: float,
                    width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points to pixel coords.

    Empirically (see bbox_projection_probe.py 4-variant sweep), the
    `camera_to_worlds` matrix stored in nerfstudio AD-style cameras behaves
    OpenCV-style: cam +Z is forward, +Y is down, so a point in front of the
    camera has Z > 0 in the cam frame and projects with the standard pinhole
    formula `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`. (The OPENCV_TO_NERFSTUDIO
    multiply in pandaset_dataparser is *not* observed in the c2w that ends up
    on `cameras.camera_to_worlds`.)

    Returns (uv_pixels Nx2 float, in_front_mask N bool).
    """
    pts_h = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=-1)
    w2c = np.linalg.inv(c2w_4x4)
    cam = pts_h @ w2c.T  # Nx4
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    in_front = Z > 0.05
    safe_Z = np.where(Z < 1e-6, 1e-6, Z)
    u = fx * X / safe_Z + cx
    v = fy * Y / safe_Z + cy
    return np.stack([u, v], axis=-1), in_front


def draw_cuboid(img: np.ndarray, uv: np.ndarray, in_front: np.ndarray,
                color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw the 12 edges of a cuboid on `img` in-place.

    Only draws an edge when *both* endpoints are in front of the camera and
    have finite uv coordinates — otherwise the line would cross the camera
    plane and one endpoint's pixel coord blows up to ±inf, which cv2.line
    rejects. (For demo purposes the lost edges are negligible — at worst the
    near vertex of a box that's clipping the image bottom isn't drawn.)
    """
    H, W = img.shape[:2]
    for ei, (a, b) in enumerate(EDGES):
        if not (in_front[a] and in_front[b]):
            continue
        pa, pb = uv[a], uv[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        # Skip lines far outside the image rect; cv2.line handles modest overshoot fine.
        if (max(pa[0], pb[0]) < -2 * W or min(pa[0], pb[0]) > 3 * W or
            max(pa[1], pb[1]) < -2 * H or min(pa[1], pb[1]) > 3 * H):
            continue
        c = FRONT_COLOR if ei in FRONT_FACE_IDX else color
        cv2.line(img,
                 (int(round(float(pa[0]))), int(round(float(pa[1])))),
                 (int(round(float(pb[0]))), int(round(float(pb[1])))),
                 c, thickness, lineType=cv2.LINE_AA)


def overlay_split(pipeline, dataparser_outputs, cameras, split_name: str,
                  render_dir: Path, rgb_subdir: str = "rgb") -> int:
    """For one split, walk every camera and overlay the cuboids onto the model
    RGB PNG that `ns-render dataset` already wrote.

    Returns the number of frames successfully written.
    """
    device = pipeline.device
    actors = pipeline.model.dynamic_actors
    dims = actors.actor_sizes.detach().cpu().numpy()  # (n_actors, 3) wlh order
    n_actors = dims.shape[0]
    print(f"  [{split_name}] n_actors={n_actors}")

    image_filenames = dataparser_outputs.image_filenames
    images_root = Path(os.path.commonpath([str(p) for p in image_filenames]))

    rgb_root = render_dir / split_name / rgb_subdir
    out_root = render_dir / split_name / "rgb_bbox"

    if not rgb_root.exists():
        print(f"  [{split_name}] rgb dir missing: {rgb_root}, skip")
        return 0

    n_done = 0
    for cam_idx in range(len(cameras)):
        # PNG path mirrors ns-render dataset: <render_dir>/<split>/rgb/<image_name>.png
        image_name = Path(str(image_filenames[cam_idx])).with_suffix("").relative_to(images_root)
        in_png = rgb_root / image_name.with_suffix(".png")
        if not in_png.exists():
            # Fallback for `.jpg` original filenames stored in dataparser
            alt = rgb_root / image_name.with_suffix(".jpg")
            if alt.exists():
                in_png = alt
            else:
                continue

        img = cv2.imread(str(in_png), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        # Per-camera intrinsics + extrinsics (OpenGL convention)
        c2w_3x4 = cameras.camera_to_worlds[cam_idx].detach().cpu().numpy()  # (3,4)
        c2w = np.eye(4); c2w[:3] = c2w_3x4
        fx = float(cameras.fx[cam_idx].item())
        fy = float(cameras.fy[cam_idx].item())
        cx = float(cameras.cx[cam_idx].item())
        cy = float(cameras.cy[cam_idx].item())

        time = cameras.times[cam_idx].view(1, 1).to(device)
        with torch.no_grad():
            boxes2world, exists_at_time = actors.get_boxes2world(time, flatten=False)
        # boxes2world: (1, n_actors, 4, 4); exists_at_time: (1, n_actors) bool
        b2w = boxes2world[0].detach().cpu().numpy()
        exists = exists_at_time[0].detach().cpu().numpy().astype(bool)

        n_drawn = 0
        for ai in range(n_actors):
            if not exists[ai]:
                continue
            dx, dy, dz = dims[ai]  # wlh (x=width, y=length, z=height)
            hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
            corners_local = np.array([
                [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
                [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
            ])  # 8x3
            corners_h = np.concatenate([corners_local, np.ones((8, 1))], axis=-1)
            corners_world = (corners_h @ b2w[ai].T)[:, :3]  # 8x3

            uv, in_front = project_corners(corners_world, c2w, fx, fy, cx, cy, W, H)
            if not in_front.any():
                continue
            draw_cuboid(img, uv, in_front, DEFAULT_COLOR)
            n_drawn += 1

        out_png = out_root / image_name.with_suffix(".png")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), img)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  [{split_name}] {n_done}/{len(cameras)} (last: {image_name}, drew {n_drawn} boxes)")
    print(f"  [{split_name}] DONE: {n_done} frames -> {out_root}")
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Path to config.yml from a trained SplatAD run")
    ap.add_argument("render_dir", type=Path,
                    help="Directory ns-render dataset wrote (contains train/ and/or val/ sub-dirs)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")

    # Re-use the same dataparser the training used so cuboid time/pose frames line up.
    dataparser = pipeline.datamanager.dataparser

    for split in args.splits:
        # Map render-dir split name to dataparser split name
        ds_split = {"train": "train", "val": "val", "test": "val"}.get(split, split)
        dpo = dataparser.get_dataparser_outputs(split=ds_split)
        cameras = dpo.cameras.to(pipeline.device)
        overlay_split(pipeline, dpo, cameras, split, args.render_dir)


if __name__ == "__main__":
    main()
