"""Diagnostic: render the same frame with 4 projection sign-variants for the
3D->2D math, on BOTH the model-rendered RGB and the GT image, so we can
visually tell which projection formula aligns the cuboids with the cars.

Also draws each *other* camera's optical centre as a coloured cross on the
chosen camera frame (a pose-chain sanity check independent of cuboids).

Output: a 4x2 mosaic PNG to <out_dir>/bbox_probe_<scene>_<cam>.png and a
text dump of intrinsics/extrinsics next to it.

Usage:
    python bbox_projection_probe.py <config.yml> <render_dir> <out_path> \
        --split train --camera front_camera --frame 30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def project(corners_world, c2w_4x4, fx, fy, cx, cy, u_sign, v_sign, z_forward_sign):
    """Project 3D world points to pixel coords with chosen sign options.

    z_forward_sign: +1 means cam-frame Z > 0 is 'in front of camera' (OpenCV);
                    -1 means cam-frame Z < 0 is 'in front'              (OpenGL/nerfstudio).
    """
    pts_h = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=-1)
    w2c = np.linalg.inv(c2w_4x4)
    cam = pts_h @ w2c.T
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    in_front = (Z * z_forward_sign) > 0.05
    safe = np.where(np.abs(Z) < 1e-6, np.sign(Z + 1e-9) * 1e-6, Z)
    u = u_sign * fx * X / safe + cx
    v = v_sign * fy * Y / safe + cy
    return np.stack([u, v], axis=-1), in_front


def draw_cuboid(img, uv, in_front, color, thickness=2):
    H, W = img.shape[:2]
    for a, b in EDGES:
        if not (in_front[a] or in_front[b]):
            continue
        pa, pb = uv[a], uv[b]
        if (max(pa[0], pb[0]) < -2 * W or min(pa[0], pb[0]) > 3 * W or
            max(pa[1], pb[1]) < -2 * H or min(pa[1], pb[1]) > 3 * H):
            continue
        cv2.line(img,
                 (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))),
                 color, thickness, lineType=cv2.LINE_AA)


def render_variant(rgb, b2w_per_actor, dims, exists, c2w_4x4, fx, fy, cx, cy,
                   u_sign, v_sign, z_forward_sign, label):
    img = rgb.copy()
    n_drawn = 0
    for ai in range(dims.shape[0]):
        if not exists[ai]:
            continue
        dx, dy, dz = dims[ai]
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
        corners_local = np.array([
            [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
            [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
        ])
        corners_h = np.concatenate([corners_local, np.ones((8, 1))], axis=-1)
        corners_world = (corners_h @ b2w_per_actor[ai].T)[:, :3]
        uv, in_front = project(corners_world, c2w_4x4, fx, fy, cx, cy, u_sign, v_sign, z_forward_sign)
        if not in_front.any():
            continue
        draw_cuboid(img, uv, in_front, (60, 220, 60))
        n_drawn += 1
    # banner
    cv2.rectangle(img, (0, 0), (640, 50), (0, 0, 0), -1)
    cv2.putText(img, f"{label}  ({n_drawn} boxes drawn)", (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def project_other_cam_origins(img, this_c2w, this_fx, this_fy, this_cx, this_cy,
                              all_c2ws, u_sign, v_sign, z_forward_sign,
                              colours):
    """Draw a coloured cross at the projected optical centre of each other camera."""
    for k, oc2w in enumerate(all_c2ws):
        origin_world = oc2w[:3, 3].reshape(1, 3)
        uv, in_front = project(origin_world, this_c2w, this_fx, this_fy, this_cx, this_cy,
                               u_sign, v_sign, z_forward_sign)
        if not in_front[0]:
            continue
        u, v = int(uv[0, 0]), int(uv[0, 1])
        cv2.drawMarker(img, (u, v), colours[k], cv2.MARKER_CROSS, 30, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("out_path", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--camera", default="front_camera")
    ap.add_argument("--frame", type=int, default=30, help="Frame index within the camera's frames in the chosen split")
    args = ap.parse_args()

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    dataparser = pipeline.datamanager.dataparser
    ds_split = "train" if args.split == "train" else "val"
    dpo = dataparser.get_dataparser_outputs(split=ds_split)
    cameras = dpo.cameras.to(pipeline.device)

    image_filenames = dpo.image_filenames
    images_root = Path(os.path.commonpath([str(p) for p in image_filenames]))
    # find the camera idx that matches the chosen camera + frame
    target = None
    seen = 0
    for idx in range(len(cameras)):
        rel = Path(str(image_filenames[idx])).with_suffix("").relative_to(images_root)
        if str(rel.parent) == args.camera or str(rel.parent).endswith(args.camera):
            if seen == args.frame:
                target = idx
                break
            seen += 1
    if target is None:
        raise RuntimeError(f"Could not find camera {args.camera} frame {args.frame}")
    cam_idx = target
    rel = Path(str(image_filenames[cam_idx])).with_suffix("").relative_to(images_root)
    rgb_png = args.render_dir / args.split / "rgb" / rel.with_suffix(".png")
    gt_png  = args.render_dir / args.split / "gt-rgb" / rel.with_suffix(".png")
    rgb = cv2.imread(str(rgb_png))
    gt  = cv2.imread(str(gt_png))
    H, W = rgb.shape[:2]
    print(f"target cam_idx={cam_idx} image={rel}, rgb shape={rgb.shape}")

    fx = float(cameras.fx[cam_idx].item())
    fy = float(cameras.fy[cam_idx].item())
    cx = float(cameras.cx[cam_idx].item())
    cy = float(cameras.cy[cam_idx].item())
    print(f"intrinsics  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"camera H={int(cameras.height[cam_idx].item())} W={int(cameras.width[cam_idx].item())}")

    c2w_3x4 = cameras.camera_to_worlds[cam_idx].detach().cpu().numpy()
    c2w = np.eye(4); c2w[:3] = c2w_3x4
    print("c2w 3x4 =\n", c2w_3x4)

    time = cameras.times[cam_idx].view(1, 1).to(pipeline.device)
    actors = pipeline.model.dynamic_actors
    with torch.no_grad():
        b2w, exists = actors.get_boxes2world(time, flatten=False)
    b2w = b2w[0].detach().cpu().numpy()
    exists = exists[0].detach().cpu().numpy().astype(bool)
    dims = actors.actor_sizes.detach().cpu().numpy()
    print(f"n_actors total={dims.shape[0]} present={exists.sum()}")
    # print first 3 visible actor world centres
    visible_idx = np.where(exists)[0][:5]
    for ai in visible_idx:
        c = b2w[ai, :3, 3]
        print(f"  actor {ai}: world center = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})  dims={dims[ai]}")

    # also gather all cam optical centres in this split for the "other cam origins" overlay
    all_c2ws = []
    cam_names_seen = set()
    for idx in range(len(cameras)):
        rel_i = Path(str(image_filenames[idx])).with_suffix("").relative_to(images_root)
        cname = rel_i.parent.name
        if cname in cam_names_seen:
            continue
        cam_names_seen.add(cname)
        c = np.eye(4); c[:3] = cameras.camera_to_worlds[idx].detach().cpu().numpy()
        all_c2ws.append(c)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]

    variants = [
        # u_sign,  v_sign, z_forward_sign, label
        (-1, -1, -1, "A: u=-fxX/Z, v=-fyY/Z, Z<0 forward (nerfstudio derivation)"),
        (-1, +1, -1, "B: u=-fxX/Z, v=+fyY/Z, Z<0 forward (Y flipped)"),
        (+1, +1, +1, "D: u=+fxX/Z, v=+fyY/Z, Z>0 forward (OpenCV interpretation)"),
        (+1, -1, +1, "E: u=+fxX/Z, v=-fyY/Z, Z>0 forward (OpenCV with Y flipped)"),
    ]
    rgb_panels = []
    gt_panels = []
    for u_s, v_s, zf, label in variants:
        ri = render_variant(rgb, b2w, dims, exists, c2w, fx, fy, cx, cy, u_s, v_s, zf, "RGB " + label)
        gi = render_variant(gt,  b2w, dims, exists, c2w, fx, fy, cx, cy, u_s, v_s, zf, "GT  " + label)
        # add coloured crosses for other-cam origins
        project_other_cam_origins(ri, c2w, fx, fy, cx, cy, all_c2ws, u_s, v_s, zf, colours)
        project_other_cam_origins(gi, c2w, fx, fy, cx, cy, all_c2ws, u_s, v_s, zf, colours)
        rgb_panels.append(ri)
        gt_panels.append(gi)

    # 4 rows x 2 cols mosaic (each row: rgb | gt). Downscale 2x for size.
    rows = []
    for ri, gi in zip(rgb_panels, gt_panels):
        row = np.concatenate([ri, gi], axis=1)
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2))
        rows.append(row)
    mosaic = np.concatenate(rows, axis=0)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_path), mosaic)
    print(f"wrote {args.out_path}  shape={mosaic.shape}")


if __name__ == "__main__":
    main()
