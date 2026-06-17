"""GT-only LiDAR BEV with optimised-actor 3D bboxes overlaid: load each scan's
points, transform to world, query `DynamicActors.get_boxes2world(time)` for the
trained box poses, and render points + box edges in one plotly Scatter3d figure.

Usage: python gt_lidar_bev_render.py <config.yml> <out_root>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import imageio.v2 as imageio
import av
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent / ".." / "neurad-studio"))
from nerfstudio.utils.eval_utils import eval_setup


CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
HEADING_FACE_EDGES = (2, 6, 10, 11)  # +y face per PandaSet cuboid convention


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


def plot_lidar_with_bboxes(points: np.ndarray, boxes_corners_world: np.ndarray,
                            out_path: Path, cmin=-6.0, cmax=5.0,
                            width=1920, height=1080, ranges=(100, 200, 10)):
    """Plot world-frame lidar points + 3D bbox edges in one Scatter3d figure.

    points: (N, 3) world coords
    boxes_corners_world: (K, 8, 3) world coords, one cuboid per row
    """
    traces = [go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
        marker=dict(size=1.0, color=points[:, 2], colorscale="Viridis",
                    opacity=0.8, cmin=cmin, cmax=cmax),
        showlegend=False,
    )]
    if boxes_corners_world.size:
        # Body edges (all except heading-face edges)
        body_x, body_y, body_z = [], [], []
        head_x, head_y, head_z = [], [], []
        for corners in boxes_corners_world:
            for ei, (a, b) in enumerate(CUBOID_EDGES):
                xs, ys, zs = (body_x, body_y, body_z) if ei not in HEADING_FACE_EDGES else (head_x, head_y, head_z)
                xs.extend([float(corners[a, 0]), float(corners[b, 0]), None])
                ys.extend([float(corners[a, 1]), float(corners[b, 1]), None])
                zs.extend([float(corners[a, 2]), float(corners[b, 2]), None])
        if body_x:
            traces.append(go.Scatter3d(x=body_x, y=body_y, z=body_z, mode="lines",
                                       line=dict(color="lime", width=3), showlegend=False))
        if head_x:
            traces.append(go.Scatter3d(x=head_x, y=head_y, z=head_z, mode="lines",
                                       line=dict(color="red",  width=4), showlegend=False))

    x_range, y_range, z_range = ranges
    max_range = 2 * max(x_range, y_range, z_range)
    aspect_ratio = dict(x=x_range / max_range, y=y_range / max_range, z=z_range / max_range)
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.0, y=-0.07, z=0.02))
    layout = go.Layout(scene=dict(
        xaxis=dict(title="", range=[-x_range, x_range], showticklabels=False, ticks="", showline=False, showgrid=False),
        yaxis=dict(title="", range=[-y_range, y_range], showticklabels=False, ticks="", showline=False, showgrid=False),
        zaxis=dict(title="", range=[-z_range, z_range], showticklabels=False, ticks="", showline=False, showgrid=False),
        aspectmode="manual", aspectratio=aspect_ratio, camera=camera,
    ))
    fig = go.Figure(data=traces, layout=layout)
    fig.write_image(str(out_path), width=width, height=height, scale=1)


def get_actor_corners_world(actors, query_time_tensor, device) -> np.ndarray:
    """Return (K, 8, 3) world-frame corners for actors present at query_time.
    Centred-on-box convention: corners are ±dims/2 in actor local frame."""
    with torch.no_grad():
        b2w, exists = actors.get_boxes2world(query_time_tensor, flatten=False)
    b2w = b2w[0].detach().cpu().numpy()                      # (n_actors, 4, 4)
    exists = exists[0].detach().cpu().numpy().astype(bool)   # (n_actors,)
    dims = actors.actor_sizes.detach().cpu().numpy()         # (n_actors, 3) wlh
    visible_idx = np.where(exists)[0]
    if not len(visible_idx):
        return np.zeros((0, 8, 3))
    out = np.zeros((len(visible_idx), 8, 3))
    for k, ai in enumerate(visible_idx):
        dx, dy, dz = dims[ai]
        hx, hy, hz = dx / 2, dy / 2, dz / 2
        corners_local = np.array([
            [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
            [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
        ])
        corners_h = np.concatenate([corners_local, np.ones((8, 1))], axis=-1)
        corners_world = (corners_h @ b2w[ai].T)[:, :3]
        out[k] = corners_world
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("out_root", type=Path, help="paper_<scene>/ root; writes renders/<split>/lidar/ and videos/<split>/lidar_gt.mp4")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    device = pipeline.device
    actors = pipeline.model.dynamic_actors

    for split in ("train", "val"):
        ds = pipeline.datamanager.train_lidar_dataset if split == "train" else pipeline.datamanager.eval_lidar_dataset
        lidars = ds.lidars  # Lidars object
        n = len(ds)
        png_root = args.out_root / "renders" / split / "lidar"
        png_root.mkdir(parents=True, exist_ok=True)
        print(f"[{split}] writing {n} scans to {png_root}")
        for i in range(n):
            item = ds[i]
            pts_local = item["lidar"]
            if isinstance(pts_local, torch.Tensor):
                pts_local = pts_local.detach().cpu().numpy()
            pts_local = pts_local[:, :3]
            l2w = lidars.lidar_to_worlds[i].detach().cpu().numpy()
            l2w_4x4 = np.eye(4); l2w_4x4[:3] = l2w
            pts_h = np.concatenate([pts_local, np.ones((pts_local.shape[0], 1))], axis=-1)
            pts_world = (l2w_4x4 @ pts_h.T).T[:, :3]

            time = lidars.times[i].view(1, 1).to(device)
            corners_world = get_actor_corners_world(actors, time, device)

            out_png = png_root / f"gt-lidar_{i:04d}.png"
            try:
                plot_lidar_with_bboxes(pts_world, corners_world, out_png)
            except Exception as e:
                print(f"  [{split} {i}] plot failed: {e}")
                continue
            if (i + 1) % 20 == 0:
                print(f"  [{split}] {i+1}/{n} done ({len(corners_world)} boxes)")

        # Encode mp4
        pngs = sorted(png_root.glob("gt-lidar_*.png"))
        if pngs:
            out_mp4 = args.out_root / "videos" / split / "lidar_gt_bbox.mp4"
            nf = _encode(out_mp4, (imageio.imread(p) for p in pngs), args.fps)
            print(f"[{split}] wrote {out_mp4} ({nf} frames)")


if __name__ == "__main__":
    main()
