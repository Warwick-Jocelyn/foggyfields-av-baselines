"""Dump SplatAD's native LiDAR render for every held-out sweep: predicted point cloud (ray-drop applied),
GT point cloud, and the GT/pred range images (azimuth x elevation raster) -> one npz per frame."""
import argparse, numpy as np, torch
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
config, pipeline, ckpt, step = eval_setup(a.config, test_mode="val"); model, dm = pipeline.model, pipeline.datamanager; model.eval()
a.out.mkdir(parents=True, exist_ok=True); n = 0
with torch.no_grad():
    for lidar, batch in dm.fixed_indices_eval_lidar_dataloader:
        outputs = model.get_lidar_outputs(lidar)
        pred, gt = model.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)
        rp = batch["raster_pts"]  # [H, W, 5]: azimuth, elevation, depth, time, intensity
        t = int(lidar.metadata["timesteps"].flatten()[0]) if "timesteps" in lidar.metadata else n
        shp = tuple(rp.shape)
        if n == 0: print("raster_pts shape", shp, "depth", tuple(outputs["depth"].shape), "median_depth", tuple(outputs["median_depth"].shape), "ray_drop", tuple(outputs["ray_drop_logits"].shape))
        fl = lambda x: x.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
        np.savez_compressed(a.out / f"frame_{t:04d}.npz",
            pred_xyz=pred["point_cloud"].cpu().numpy().astype(np.float32), gt_xyz=gt["point_cloud"].cpu().numpy().astype(np.float32),
            pred_median_xyz=pred["median_point_cloud"].cpu().numpy().astype(np.float32),
            raster_shape=np.array(shp), azimuth=fl(rp[..., 0]), elevation=fl(rp[..., 1]), gt_range=fl(rp[..., 2]), gt_intensity=fl(rp[..., 4]),
            pred_range=fl(outputs["depth"]), pred_median_range=fl(outputs["median_depth"]), pred_intensity=fl(outputs["intensity"]), pred_ray_drop_logit=fl(outputs["ray_drop_logits"]),
            lidar_to_world=lidar.lidar_to_worlds[0].cpu().numpy().astype(np.float32))
        n += 1
print(f"dumped {n} frames -> {a.out}")
