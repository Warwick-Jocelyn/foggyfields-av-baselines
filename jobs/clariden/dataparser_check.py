"""CPU check: run the NVIDIA-AV-Fog dataparser on one NFF clip and print split/shape sanity."""
import argparse, numpy as np, torch
from pathlib import Path
from nerfstudio.data.dataparsers.nvidia_av_fog_dataparser import NvidiaAvFogDataParserConfig
ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--sequence", required=True)
ap.add_argument("--splits-json", default="auto"); ap.add_argument("--lidar-points-frame", default="ego"); ap.add_argument("--use-valid-masks", action="store_true"); a = ap.parse_args()
cfg = NvidiaAvFogDataParserConfig(data=a.data, sequence=a.sequence, splits_json=a.splits_json, train_split_fraction=0.5, lidar_points_frame=a.lidar_points_frame, use_valid_masks=a.use_valid_masks)
out = {s: cfg.setup().get_dataparser_outputs(split=s) for s in ("train", "val")}
for s, o in out.items():
    ts = o.cameras.metadata["timesteps"].squeeze(-1).unique().tolist()
    print(f"{s}: cams={len(o.cameras)} imgs={len(o.image_filenames)} lidars={len(o.metadata['lidars'])} "
          f"timesteps[{len(ts)}] first={ts[:5]} last={ts[-3:]} masks={'yes' if o.mask_filenames else 'NO'}")
tr = set(out['train'].cameras.metadata['timesteps'].squeeze(-1).tolist()); va = set(out['val'].cameras.metadata['timesteps'].squeeze(-1).tolist())
print("train∩val timesteps:", len(tr & va), "| train even-only:", all(t % 2 == 0 for t in tr), "| val odd-only:", all(t % 2 == 1 for t in va))
o = out["train"]; print("cam res (unique w,h):", sorted(set(zip(o.cameras.width.squeeze(-1).tolist(), o.cameras.height.squeeze(-1).tolist()))))
print("c2w[0]:\n", o.cameras.camera_to_worlds[0].numpy().round(3)); print("missing image files:", sum(not p.exists() for p in o.image_filenames))
pc = o.metadata["point_clouds"][0]; print("lidar pc0 shape", tuple(pc.shape), "xyz range", pc[:, :3].min(0).values.tolist(), pc[:, :3].max(0).values.tolist(), "t-offset range", pc[:, 4].min().item(), pc[:, 4].max().item())
print("num trajectories:", len(o.metadata.get("trajectories", [])))
# ground-plane sanity: lidar points in WORLD frame should have z ≈ ego ground (~0) near the car; -1.9 m means the sweep frame is wrong
l2w = o.metadata["lidars"].lidar_to_worlds[0]; pw = (pc[:, :3] @ l2w[:3, :3].T + l2w[:3, 3]); near = pw[(pc[:, :2].norm(dim=1) > 5) & (pc[:, :2].norm(dim=1) < 25)]
import numpy as np; hist, edges = np.histogram(near[:, 2].numpy(), bins=np.arange(-4, 2.01, 0.25)); print(f"lidar_points_frame={a.lidar_points_frame}: world-z mode of 5-25 m points = {edges[hist.argmax()]:+.2f} m (expect ≈ -0.1 if frame is right, ≈ -2.0 if wrong)")
print("DATAPARSER CHECK OK")
