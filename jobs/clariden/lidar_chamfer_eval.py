"""Native-LiDAR evaluation + inference timing for a trained SplatAD run.
For every held-out sweep: render through SplatAD's lidar rasterizer, build the predicted point cloud exactly as the
model's own eval does (filter_lidar_pred_and_gt, ray-drop applied), then
  CD   = 0.5 * (mean_p min_g |p-g| + mean_g min_p |g-p|)   [metres, symmetric nearest-neighbour; the June-2026 definition]
  CD2  = sum of both directions of squared NN distances / |gt|  [m^2, nerfstudio/SplatAD's chamfer_distance(…, normalize_with_target=True)]
  RCD  = CD / mean GT range
plus depth_median_l2 on returned rays, ray-drop accuracy, and wall-clock inference time per camera frame / per sweep."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
from nerfstudio.utils.eval_utils import eval_setup

ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--n-cameras", type=int, default=70); a = ap.parse_args()
config, pipeline, ckpt, step = eval_setup(a.config, test_mode="val")
model, dm = pipeline.model, pipeline.datamanager
model.eval()

# ---- camera inference timing (first n held-out frames, all cameras interleaved) ----
cam_times, pixels = [], 0
with torch.no_grad():
    for i, (camera, batch) in enumerate(dm.fixed_indices_eval_dataloader):
        if i >= a.n_cameras: break
        torch.cuda.synchronize(); t0 = time.time(); out = model.get_outputs_for_camera(camera); torch.cuda.synchronize()
        cam_times.append(time.time() - t0); pixels += out["rgb"].shape[0] * out["rgb"].shape[1]
cam_ms = 1000 * float(np.mean(cam_times[3:])) if len(cam_times) > 3 else None  # skip warm-up

# ---- lidar: timing + chamfer ----
rows, lid_times = [], []
with torch.no_grad():
    for lidar, batch in dm.fixed_indices_eval_lidar_dataloader:
        torch.cuda.synchronize(); t0 = time.time(); outputs = model.get_lidar_outputs(lidar); torch.cuda.synchronize(); lid_times.append(time.time() - t0)
        pred, gt = model.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)
        P = pred["point_cloud"].detach().float().cpu().numpy(); G = gt["point_cloud"].detach().float().cpu().numpy()
        Pm = pred["median_point_cloud"].detach().float().cpu().numpy()
        if len(P) == 0 or len(G) == 0: continue
        tg, tp = cKDTree(G), cKDTree(P)
        d_pg = tg.query(P, k=1, workers=16)[0]; d_gp = tp.query(G, k=1, workers=16)[0]
        tpm = cKDTree(Pm); d_pmg = tg.query(Pm, k=1, workers=16)[0]; d_gpm = tpm.query(G, k=1, workers=16)[0]
        gt_range = float(np.linalg.norm(G, axis=1).mean())
        dd = (pred["depth"] - gt["depth"]).abs()
        rows.append(dict(
            n_pred=int(len(P)), n_gt=int(len(G)), gt_range_m=gt_range,
            cd_m=0.5 * (d_pg.mean() + d_gp.mean()), cd2_m2=float((d_pg ** 2).sum() + (d_gp ** 2).sum()) / len(G),
            cd_median_pc_m=0.5 * (d_pmg.mean() + d_gpm.mean()),
            depth_median_l2_m=float(dd.median()), depth_mean_l2_m=float(dd.mean()),
            ray_drop_acc=float((((pred["ray_drop"].sigmoid() > 0.5) == gt["ray_drop"]) * gt["valid"]).sum() / gt["valid"].sum()),
        ))
agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("n_pred", "n_gt")}
res = dict(config=str(a.config), step=int(step), n_frames=len(rows), n_pred=int(sum(r["n_pred"] for r in rows)), n_gt=int(sum(r["n_gt"] for r in rows)),
           **agg, rcd=agg["cd_m"] / agg["gt_range_m"],
           infer_ms_per_camera_frame=cam_ms, infer_cameras_timed=len(cam_times), infer_megapixels_per_s=(pixels / sum(cam_times) / 1e6) if cam_times else None,
           infer_ms_per_lidar_sweep=1000 * float(np.mean(lid_times[2:])) if len(lid_times) > 2 else None,
           definitions="cd_m = symmetric mean NN distance (m); cd2_m2 = nerfstudio chamfer (sum sq NN both ways / |gt|); rcd = cd_m / mean GT range; "
                       "cd_median_pc_m uses the model's median-depth point cloud; inference = wall-clock on one GH200, cuda-synchronized, warm-up excluded",
           per_frame=rows)
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=2))
