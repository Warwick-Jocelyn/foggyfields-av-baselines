#!/usr/bin/env python3
"""Native-LiDAR render + metrics for ANY neurad-studio model: SplatAD, NeuRAD, UniSim (plugin), ...

One command, one render pass per held-out sweep, two optional outputs:
  --output   metrics JSON (same keys/definitions as lidar_chamfer_eval.py, plus a few extras)
  --dump     one npz per sweep (same schema as lidar_dump.py, plus gt_did_return/pred_did_return)

    python jobs/lidar_dump_ns.py --config <run>/config.yml --output res.json [--dump DIR]

Two adapters, picked from the model's API (never from the method name):

  raster  (SplatAD: has get_lidar_outputs + filter_lidar_pred_and_gt)
          EXACTLY the lidar_chamfer_eval.py / lidar_dump.py path: the model's own filter builds
          pred/GT point clouds and the returned-ray depth pairs. Rays = SplatAD's tile bucket;
          'valid' = bucket cells with GT depth > 0 (padding excluded).

  ray     (NeuRAD / UniSim / any ADModel: get_outputs_for_lidar)
          EXACTLY what the model's own get_image_metrics_and_images does for lidar:
            GT did_return   = ray_bundle.metadata["did_return"] = |p| < valid_lidar_distance_threshold
                              (= DUMMY_DISTANCE_VALUE/2 = 1000 m) -> the 2000 m dummy non-return rays
                              are excluded from the GT cloud and from the depth metric;
            pred did_return = sigmoid(ray_drop_logit) <= 0.5   if the model's ray_drop_loss_mult > 0
                              (NeuRAD: loss.ray_drop_loss_mult; UniSim: ray_drop_loss_mult; the
                              ADPipeline zeroes it when add_missing_points=False, and we read the
                              live value after pipeline setup), else depth < non_return_lidar_distance;
            pred cloud      = outputs["points"][pred did_return]  (sensor frame, model's own transform);
            GT cloud        = batch["lidar"][did_return, :3]      (sensor frame);
            depth pairs     = outputs["depth"][did_return] vs batch["distance"][did_return].
          Median depth is not rendered by these models, so a tap on model.renderer_depth computes
          the nerfstudio-style median (first sample where the cumulative weight reaches 0.5) from the
          SAME weights/samples the expected depth uses (padded [N,S,1] for NeuRAD, packed
          nerfacc samples for UniSim).

Metrics are then computed by one shared function (identical arithmetic for both adapters):
  cd_m, cd2_m2, cd_median_pc_m, gt_range_m, rcd, depth_median_l2_m (= median |pred-gt|, metres,
  NOT squared -- the lidar_chamfer_eval.py definition), depth_mean_l2_m, ray_drop_acc
  (= mean over valid rays of [pred_drop == gt_drop], dummy non-return rays INCLUDED as the drop
  class), and extras: depth_median_sq_l2_m2 (neurad-studio's own 'depth_median_l2', squared),
  ray_drop_acc_returns / ray_drop_acc_nonreturns (per-class), frac_gt_nonreturn.

EVAL RAY SET (benchmark protocol). Every method is scored on the same rays: the real returns plus the
real non-returns read from lidar/range_image/ (current dataparser: half-rate beam column pairs merged).
So dataparser.use_range_image_raydrop is FORCED to True at eval, whatever the training config said
(UniSim trains with it off). `--set use_range_image_raydrop=False` still overrides it (regression runs
against pre-range-image JSONs). The JSON records eval_ray_set, n_return, n_nonreturn; if the flag is on
but a clip has no range images (n_nonreturn == 0) the set is labelled "...(range_image_missing)" so
lidar_collect.py refuses to mix it with the others.

SECONDARY, DROP-INDEPENDENT GEOMETRY METRIC (copy this definition for methods without a ray-drop head,
e.g. EmerNeRF). Per held-out sweep, sensor frame:
  R  = GT-returned rays: GT range < 1000 m  (= DUMMY_DISTANCE_VALUE/2; the 2000 m non-return rays are
       excluded; no prediction of ray drop is used anywhere)
  P  = { o_i + d_i * depth_pred_i : i in R }     one predicted point for EVERY ray in R, rendered depth
                                                 along the GT ray (o_i, d_i = the ray's origin/direction)
  G  = { GT point_i : i in R }
  cd_returns_m              = 0.5 * ( mean_{p in P} min_{g in G} |p-g|  +  mean_{g in G} min_{p in P} |g-p| )
  cd2_returns_m2            = ( sum_{p in P} min_g |p-g|^2  +  sum_{g in G} min_p |g-p|^2 ) / |G|
  depth_median_l2_returns_m = median_{i in R} | depth_pred_i - range_gt_i |    (metres, NOT squared)
  JSON value = plain mean of the per-sweep values over all held-out sweeps.
(SplatAD: P is built from its lidar raster cells with GT depth in (0, 1000) m, using the same
depth*direction + v*t formula as its own filter_lidar_pred_and_gt.)
"""
EVAL_RAY_SET_FULL = "returns+range_image_nonreturns_v2"
EVAL_RAY_SET_RETURNS = "returns_only"
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DEFAULT_PLUGIN_SRC = [HERE.parent / "code" / "_src" / "unisim"]  # importable without pip-installing


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None, help="metrics JSON")
    ap.add_argument("--dump", type=Path, default=None, help="npz dump directory (big: ~14 MB/sweep)")
    ap.add_argument("--n-cameras", type=int, default=70, help="camera frames to time (0 = skip)")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: stop after N sweeps")
    ap.add_argument("--eval-num-rays-per-chunk", type=int, default=None)
    ap.add_argument("--set", dest="dp_set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a DATAPARSER config field before setup (yaml-parsed value), "
                         "e.g. --set use_range_image_raydrop=False to reproduce pre-range-image evals")
    ap.add_argument("--pythonpath", action="append", default=[], help="extra source dirs for plugins")
    ap.add_argument("--no-median", action="store_true", help="ray adapter: skip the median-depth tap")
    ap.add_argument("--crosscheck", type=int, default=0,
                    help="ray adapter: for the first N sweeps also run the model's OWN get_image_metrics_and_images "
                         "and store native depth_median_l2 / chamfer_distance / ray_drop_accuracy next to ours "
                         "(native chamfer is a CPU cdist: ~1 min per sweep)")
    a = ap.parse_args()
    if a.output is None and a.dump is None:
        ap.error("give --output and/or --dump")
    return a


# ------------------------------------------------------------------------------------------
# median-depth tap for ray-based models
# ------------------------------------------------------------------------------------------
class MedianDepthTap(torch.nn.Module):
    """Wraps model.renderer_depth. The first call after arm() also computes the median depth from
    the same (weights, ray_samples[, ray_indices, num_rays]) -- in NeuRAD and UniSim that first call
    is the main-field depth (proposal depths come after it)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.armed, self.median = False, None

    def arm(self):
        self.armed, self.median = True, None

    def forward(self, weights, ray_samples, ray_indices=None, num_rays=None, **kw):
        if ray_indices is None and num_rays is None and not kw:
            out = self.inner(weights=weights, ray_samples=ray_samples)
        else:
            out = self.inner(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays, **kw)
        if self.armed:
            self.armed = False
            self.median = median_depth(weights, ray_samples, ray_indices, num_rays, out)
        return out


@torch.no_grad()
def median_depth(weights, ray_samples, ray_indices, num_rays, like):
    steps = ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2)[..., 0]
    w = weights[..., 0]
    if ray_indices is None:  # padded [..., S]
        cw = torch.cumsum(w, dim=-1).contiguous()
        idx = torch.searchsorted(cw, torch.full((*cw.shape[:-1], 1), 0.5, device=cw.device, dtype=cw.dtype), side="left")
        idx = idx.clamp(0, steps.shape[-1] - 1)
        return torch.gather(steps, -1, idx).to(like.dtype).reshape(like.shape)
    # packed (nerfacc): samples sorted by ray, then by distance
    import nerfacc
    ray_indices = ray_indices.long()
    M = w.shape[0]
    packed = nerfacc.pack_info(ray_indices, num_rays)
    cw = nerfacc.inclusive_sum(w.float().contiguous(), packed_info=packed)
    ar = torch.arange(M, device=w.device)
    cand = torch.where(cw >= 0.5, ar, torch.full_like(ar, M))
    first = torch.full((num_rays,), M, device=w.device, dtype=ar.dtype).scatter_reduce(0, ray_indices, cand, "amin")
    last = torch.full((num_rays,), -1, device=w.device, dtype=ar.dtype).scatter_reduce(0, ray_indices, ar, "amax")
    sel = torch.where(first < M, first, last)
    med = torch.where(sel >= 0, steps.reshape(-1)[sel.clamp(min=0)], torch.zeros_like(sel, dtype=steps.dtype))
    return med.to(like.dtype).reshape(like.shape)


def install_ray_hooks(model, want_median):
    """Instance-level wrappers (the model class is untouched)."""
    state = {"rb": None}
    orig_cam = model.get_outputs_for_camera_ray_bundle

    def cam_wrap(camera_ray_bundle, *args, **kw):
        state["rb"] = camera_ray_bundle  # the object get_outputs_for_lidar later builds 'points' from
        return orig_cam(camera_ray_bundle, *args, **kw)

    model.get_outputs_for_camera_ray_bundle = cam_wrap
    tap = None
    if want_median and hasattr(model, "renderer_depth") and hasattr(model, "get_nff_outputs"):
        tap = MedianDepthTap(model.renderer_depth)
        model.renderer_depth = tap
        orig_nff = model.get_nff_outputs

        def nff_wrap(ray_bundle, *args, **kw):
            tap.arm()
            out = orig_nff(ray_bundle, *args, **kw)
            if tap.median is not None:
                out["median_depth"] = tap.median
            tap.armed = False
            return out

        model.get_nff_outputs = nff_wrap
    return state, tap


def ray_drop_rule(model):
    c = model.config
    src = c.loss if hasattr(c, "loss") and hasattr(c.loss, "ray_drop_loss_mult") else c
    mult = float(getattr(src, "ray_drop_loss_mult", 0.0))
    dist = float(getattr(src, "non_return_lidar_distance", 150.0))
    return mult, dist


# ------------------------------------------------------------------------------------------
# adapters -> one standard per-frame record
# ------------------------------------------------------------------------------------------
fl = lambda x: x.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
npf = lambda x: x.detach().float().cpu().numpy().astype(np.float32)


def frame_raster(model, lidar, batch):
    """SplatAD. Byte-for-byte the lidar_chamfer_eval.py / lidar_dump.py logic."""
    torch.cuda.synchronize(); t0 = time.time()
    outputs = model.get_lidar_outputs(lidar)
    torch.cuda.synchronize(); dt = time.time() - t0
    pred, gt = model.filter_lidar_pred_and_gt(outputs, batch, output_point_cloud=True)
    rp = batch["raster_pts"]  # [.., 5]: azimuth, elevation, depth, time, intensity  (tile bucket)
    gt_drop = gt["ray_drop"].reshape(-1)
    pred_drop = (pred["ray_drop"].sigmoid() > 0.5).reshape(-1)
    az, el = torch.deg2rad(rp[..., 0].flatten()), torch.deg2rad(rp[..., 1].flatten())
    dirs = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], dim=-1)
    pts_all = outputs["depth"].view(-1, 1) * dirs + batch["linear_velocities_local"] * rp[..., 3].view(-1, 1)
    Pr = pts_all[batch["raster_pts_valid_depth_and_did_return"]].reshape(-1, 3)
    rec = dict(Pr=npf(Pr),
        P=npf(pred["point_cloud"]), G=npf(gt["point_cloud"]), Pm=npf(pred["median_point_cloud"]),
        pred_depth_ret=pred["depth"], gt_depth_ret=gt["depth"],
        valid=gt["valid"].reshape(-1), gt_drop=gt_drop, pred_drop=pred_drop, dt=dt,
    )
    dump = dict(
        raster_shape=np.array(tuple(rp.shape)), azimuth=fl(rp[..., 0]), elevation=fl(rp[..., 1]),
        gt_range=fl(rp[..., 2]), gt_intensity=fl(rp[..., 4]),
        pred_range=fl(outputs["depth"]), pred_median_range=fl(outputs["median_depth"]),
        pred_intensity=fl(outputs["intensity"]), pred_ray_drop_logit=fl(outputs["ray_drop_logits"]),
        gt_did_return=(~gt_drop).cpu().numpy(), pred_did_return=((~pred_drop) & gt["valid"].reshape(-1)).cpu().numpy(),
    )
    return rec, dump


def frame_ray(model, lidar, batch, state, tap, rule, crosscheck=False):
    """NeuRAD / UniSim. Mirrors <Model>.get_image_metrics_and_images' lidar branch."""
    torch.cuda.synchronize(); t0 = time.time()
    outputs, batch = model.get_outputs_for_lidar(lidar, batch=batch)
    torch.cuda.synchronize(); dt = time.time() - t0
    dev = outputs["depth"].device
    pts = batch["lidar"].to(dev)
    did_return = batch["did_return"][:, 0].to(dev)
    is_lidar = batch["is_lidar"][:, 0].to(dev)
    assert bool(is_lidar.all()), "lidar eval batch is expected to contain only lidar rays"
    depth = outputs["depth"].reshape(-1)
    logit = outputs["ray_drop_logits"].reshape(-1) if outputs.get("ray_drop_logits") is not None else torch.zeros_like(depth)
    mult, dist = rule
    pred_ret = (logit.sigmoid() <= 0.5) if mult > 0.0 else (depth < dist)
    points = outputs["points"][..., :3].reshape(-1, 3)

    l2w = lidar.lidar_to_worlds[0].to(dev)
    R, tr = l2w[:3, :3], l2w[:3, 3]
    rb = state["rb"]
    if "median_depth" in outputs and rb is not None:
        med = outputs["median_depth"].reshape(-1)
        o, d = rb.origins.reshape(-1, 3), rb.directions.reshape(-1, 3)
        # same transform as ADModel.get_outputs_for_lidar: w2l @ (o + d * depth)
        chk = ((o + d * depth[:, None]) - tr) @ R
        err = float((chk - points).abs().max())
        assert err < 1e-2, f"ray-bundle capture does not reproduce outputs['points'] (max err {err:.3g} m)"
        med_pts = ((o + d * med[:, None]) - tr) @ R
    else:
        med = torch.full_like(depth, float("nan")); med_pts = torch.full_like(points, float("nan"))

    rec = dict(Pr=npf(points[did_return]),
        P=npf(points[pred_ret]), G=npf(pts[did_return, :3]), Pm=npf(med_pts[pred_ret]),
        pred_depth_ret=depth[did_return], gt_depth_ret=batch["distance"].reshape(-1).to(dev)[did_return],
        valid=torch.ones_like(did_return), gt_drop=~did_return, pred_drop=~pred_ret, dt=dt,
    )
    xyz = pts[:, :3]
    rng = xyz.norm(dim=-1)
    dump = dict(
        raster_shape=np.array([xyz.shape[0]]),
        azimuth=fl(torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0]))),
        elevation=fl(torch.rad2deg(torch.atan2(xyz[:, 2], xyz[:, :2].norm(dim=-1)))),
        gt_range=fl(batch["distance"].to(dev)), gt_intensity=fl(pts[:, 3]),
        pred_range=fl(depth), pred_median_range=fl(med),
        pred_intensity=fl(outputs["intensity"]) if outputs.get("intensity") is not None else np.zeros(len(depth), np.float32),
        pred_ray_drop_logit=fl(logit),
        gt_did_return=did_return.cpu().numpy(), pred_did_return=pred_ret.cpu().numpy(),
    )
    del rng
    if crosscheck:
        native, _ = model.get_image_metrics_and_images(outputs, batch)
        rec["native"] = {k: float(native[k]) for k in ("depth_median_l2", "chamfer_distance", "ray_drop_accuracy") if k in native}
    return rec, dump


# ------------------------------------------------------------------------------------------
# shared metric arithmetic (identical to lidar_chamfer_eval.py for the shared keys)
# ------------------------------------------------------------------------------------------
def frame_metrics(r):
    P, G, Pm = r["P"], r["G"], r["Pm"]
    if len(P) == 0 or len(G) == 0:
        return None
    tg, tp = cKDTree(G), cKDTree(P)
    d_pg = tg.query(P, k=1, workers=16)[0]; d_gp = tp.query(G, k=1, workers=16)[0]
    gt_range = float(np.linalg.norm(G, axis=1).mean())
    dd = (r["pred_depth_ret"] - r["gt_depth_ret"]).abs()
    valid, gd, pd = r["valid"], r["gt_drop"], r["pred_drop"]
    row = dict(
        n_pred=int(len(P)), n_gt=int(len(G)), gt_range_m=gt_range,
        cd_m=0.5 * (d_pg.mean() + d_gp.mean()), cd2_m2=float((d_pg ** 2).sum() + (d_gp ** 2).sum()) / len(G),
    )
    if np.isfinite(Pm).all() and len(Pm):
        tpm = cKDTree(Pm); d_pmg = tg.query(Pm, k=1, workers=16)[0]; d_gpm = tpm.query(G, k=1, workers=16)[0]
        row["cd_median_pc_m"] = 0.5 * (d_pmg.mean() + d_gpm.mean())
    else:
        row["cd_median_pc_m"] = None
    row.update(
        depth_median_l2_m=float(dd.median()), depth_mean_l2_m=float(dd.mean()),
        ray_drop_acc=float(((pd == gd) * valid).sum() / valid.sum()),
        # ---- extras (not in lidar_chamfer_eval.py) ----
        depth_median_sq_l2_m2=float(torch.median(dd ** 2)),
        frac_gt_nonreturn=float((gd & valid).sum() / valid.sum()),
        ray_drop_acc_returns=float(((~pd) & (~gd) & valid).sum() / max(int(((~gd) & valid).sum()), 1)),
        ray_drop_acc_nonreturns=(float((pd & gd & valid).sum() / int((gd & valid).sum()))
                                 if int((gd & valid).sum()) > 0 else None),
        n_rays=int(valid.sum()), n_return=int(((~gd) & valid).sum()), n_nonreturn=int((gd & valid).sum()),
    )
    # secondary, drop-independent: every GT-returned ray, no ray-drop filtering
    Pr = r["Pr"]
    if len(Pr):
        d_rg = tg.query(Pr, k=1, workers=16)[0]; d_gr = cKDTree(Pr).query(G, k=1, workers=16)[0]
        row.update(cd_returns_m=0.5 * (d_rg.mean() + d_gr.mean()),
                   cd2_returns_m2=float((d_rg ** 2).sum() + (d_gr ** 2).sum()) / len(G),
                   depth_median_l2_returns_m=float(dd.median()))
    return row


def main():
    a = parse()
    for p in [*a.pythonpath, *map(str, DEFAULT_PLUGIN_SRC)]:
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, str(p))
    import yaml
    from nerfstudio.utils.eval_utils import eval_setup

    def update(cfg):
        dp = cfg.pipeline.datamanager.dataparser
        user_keys = {kv.split("=", 1)[0] for kv in a.dp_set}
        if "use_range_image_raydrop" not in user_keys:
            assert hasattr(dp, "use_range_image_raydrop"), \
                f"{type(dp).__name__} has no use_range_image_raydrop: cannot build the benchmark eval ray set"
            if not dp.use_range_image_raydrop:
                print("[protocol] training config had use_range_image_raydrop=False -> forcing True for eval")
            dp.use_range_image_raydrop = True
        for kv in a.dp_set:
            k, v = kv.split("=", 1)
            assert hasattr(dp, k), f"dataparser config has no field '{k}'"
            val = Path(v) if isinstance(getattr(dp, k), Path) else yaml.safe_load(v)
            setattr(dp, k, val); print(f"[override] dataparser.{k} = {getattr(dp, k)!r}")
        return cfg

    config, pipeline, ckpt, step = eval_setup(a.config, eval_num_rays_per_chunk=a.eval_num_rays_per_chunk,
                                              test_mode="val", update_config_callback=update)
    model, dm = pipeline.model, pipeline.datamanager
    model.eval()
    range_image_on = bool(getattr(config.pipeline.datamanager.dataparser, "use_range_image_raydrop", False))
    adapter = "raster" if (hasattr(model, "get_lidar_outputs") and hasattr(model, "filter_lidar_pred_and_gt")) else "ray"
    rule = ray_drop_rule(model) if adapter == "ray" else None
    print(f"model={type(model).__name__} adapter={adapter} step={step} ckpt={ckpt}"
          + (f" ray_drop_loss_mult={rule[0]} non_return_lidar_distance={rule[1]}" if rule else ""))

    # ---- camera inference timing (first n held-out frames) ----
    cam_times, pixels = [], 0
    with torch.no_grad():
        for i, (camera, batch) in enumerate(dm.fixed_indices_eval_dataloader):
            if i >= a.n_cameras: break
            torch.cuda.synchronize(); t0 = time.time(); out = model.get_outputs_for_camera(camera); torch.cuda.synchronize()
            cam_times.append(time.time() - t0); pixels += out["rgb"].shape[0] * out["rgb"].shape[1]
    cam_ms = 1000 * float(np.mean(cam_times[3:])) if len(cam_times) > 3 else None

    state, tap = (install_ray_hooks(model, not a.no_median) if adapter == "ray" else (None, None))
    if a.dump: a.dump.mkdir(parents=True, exist_ok=True)
    rows, lid_times, n, crosschecks = [], [], 0, []
    with torch.no_grad():
        for lidar, batch in dm.fixed_indices_eval_lidar_dataloader:
            if a.max_frames and n >= a.max_frames: break
            rec, dump = frame_raster(model, lidar, batch) if adapter == "raster" else frame_ray(model, lidar, batch, state, tap, rule, crosscheck=n < a.crosscheck)
            lid_times.append(rec["dt"])
            t = int(lidar.metadata["timesteps"].flatten()[0]) if (lidar.metadata and "timesteps" in lidar.metadata) else n
            if n == 0:
                print("frame", t, "rays", int(rec["valid"].numel()), "valid", int(rec["valid"].sum()),
                      "gt returns", len(rec["G"]), "gt non-returns", int((rec["gt_drop"] & rec["valid"]).sum()),
                      "pred pts", len(rec["P"]))
            if a.dump:
                np.savez_compressed(a.dump / f"frame_{t:04d}.npz", pred_xyz=rec["P"], gt_xyz=rec["G"], pred_median_xyz=rec["Pm"],
                                    **dump, lidar_to_world=lidar.lidar_to_worlds[0].cpu().numpy().astype(np.float32))
            if a.output:
                row = frame_metrics(rec)
                if row is not None:
                    row["timestep"] = t; rows.append(row)
                    if "native" in rec:
                        xc = dict(timestep=t, native=rec["native"], ours=dict(depth_median_l2=row["depth_median_sq_l2_m2"],
                                  chamfer_distance=row["cd2_m2"], ray_drop_accuracy=row["ray_drop_acc"]))
                        print("crosscheck", json.dumps(xc)); crosschecks.append(xc)
            n += 1
    print(f"{n} sweeps rendered" + (f" -> dump {a.dump}" if a.dump else ""))
    if not a.output:
        return
    skip = ("n_pred", "n_gt", "timestep", "n_rays", "n_return", "n_nonreturn")
    n_return, n_nonreturn = int(sum(r["n_return"] for r in rows)), int(sum(r["n_nonreturn"] for r in rows))
    if range_image_on:
        ray_set = EVAL_RAY_SET_FULL if n_nonreturn > 0 else EVAL_RAY_SET_FULL + "(range_image_missing)"
    else:
        ray_set = EVAL_RAY_SET_RETURNS
    if adapter == "raster":
        has_head = float(getattr(model.config, "ray_drop_lambda", 0.0)) > 0.0
    else:
        has_head = rule[0] > 0.0
    def mean_of(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None
    agg = {k: mean_of(k) for k in dict.fromkeys(k for r in rows for k in r) if k not in skip}
    shared = ["gt_range_m", "cd_m", "cd2_m2", "cd_median_pc_m", "depth_median_l2_m", "depth_mean_l2_m", "ray_drop_acc"]
    extras = {k: agg[k] for k in agg if k not in shared}
    res = dict(config=str(a.config), step=int(step), n_frames=len(rows), n_pred=int(sum(r["n_pred"] for r in rows)), n_gt=int(sum(r["n_gt"] for r in rows)),
               **{k: agg[k] for k in shared}, rcd=agg["cd_m"] / agg["gt_range_m"],
               infer_ms_per_camera_frame=cam_ms, infer_cameras_timed=len(cam_times), infer_megapixels_per_s=(pixels / sum(cam_times) / 1e6) if cam_times else None,
               infer_ms_per_lidar_sweep=1000 * float(np.mean(lid_times[2:])) if len(lid_times) > 2 else None,
               definitions="cd_m = symmetric mean NN distance (m); cd2_m2 = nerfstudio chamfer (sum sq NN both ways / |gt|); rcd = cd_m / mean GT range; "
                           "cd_median_pc_m uses the model's median-depth point cloud; inference = wall-clock on one GH200, cuda-synchronized, warm-up excluded",
               model=type(model).__name__, adapter=adapter, n_rays=int(sum(r["n_rays"] for r in rows)),
               eval_ray_set=ray_set, n_return=n_return, n_nonreturn=n_nonreturn, has_ray_drop_head=has_head,
               ray_drop_rule=("sigmoid(logit)<=0.5 -> return" if adapter == "raster" or rule[0] > 0 else f"depth<{rule[1]} m -> return (ray-drop head untrained)"),
               dataparser_overrides=a.dp_set, **extras,
               definitions_extra="ray_drop_acc counts the dummy (2000 m) non-return rays as the drop class; GT cloud/depth use did_return rays only "
                                 "(|p| < valid_lidar_distance_threshold = 1000 m); depth_median_sq_l2_m2 = neurad-studio's own depth_median_l2 "
                                 "(median squared error); ray adapter timing includes the median-depth tap; cd_returns_m / cd2_returns_m2 / "
                                 "depth_median_l2_returns_m are drop-independent: one predicted point (rendered depth along the GT ray) for "
                                 "EVERY GT-returned ray, no ray-drop filtering (full definition in lidar_dump_ns.py docstring)",
               native_crosscheck=crosschecks, per_frame=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
