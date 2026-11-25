#!/usr/bin/env python3
"""LiDAR reconstruction visualisations from the per-frame npz dumps.

Produces, per run:
  bev.png            predicted vs GT point cloud, top-down, coloured by height, + an
                     error map (per-GT-point nearest-neighbour distance to the prediction)
  range.png          GT / predicted / signed-difference range image (the 128 x 3600 raster)
  occupancy.png      BEV occupancy grid: GT, predicted, and the disagreement (miss / false)
  cd_by_frame.png    Chamfer per frame, and error vs range
  bev.mp4            the BEV pair as a video over all dumped frames

CPU only. Reads what lidar_dump.py wrote.
"""
import argparse, glob, json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}

def bev_panel(ax, xyz, title, rng, cmap="turbo"):
    m = (np.abs(xyz[:, 0]) < rng) & (np.abs(xyz[:, 1]) < rng)
    p = xyz[m]
    ax.scatter(p[:, 0], p[:, 1], s=0.05, c=np.clip(p[:, 2], -3, 6), cmap=cmap, linewidths=0)
    ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9); ax.set_facecolor("#0d0d0d")
    ax.tick_params(labelsize=7)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=-1, help="which dumped frame for the still images")
    ap.add_argument("--range", type=float, default=60.0)
    ap.add_argument("--cell", type=float, default=0.4, help="occupancy cell size (m)")
    ap.add_argument("--max-frames-curve", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(a.dump.glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no npz under {a.dump}")
    pick = files[len(files) // 2] if a.frame < 0 else a.dump / f"frame_{a.frame:04d}.npz"
    d = load(pick)
    tag = f"{a.dump.name}  {pick.stem}"
    print(f"{len(files)} frames; stills from {pick.name}")

    # ---------------- BEV + per-point error ----------------
    gt, pr = d["gt_xyz"], d["pred_xyz"]
    tree = cKDTree(pr)
    err, _ = tree.query(gt, k=1, workers=-1)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), facecolor="#0d0d0d")
    bev_panel(axs[0], gt, f"GT point cloud  ({len(gt):,} pts)", a.range)
    bev_panel(axs[1], pr, f"PREDICTED  ({len(pr):,} pts)", a.range)
    m = (np.abs(gt[:, 0]) < a.range) & (np.abs(gt[:, 1]) < a.range)
    sc = axs[2].scatter(gt[m, 0], gt[m, 1], s=0.05, c=np.clip(err[m], 0, 1.0),
                        cmap="inferno", linewidths=0)
    axs[2].set_xlim(-a.range, a.range); axs[2].set_ylim(-a.range, a.range)
    axs[2].set_aspect("equal"); axs[2].set_facecolor("#0d0d0d")
    axs[2].set_title(f"NN error GT->pred   median {np.median(err):.3f} m", fontsize=9)
    axs[2].tick_params(labelsize=7)
    cb = fig.colorbar(sc, ax=axs[2], fraction=0.046); cb.set_label("m", fontsize=8)
    for ax in axs: [s.set_color("#555") for s in ax.spines.values()]
    fig.suptitle(tag, color="w", fontsize=11)
    fig.tight_layout(); fig.savefig(a.outdir / "bev.png", dpi=130, facecolor="#0d0d0d"); plt.close(fig)

    # ---------------- range image, rebuilt from the stored angles ----------------
    # NOTE: the dumped [B, H, W, 5] array is SplatAD's tile BUCKET, not a sensor raster --
    # (h, w) carries no geometric meaning, each cell stores its own azimuth/elevation. So the
    # range image has to be re-binned from those angles; reshaping (h, w) would be nonsense.
    az, el = d["azimuth"], d["elevation"]
    gr_f, pr_f = d["gt_range"], d["pred_range"]
    # returned rays only: dummy non-return rays sit at DUMMY_DISTANCE_VALUE = 2000 m (> 1000 m threshold);
    # older SplatAD dumps have none, so for them this is the same mask as before
    keep = (gr_f > 0) & (gr_f < 1000.0)
    az, el, gr_f, pr_f = az[keep], el[keep], gr_f[keep], pr_f[keep]
    NA, NE = 1800, 128
    ai = np.clip(((az + 180.0) / 360.0 * NA).astype(int), 0, NA - 1)
    e_lo, e_hi = np.percentile(el, 0.1), np.percentile(el, 99.9)
    ei = np.clip(((el - e_lo) / max(e_hi - e_lo, 1e-6) * NE).astype(int), 0, NE - 1)
    flat = ei * NA + ai
    def raster(v):
        out = np.full(NE * NA, np.nan, np.float32)
        order = np.argsort(-v)                 # farthest first, nearest wins the cell
        out[flat[order]] = v[order]
        return out.reshape(NE, NA)
    GR, PR = raster(gr_f), raster(pr_f)
    DF = PR - GR
    fig, axs = plt.subplots(3, 1, figsize=(15, 8.2), facecolor="#0d0d0d")
    for ax, img, t, kw in [
        (axs[0], GR, "GT range (m)", dict(cmap="viridis", vmin=0, vmax=60)),
        (axs[1], PR, "Predicted range (m)", dict(cmap="viridis", vmin=0, vmax=60)),
        (axs[2], DF, "pred - GT (m)", dict(cmap="coolwarm", vmin=-1.5, vmax=1.5))]:
        im = ax.imshow(img, aspect="auto", origin="lower", interpolation="nearest",
                       extent=[-180, 180, e_lo, e_hi], **kw)
        ax.set_title(t, fontsize=9, color="w"); ax.tick_params(labelsize=7, colors="w")
        ax.set_ylabel("elevation (deg)", fontsize=8, color="w")
        fig.colorbar(im, ax=ax, fraction=0.02)
    axs[2].set_xlabel("azimuth (deg)", fontsize=8, color="w")
    good = np.isfinite(DF)
    fig.suptitle(tag + f"   -   range image re-binned from stored angles"
                 f"   |pred-GT| median {np.nanmedian(np.abs(DF[good])):.3f} m", color="w", fontsize=11)
    fig.tight_layout(); fig.savefig(a.outdir / "range.png", dpi=120, facecolor="#0d0d0d"); plt.close(fig)

    # ---------------- BEV occupancy ----------------
    R, C = a.range, a.cell
    n = int(2 * R / C)
    def grid(xyz):
        m = (np.abs(xyz[:, 0]) < R) & (np.abs(xyz[:, 1]) < R) & (xyz[:, 2] > -2.5) & (xyz[:, 2] < 3.0)
        xi = ((xyz[m, 0] + R) / C).astype(int).clip(0, n - 1)
        yi = ((xyz[m, 1] + R) / C).astype(int).clip(0, n - 1)
        g = np.zeros((n, n), bool); g[yi, xi] = True; return g
    gg, gp = grid(gt), grid(pr)
    inter = (gg & gp).sum(); union = (gg | gp).sum()
    rgb = np.zeros((n, n, 3))
    rgb[..., 1] = (gg & gp)                 # green  = both
    rgb[..., 0] = (gp & ~gg)                # red    = predicted only (false positive)
    rgb[..., 2] = (gg & ~gp)                # blue   = GT only (missed)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), facecolor="#0d0d0d")
    for ax, img, t in [(axs[0], gg, "GT occupancy"), (axs[1], gp, "Predicted occupancy"),
                       (axs[2], rgb, f"green=both  red=false  blue=missed   IoU {inter/max(union,1):.3f}")]:
        ax.imshow(img if img.ndim == 3 else img, origin="lower", cmap=None if img.ndim == 3 else "gray",
                  extent=[-R, R, -R, R], interpolation="nearest")
        ax.set_title(t, fontsize=9, color="w"); ax.set_facecolor("#0d0d0d"); ax.tick_params(labelsize=7, colors="w")
    fig.suptitle(tag + f"   —   BEV occupancy, {C} m cells, z in [-2.5, 3.0] m", color="w", fontsize=11)
    fig.tight_layout(); fig.savefig(a.outdir / "occupancy.png", dpi=130, facecolor="#0d0d0d"); plt.close(fig)

    # ---------------- per-frame curves ----------------
    sel = files if not a.max_frames_curve else files[: a.max_frames_curve]
    cds, meds, rngs = [], [], []
    err_by_rng = [[] for _ in range(12)]
    for f in sel:
        e = load(f)
        g, p = e["gt_xyz"], e["pred_xyz"]
        t1 = cKDTree(p); d1, _ = t1.query(g, k=1, workers=-1)
        t2 = cKDTree(g); d2, _ = t2.query(p, k=1, workers=-1)
        cds.append(0.5 * (d1.mean() + d2.mean())); meds.append(np.median(d1))
        r = np.linalg.norm(g, axis=1); rngs.append(r.mean())
        b = np.clip((r / 10).astype(int), 0, 11)
        for k in range(12):
            mm = b == k
            if mm.any(): err_by_rng[k].append(float(np.median(d1[mm])))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    axs[0].plot(cds, lw=1.2, label="CD (mean of both directions)")
    axs[0].plot(meds, lw=1.0, alpha=.7, label="median NN, GT->pred")
    axs[0].set_xlabel("held-out frame"); axs[0].set_ylabel("m"); axs[0].legend(fontsize=8)
    axs[0].set_title(f"{a.dump.name} — per-frame Chamfer   mean {np.mean(cds):.4f} m", fontsize=10)
    axs[0].grid(alpha=.3)
    xs = [(k + .5) * 10 for k in range(12)]
    ys = [np.mean(v) if v else np.nan for v in err_by_rng]
    axs[1].plot(xs, ys, "o-", lw=1.4)
    axs[1].set_xlabel("GT range (m)"); axs[1].set_ylabel("median NN error (m)")
    axs[1].set_title("error vs distance", fontsize=10); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.outdir / "cd_by_frame.png", dpi=130); plt.close(fig)

    json.dump({"dump": str(a.dump), "n_frames": len(sel), "cd_mean": float(np.mean(cds)),
               "cd_per_frame": [float(x) for x in cds],
               "bev_iou": float(inter / max(union, 1)), "occ_cell_m": C,
               "median_nn_err_m": float(np.median(err)),
               "err_by_range_m": {f"{int(x-5)}-{int(x+5)}": (None if math.isnan(y) else float(y))
                                  for x, y in zip(xs, ys)}},
              open(a.outdir / "lidar_viz_stats.json", "w"), indent=2)
    print("wrote", a.outdir)

if __name__ == "__main__":
    main()
