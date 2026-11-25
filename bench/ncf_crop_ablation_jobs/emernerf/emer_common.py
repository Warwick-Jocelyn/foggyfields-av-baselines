"""Shared helpers for the EmerNeRF harness scripts: load a trained run exactly as
train_emernerf.py built it (same merged config.yaml, same dataset class, same split, same
builders), then load the checkpoint.

Run with the EmerNeRF checkout importable (the scripts chdir into EMERNERF_ROOT, because
the upstream code resolves configs/ relative to the cwd).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

EMERNERF_ROOT = os.environ.get("EMERNERF_ROOT", "/capstor/scratch/cscs/yitiwang/ncf-crop-ablation/code/emernerf")


def enter_emernerf() -> None:
    os.chdir(EMERNERF_ROOT)
    if EMERNERF_ROOT not in sys.path:
        sys.path.insert(0, EMERNERF_ROOT)


def latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    cks = sorted(glob.glob(str(Path(run_dir) / "checkpoint_*.pth")))
    if not cks:
        raise FileNotFoundError(f"no checkpoint_*.pth under {run_dir}")
    best = max(cks, key=lambda p: int(re.search(r"checkpoint_(\d+)\.pth$", p).group(1)))
    return Path(best), int(re.search(r"checkpoint_(\d+)\.pth$", best).group(1))


def load_run(run_dir: Path, ckpt: Path | None = None, chunk: int | None = None):
    """-> (cfg, dataset, model, proposal_estimator, proposal_networks, ckpt_path, step)."""
    enter_emernerf()
    import torch
    from omegaconf import OmegaConf

    import builders
    import utils.misc as misc
    from datasets.nvidia_av_fog import NVIDIAAVFogDataset

    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    if cfg.data.dataset != "nvidia_av_fog":
        raise ValueError(f"unexpected dataset {cfg.data.dataset}")
    # A config.yaml written before the timing fixes (F1b camera pose time, E3 per-point
    # LiDAR poses) lacks their keys, and the loader would apply the corrected defaults to
    # a model trained without them. Such a run gets the pre-fix values; runs trained since
    # carry the keys explicitly (train_emernerf.setup writes them).
    from datasets.nvidia_av_fog import fill_fix_defaults

    filled = fill_fix_defaults(cfg, legacy=True)
    if filled:
        print(f"[emer_common] {run_dir}/config.yaml predates the timing fixes -> pre-fix values: {filled}",
              flush=True)
    print("[emer_common] rays: camera_pose_time=%s lidar_pose_time=%s" % (
        cfg.data.pixel_source.camera_pose_time, cfg.data.lidar_source.lidar_pose_time), flush=True)
    misc.fix_random_seeds(cfg.optim.seed)
    if chunk:
        cfg.render.render_chunk_size = int(chunk)
    if ckpt is None:
        ckpt, step = latest_checkpoint(run_dir)
    else:
        step = int(re.search(r"(\d+)", Path(ckpt).stem).group(1))
    device = torch.device("cuda")
    dataset = NVIDIAAVFogDataset(data_cfg=cfg.data)
    est, props = builders.build_estimator_and_propnet_from_cfg(
        nerf_cfg=cfg.nerf, optim_cfg=cfg.optim, dataset=dataset, device=device
    )
    model = builders.build_model_from_cfg(cfg=cfg.nerf.model, dataset=dataset, device=device)
    misc.resume_from_checkpoint(ckpt_path=str(ckpt), model=model, proposal_networks=props,
                                proposal_estimator=est)
    model.eval()
    est.eval()
    for p in props:
        p.eval()
    return cfg, dataset, model, est, props, Path(ckpt), step


def heldout_timesteps(dataset, max_frames: int | None):
    """Held-out ABSOLUTE timesteps (from the dataset's split, i.e. the split json), and the
    subset to render. A cap picks evenly spaced held-out timesteps (deterministic)."""
    import numpy as np

    allt = sorted(int(t) + dataset.start_timestep for t in dataset.test_timesteps.tolist())
    if max_frames and 0 < max_frames < len(allt):
        idx = np.unique(np.linspace(0, len(allt) - 1, max_frames).round().astype(int))
        return allt, [allt[i] for i in idx]
    return allt, allt
