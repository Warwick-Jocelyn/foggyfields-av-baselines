"""NVIDIA_AV_Fog dataparser for EmerNeRF.

Reads the user's NVIDIA Physical AI AV fog clips at
  /networkhome/WMGDS/wang3_y/ETH/Dataset/NVIDIA_AV_Fog/processed/<clip>/processed_pinhole/

Per-clip layout (processed_pinhole/):
  images/<cam>/frame_NNNN.jpg              7 cams, NNNN in 0000..0198
  lidar/sweep_NNNN.npy                      [N,6] float32, EGO frame
                                            cols = [x, y, z, intensity, t_offset_us, ring_id]
  calib/K_rectified.json                    per-cam intrinsics {fx, fy, cx, cy, w, h}
  calib/T_camera_rig.json                   per-cam 4x4 = p_cam = T @ p_ego  (cam_from_ego)
  calib/T_lidar_rig.json                    {"lidar_top_360fov": 4x4 = lidar_from_ego}
  calib/T_rig_world.parquet                 per-frame ego->world (3x4, 199 rows)
  splits/{nvs_50_50,nvs_80_20,nvs_90_10,reconstruction_all}.json
  timestamps.txt                            one int per line (spin_center_us per timestep)

Coordinate convention:
  Source dataset frames (NCore native): +x left, +y forward, +z up.
  EmerNeRF AD convention:               +x forward, +y left, +z up.
  We apply ISO 8855 rotation R = [[0,1,0],[1,0,0],[0,0,1]] to all WORLD-frame poses
  (so the canonical "world" is the start-of-clip ego frame, ISO-aligned). LiDAR
  points are loaded in ego frame and transformed through the same ISO-rotated
  lidar_to_world transform, so all modalities live in the same world frame.

Heterogeneous camera resolutions:
  Three different native sizes (1280x960, 1120x800, 480x320). EmerNeRF's base
  ScenePixelSource caches a single (H, W) load_size for ALL images, so we resize
  every camera to a uniform load_size (default 960 x 1280) and scale intrinsics
  per camera accordingly. ORIGINAL_SIZE is a list-of-lists keyed by slot index so
  the per-camera scaling stays correct.

Camera frame convention:
  T_camera_rig stores cam_from_ego in OpenCV convention (x-right, y-down, z-fwd).
  EmerNeRF's get_rays() expects c2w whose rotation columns are the camera basis
  vectors expressed in world coordinates, with the camera frame also in OpenCV
  convention (x-right, y-down, z-fwd). We therefore do NOT post-multiply by
  OPENCV2DATASET (same call as PandaSet — verified empirically there).

Conda env: tested under `splatad` (has torch / pandas / numpy / pyarrow / PIL).
"""
from __future__ import annotations

import json
import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import trange

from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Camera ordering (Waymo-style: left → front → right is the canonical fan).
# The dataset's 7 cams map cleanly onto an L / FL / F / FR / R / front-tele /
# rear-tele subset. We expose num_cams in {1, 3, 5, 7} mirroring PandaSet's
# convention so existing config flows just work.
# ---------------------------------------------------------------------------
NVIDIA_AV_FOG_CAMERAS_ALL = [
    "camera_front_wide_120fov",   # 0: F (wide)
    "camera_cross_left_120fov",   # 1: FL / L (cross-left, ~120 deg)
    "camera_cross_right_120fov",  # 2: FR / R (cross-right, ~120 deg)
    "camera_rear_left_70fov",     # 3: rear-left
    "camera_rear_right_70fov",    # 4: rear-right
    "camera_front_tele_30fov",    # 5: front-tele
    "camera_rear_tele_30fov",     # 6: rear-tele
]

# Per-camera native resolution (h, w). Index matches NVIDIA_AV_FOG_CAMERAS_ALL.
NATIVE_HW: Dict[str, Tuple[int, int]] = {
    "camera_front_wide_120fov":   (960, 1280),
    "camera_cross_left_120fov":   (960, 1280),
    "camera_cross_right_120fov":  (960, 1280),
    "camera_rear_left_70fov":     (800, 1120),
    "camera_rear_right_70fov":    (800, 1120),
    "camera_front_tele_30fov":    (320,  480),
    "camera_rear_tele_30fov":     (320,  480),
}

# ISO 8855 axis remap: NCore (+x left, +y fwd, +z up) → AD (+x fwd, +y left, +z up).
# This is a permutation matrix (swap x and y) — its own inverse.
ISO_8855_R = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
ISO_8855_T4 = np.eye(4, dtype=np.float64)
ISO_8855_T4[:3, :3] = ISO_8855_R


def _resolve_clip_dir(data_root: str, scene_idx) -> str:
    """Resolve <data_root>/processed/<NNN>_*_/processed_pinhole by clip-prefix match.

    scene_idx may be an int (002) or a str ("002") — both work.
    """
    if isinstance(scene_idx, int):
        prefix = f"{scene_idx:03d}"
    else:
        prefix = str(scene_idx).zfill(3)
    processed_root = os.path.join(data_root, "processed")
    matches = sorted(glob(os.path.join(processed_root, f"{prefix}_*")))
    # filter to directories (skip *.md, *.log siblings)
    matches = [m for m in matches if os.path.isdir(m)]
    if not matches:
        raise FileNotFoundError(
            f"No clip dir found at {processed_root}/{prefix}_*"
        )
    if len(matches) > 1:
        logger.warning(f"Multiple clip dirs match prefix {prefix}: {matches}; using {matches[0]}")
    clip_dir = matches[0]
    pin = os.path.join(clip_dir, "processed_pinhole")
    if not os.path.isdir(pin):
        raise FileNotFoundError(f"{pin} does not exist (clip {prefix})")
    return pin


def _load_rig_world_poses(parquet_path: str) -> np.ndarray:
    """Load T_rig_world.parquet → (T, 4, 4) ego_to_world (NCore native frame)."""
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestep_idx").reset_index(drop=True)
    T = len(df)
    poses = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))  # (T, 4, 4)
    for r in range(3):
        for c in range(4):
            col = f"T_rig_world_{r}{c}"
            poses[:, r, c] = df[col].to_numpy(dtype=np.float64)
    return poses


# ===========================================================================
# Pixel source
# ===========================================================================
class NVIDIAAVFogPixelSource(ScenePixelSource):
    # 7 camera slots; per-camera ORIGINAL_SIZE in (h, w) — the base class only
    # uses ORIGINAL_SIZE for documentation, but load_calibrations() below uses
    # this list to scale intrinsics per camera.
    # (Each slot is filled in in __init__ once the cam ordering is known.)
    ORIGINAL_SIZE: List[List[int]] = [[960, 1280]] * 7

    # The cam frame stored in T_camera_rig is already OpenCV-convention.
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )

    def __init__(
        self,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        # We don't have separately-shipped sky / dynamic masks in usable form.
        pixel_data_config.load_dynamic_mask = False
        pixel_data_config.load_sky_mask = False
        super().__init__(pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        # Bookkeeping populated by load_calibrations(); the LiDAR source
        # consumes inv_start_ego_to_world to share the same canonical frame.
        self.inv_start_ego_to_world: np.ndarray = None
        self.create_all_filelist()
        self.load_data()

    # ------------------------------------------------------------------
    def create_all_filelist(self):
        if self.num_cams == 1:
            self.camera_list = [0]
            cam_names = ["camera_front_wide_120fov"]
        elif self.num_cams == 3:
            # FL, F, FR (cross_left, front_wide, cross_right)
            self.camera_list = [1, 0, 2]
            cam_names = [
                "camera_cross_left_120fov",
                "camera_front_wide_120fov",
                "camera_cross_right_120fov",
            ]
        elif self.num_cams == 5:
            # rear_left, FL, F, FR, rear_right
            self.camera_list = [3, 1, 0, 2, 4]
            cam_names = [
                "camera_rear_left_70fov",
                "camera_cross_left_120fov",
                "camera_front_wide_120fov",
                "camera_cross_right_120fov",
                "camera_rear_right_70fov",
            ]
        elif self.num_cams == 7:
            # rear_left, FL, F, FR, rear_right, front_tele, rear_tele
            self.camera_list = [3, 1, 0, 2, 4, 5, 6]
            cam_names = [
                "camera_rear_left_70fov",
                "camera_cross_left_120fov",
                "camera_front_wide_120fov",
                "camera_cross_right_120fov",
                "camera_rear_right_70fov",
                "camera_front_tele_30fov",
                "camera_rear_tele_30fov",
            ]
        else:
            raise NotImplementedError(
                f"num_cams={self.num_cams} not supported for nvidia_av_fog"
            )
        self._cam_names = cam_names
        # Fill per-slot native resolution so intrinsic scaling is correct.
        self.ORIGINAL_SIZE = [list(NATIVE_HW[c]) for c in cam_names]

        scene_path = Path(self.data_path)
        img_filepaths, dynamic_mask_filepaths, sky_mask_filepaths, feat_filepaths = (
            [], [], [], []
        )
        for t in range(self.start_timestep, self.end_timestep):
            for slot_idx, cam_name in enumerate(cam_names):
                img_filepaths.append(
                    str(scene_path / "images" / cam_name / f"frame_{t:04d}.jpg")
                )
                dynamic_mask_filepaths.append("")
                sky_mask_filepaths.append("")
                feat_filepaths.append("")
        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.feat_filepaths = np.array(feat_filepaths)

    # ------------------------------------------------------------------
    def load_calibrations(self):
        scene_path = Path(self.data_path)
        with open(scene_path / "calib" / "K_rectified.json") as f:
            K_all = json.load(f)
        with open(scene_path / "calib" / "T_camera_rig.json") as f:
            T_cam_rig_all = json.load(f)
        # T_rig_world (NCore native ego→world per timestep)
        ego_to_world_native = _load_rig_world_poses(
            str(scene_path / "calib" / "T_rig_world.parquet")
        )  # (T, 4, 4)

        # The processed_pinhole world frame is already ISO 8855 (T_rig_world
        # translation grows along world +x = forward; obstacles_3d_world is in
        # the same frame). Premultiplying by ISO_8855_T4 used to mirror-reflect
        # everything (det=-1) and scrambled poses. Use the native frame directly.
        ego_to_world_iso = ego_to_world_native  # (T,4,4) -- already ISO
        ego_to_world_start = ego_to_world_iso[self.start_timestep]
        inv_start = np.linalg.inv(ego_to_world_start)
        self.inv_start_ego_to_world = inv_start  # shared with LiDAR source

        # ------- per-slot intrinsics scaled to load_size -------
        load_h, load_w = self.data_cfg.load_size[0], self.data_cfg.load_size[1]
        _intrinsics = []
        for slot_idx, cam_name in enumerate(self._cam_names):
            K = K_all[cam_name]
            native_h, native_w = NATIVE_HW[cam_name]
            sx = load_w / native_w
            sy = load_h / native_h
            fx, fy = K["fx"] * sx, K["fy"] * sy
            cx, cy = K["cx"] * sx, K["cy"] * sy
            _intrinsics.append(
                np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            )

        # ------- cam_to_ego (time-invariant) -------
        # T_camera_rig.json actually stores T_rig_from_cam (= cam_to_ego) despite the
        # naming convention. Verified by physical sanity: file translation column places
        # front cams at (1.95, ~0, 1.32) m (windshield); inverting puts them underground.
        cam_to_egos: List[np.ndarray] = []
        for cam_name in self._cam_names:
            T_rig_from_cam = np.array(T_cam_rig_all[cam_name], dtype=np.float64)
            cam_to_egos.append(T_rig_from_cam)

        # ------- per-frame ego_to_world (rebased to start) and cam_to_world -------
        ego_to_worlds, cam_to_worlds, intrinsics = [], [], []
        cam_ids, timestamps, timesteps = [], [], []
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_t = inv_start @ ego_to_world_iso[t]
            ego_to_worlds.append(ego_to_world_t)
            for slot_idx in range(len(self._cam_names)):
                cam_ids.append(self.camera_list[slot_idx])
                c2w = ego_to_world_t @ cam_to_egos[slot_idx]
                cam_to_worlds.append(c2w)
                intrinsics.append(_intrinsics[slot_idx])
                # timesteps + timestamps both integer-relative-to-start (matches PandaSet)
                timestamps.append(t - self.start_timestep)
                timesteps.append(t - self.start_timestep)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        self.cam_ids = torch.from_numpy(np.array(cam_ids, dtype=np.int64)).long()
        self._timestamps = torch.from_numpy(np.array(timestamps, dtype=np.float32)).float()
        self._timesteps = torch.from_numpy(np.array(timesteps, dtype=np.int64)).long()


# ===========================================================================
# LiDAR source
# ===========================================================================
class NVIDIAAVFogLiDARSource:
    """LiDAR source for NVIDIA_AV_Fog. Mirrors PandaSetLiDARSource's API surface.

    .npy sweeps are stored in the EGO frame at spin_center timestamp t. We
    transform each point to world coordinates via
        p_world = inv_start @ ISO_8855_T4 @ T_rig_world[t] @ p_ego
    (i.e. ego→native_world→ISO_world→canonical_world). The same compound
    transform IS the lidar_to_world we expose; lidar = ego here because
    T_lidar_rig is so close to identity (translation only) we treat the
    lidar's origin as the ego origin for ray-bookkeeping, then we apply the
    actual T_lidar_rig translation/rotation onto the ego-frame points before
    transforming. (Equivalent to using T_lidar_world as the ray origin.)
    """

    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        inv_start_ego_to_world: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ):
        self.data_cfg = lidar_data_config
        self.device = device
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.inv_start = inv_start_ego_to_world  # canonical-world from ISO-world
        self.origins = self.directions = self.ranges = None
        self.lidar_to_worlds = None
        self._timestamps = self._timesteps = self._normalized_timestamps = None
        self.cached_origins = self.cached_directions = self.cached_ranges = None
        self.cached_normalized_timestamps = None
        self.cached_indices = None
        self._unique_normalized_timestamps = None
        self.laser_ids = self.intensities = None
        self.create_all_filelist()
        self.load_data()

    # ------------------------------------------------------------------
    def create_all_filelist(self):
        self.lidar_filepaths = np.array([
            os.path.join(self.data_path, "lidar", f"sweep_{t:04d}.npy")
            for t in range(self.start_timestep, self.end_timestep)
        ])

    def load_data(self):
        self.load_calibrations()
        self.load_lidar()
        logger.info("[NVIDIA_AV_Fog-LiDAR] All Lidar Data loaded.")

    def load_calibrations(self):
        scene_path = Path(self.data_path)
        with open(scene_path / "calib" / "T_lidar_rig.json") as f:
            tl = json.load(f)
        # T_lidar_rig.json actually stores T_rig_from_lidar = lidar_to_ego (despite the
        # "lidar_from_ego" naming). Translation (1.14, 0, 1.94) m = roof-mounted lidar
        # position on the rig; inverting puts it underground.
        T_rig_from_lidar = np.array(tl["lidar_top_360fov"], dtype=np.float64)
        self.lidar_to_ego = T_rig_from_lidar  # (4,4)

        ego_to_world_native = _load_rig_world_poses(
            str(scene_path / "calib" / "T_rig_world.parquet")
        )  # (T,4,4)
        ego_to_world_iso = ego_to_world_native  # already ISO 8855; no extra rotation
        # canonical_from_world @ world_from_ego @ ego_from_lidar = canonical_from_lidar
        l2ws = []
        for t in range(self.start_timestep, self.end_timestep):
            l2ws.append(self.inv_start @ ego_to_world_iso[t] @ self.lidar_to_ego)
        self.lidar_to_worlds = torch.from_numpy(np.stack(l2ws, axis=0)).float()

        # Also cache ego_to_world (canonical) so we can transform raw ego-frame points.
        e2ws = []
        for t in range(self.start_timestep, self.end_timestep):
            e2ws.append(self.inv_start @ ego_to_world_iso[t])
        self.ego_to_worlds = np.stack(e2ws, axis=0)  # (T_seg, 4, 4) np.float64

    def load_lidar(self):
        origins, directions, ranges, laser_ids, intensities = [], [], [], [], []
        timestamps, timesteps = [], []

        accumulated = 0
        accumulated_kept = 0
        for i, t_global in enumerate(trange(
            self.start_timestep, self.end_timestep,
            desc="Loading NVIDIA_AV_Fog LiDAR", dynamic_ncols=True,
        )):
            sweep = np.load(self.lidar_filepaths[i])  # (N, 6) float32
            xyz_ego = sweep[:, :3].astype(np.float32)
            intensity = sweep[:, 3].astype(np.float32)
            ring_id = sweep[:, 5].astype(np.int64)

            # ego → canonical_world (ISO-rotated, start-of-clip origin)
            E = self.ego_to_worlds[i].astype(np.float32)  # (4,4)
            R = E[:3, :3]
            tvec = E[:3, 3]
            xyz_world = xyz_ego @ R.T + tvec[None, :]
            n = xyz_world.shape[0]
            accumulated += n
            xyz_t = torch.from_numpy(xyz_world).float()

            # ray origin = lidar position in canonical world frame
            origin_xyz = self.lidar_to_worlds[i][:3, 3]
            origin_t = origin_xyz.unsqueeze(0).expand(n, -1).contiguous()

            dirs = xyz_t - origin_t
            rng = torch.norm(dirs, dim=-1, keepdim=True)
            safe = rng.clamp_min(1e-6)
            dirs = dirs / safe

            laser_id_t = torch.from_numpy(ring_id).long()
            intensity_t = torch.from_numpy(intensity).float()

            # only_use_top_lidar: this dataset has only one (top) lidar; no-op.
            if self.data_cfg.only_use_top_lidar:
                mask = torch.ones(n, dtype=torch.bool)
            else:
                mask = torch.ones(n, dtype=torch.bool)

            mn = self.data_cfg.truncated_min_range
            mx = self.data_cfg.truncated_max_range
            rng_flat = rng.squeeze(-1)
            mask = mask & (rng_flat > max(mn if mn is not None else 0.0, 0.5))
            if mx is not None:
                mask = mask & (rng_flat < mx)
            origin_t, dirs, rng = origin_t[mask], dirs[mask], rng[mask]
            laser_id_t, intensity_t = laser_id_t[mask], intensity_t[mask]
            n_kept = origin_t.shape[0]
            accumulated_kept += n_kept

            timestep_idx = t_global - self.start_timestep
            ts_t = torch.full((n_kept,), float(timestep_idx))

            origins.append(origin_t)
            directions.append(dirs)
            ranges.append(rng)
            laser_ids.append(laser_id_t)
            intensities.append(intensity_t)
            timestamps.append(ts_t)
            timesteps.append(ts_t.long())

        logger.info(
            f"[NVIDIA_AV_Fog-LiDAR] kept {accumulated_kept}/{accumulated} rays "
            f"({100.0*accumulated_kept/max(accumulated,1):.1f}%)"
        )
        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self.intensities = torch.cat(intensities, dim=0)
        self._timestamps = torch.cat(timestamps, dim=0).float()
        self._timesteps = torch.cat(timesteps, dim=0).long()

    # ------------------------------------------------------------------
    def to(self, device):
        self.device = device
        for name in ("origins", "directions", "ranges",
                     "_timestamps", "_timesteps", "_normalized_timestamps",
                     "lidar_to_worlds", "laser_ids", "intensities"):
            v = getattr(self, name, None)
            if isinstance(v, Tensor):
                setattr(self, name, v.to(device))
        return self

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def normalized_timestamps(self):
        return self._normalized_timestamps

    @property
    def unique_normalized_timestamps(self):
        return self._unique_normalized_timestamps

    @property
    def num_timesteps(self):
        return int(self.timesteps.unique().numel())

    def register_normalized_timestamps(self, normalized: Tensor):
        assert normalized.size(0) == self.origins.size(0)
        self._normalized_timestamps = normalized.to(self.device)
        self._unique_normalized_timestamps = self._normalized_timestamps.unique()

    def get_aabb(self):
        lidar_pts = self.origins + self.directions * self.ranges
        lidar_pts = lidar_pts[
            torch.randperm(len(lidar_pts))[
                : int(len(lidar_pts) / self.data_cfg.lidar_downsample_factor)
            ]
        ]
        aabb_min = torch.quantile(lidar_pts, self.data_cfg.lidar_percentile, dim=0)
        aabb_max = torch.quantile(lidar_pts, 1 - self.data_cfg.lidar_percentile, dim=0)
        if aabb_max[-1] < 20:
            aabb_max[-1] = 20.0
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[NVIDIA_AV_Fog-LiDAR] Auto AABB from LiDAR: {aabb}")
        return aabb

    # ------------------------------------------------------------------
    def sample_uniform_rays(self, num_rays: int, candidate_indices=None):
        if candidate_indices is None:
            return torch.randint(0, len(self.origins), size=(num_rays,), device=self.device)
        if not isinstance(candidate_indices, Tensor):
            candidate_indices = torch.tensor(candidate_indices, device=self.device)
        if self.cached_indices is None or not torch.equal(candidate_indices, self.cached_indices):
            self.cached_indices = candidate_indices
            mask = self.timesteps.new_zeros(self.timesteps.size(0), dtype=torch.bool)
            for idx in self.cached_indices:
                mask |= self.timesteps == idx
            self.cached_origins = self.origins[mask]
            self.cached_directions = self.directions[mask]
            self.cached_ranges = self.ranges[mask]
            self.cached_normalized_timestamps = self.normalized_timestamps[mask]
        return torch.randint(0, len(self.cached_origins), size=(num_rays,), device=self.device)

    def get_train_rays(self, num_rays: int, candidate_indices=None):
        idx = self.sample_uniform_rays(num_rays=num_rays, candidate_indices=candidate_indices)
        return {
            "lidar_origins": self.cached_origins[idx] if self.cached_origins is not None else self.origins[idx],
            "lidar_viewdirs": self.cached_directions[idx] if self.cached_directions is not None else self.directions[idx],
            "lidar_ranges": self.cached_ranges[idx] if self.cached_ranges is not None else self.ranges[idx],
            "lidar_normed_timestamps": (
                self.cached_normalized_timestamps[idx] if self.cached_normalized_timestamps is not None
                else self.normalized_timestamps[idx]
            ),
        }

    def get_render_rays(self, time_idx: int):
        mask = self.timesteps == time_idx
        return {
            "lidar_origins": self.origins[mask],
            "lidar_viewdirs": self.directions[mask],
            "lidar_ranges": self.ranges[mask],
            "lidar_normed_timestamps": self.normalized_timestamps[mask],
        }


# ===========================================================================
# Top-level dataset
# ===========================================================================
class NVIDIAAVFogDataset(SceneDataset):
    dataset: str = "nvidia_av_fog"

    def __init__(self, data_cfg: OmegaConf) -> None:
        super().__init__(data_cfg)
        # scene_idx is a string (or int) prefix like "002"; resolve full clip dir.
        self.data_path = _resolve_clip_dir(self.data_cfg.data_root, self.data_cfg.scene_idx)
        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"

        # Count frames via the front_wide camera dir (199 frames per clip nominally).
        num_frames = len([
            f for f in os.listdir(
                os.path.join(self.data_path, "images", "camera_front_wide_120fov")
            ) if f.endswith(".jpg")
        ])
        if self.data_cfg.end_timestep == -1:
            end_timestep = num_frames - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep

        self.pixel_source = NVIDIAAVFogPixelSource(
            self.data_cfg.pixel_source,
            self.data_path,
            self.start_timestep,
            self.end_timestep,
            device=self.data_cfg.preload_device,
        )
        self.pixel_source.to(self.data_cfg.preload_device)

        load_lidar = bool(getattr(self.data_cfg.lidar_source, "load_lidar", False))
        if load_lidar:
            self.lidar_source = NVIDIAAVFogLiDARSource(
                self.data_cfg.lidar_source,
                self.data_path,
                self.start_timestep,
                self.end_timestep,
                inv_start_ego_to_world=self.pixel_source.inv_start_ego_to_world,
                device=self.data_cfg.preload_device,
            )
            self.lidar_source.to(self.data_cfg.preload_device)
        else:
            self.lidar_source = None

        # Joint timestamp normalization (pixel + lidar share [tmin, tmax]).
        ts_pix = self.pixel_source.timestamps.float()
        if self.lidar_source is not None:
            ts_lid = self.lidar_source.timestamps.float()
            tmin = torch.minimum(ts_pix.min(), ts_lid.min())
            tmax = torch.maximum(ts_pix.max(), ts_lid.max())
        else:
            tmin, tmax = ts_pix.min(), ts_pix.max()
        span = (tmax - tmin).clamp(min=1e-6)
        self.pixel_source.register_normalized_timestamps((ts_pix - tmin) / span)
        if self.lidar_source is not None:
            self.lidar_source.register_normalized_timestamps((ts_lid - tmin) / span)

        self.aabb = self.get_aabb()

        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()

        pixel_sets, lidar_sets = self.build_split_wrapper()
        self.train_pixel_set, self.test_pixel_set, self.full_pixel_set = pixel_sets
        self.train_lidar_set, self.test_lidar_set, self.full_lidar_set = lidar_sets

    # ------------------------------------------------------------------
    def build_data_source(self):
        return self.pixel_source, self.lidar_source

    def build_split_wrapper(self):
        train_pixel_set = SplitWrapper(
            datasource=self.pixel_source,
            split_indices=self.train_indices,
            split="train",
            ray_batch_size=self.data_cfg.ray_batch_size,
        )
        test_pixel_set = None
        if len(self.test_indices) > 0:
            test_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                split_indices=self.test_indices,
                split="test",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
        full_pixel_set = SplitWrapper(
            datasource=self.pixel_source,
            split_indices=np.arange(self.pixel_source.num_imgs),
            split="full",
            ray_batch_size=self.data_cfg.ray_batch_size,
        )
        pixel_sets = (train_pixel_set, test_pixel_set, full_pixel_set)

        if self.lidar_source is None:
            lidar_sets = (None, None, None)
        else:
            train_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                split_indices=self.train_timesteps.tolist(),
                split="train",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            test_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                split_indices=self.test_timesteps.tolist(),
                split="test",
                ray_batch_size=self.data_cfg.ray_batch_size,
            ) if len(self.test_timesteps) > 0 else None
            full_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                split_indices=np.arange(self.start_timestep, self.end_timestep).tolist(),
                split="full",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            lidar_sets = (train_lidar_set, test_lidar_set, full_lidar_set)
        return pixel_sets, lidar_sets

    def split_train_test(self):
        """Timestep-level train/eval split.

        To be byte-for-byte consistent with the SplatAD / NeuRAD baselines
        (nerfstudio ``nvs_50_50`` protocol), we read the dataset's OFFICIAL split
        file ``processed_pinhole/splits/nvs_50_50.json`` directly when present.
        Its ``train_indices`` / ``eval_indices`` are timestep indices (even /
        odd for the 50/50 split), so all three pipelines hold out the SAME
        frames and the metrics are comparable. We fall back to a stride split
        (holding out ODD frames, matching the nvs_50_50 eval parity) only if the
        JSON is absent.
        """
        num_t = self.end_timestep - self.start_timestep
        all_t = np.arange(num_t)
        split_json = os.path.join(self.data_path, "splits", "nvs_50_50.json")
        if os.path.exists(split_json):
            with open(split_json, "r") as f:
                split = json.load(f)
            # JSON values are absolute timesteps; shift into this clip's window.
            train_t = np.asarray(split["train_indices"], dtype=np.int64) - self.start_timestep
            test_t = np.asarray(split["eval_indices"], dtype=np.int64) - self.start_timestep
            train_t = train_t[(train_t >= 0) & (train_t < num_t)]
            test_t = test_t[(test_t >= 0) & (test_t < num_t)]
        else:
            stride = self.data_cfg.pixel_source.test_image_stride
            if stride and stride > 0:
                # hold out ODD frames {1,3,...} to match nvs_50_50 eval parity
                test_t = all_t[1::stride]
                train_t = np.setdiff1d(all_t, test_t)
            else:
                test_t = np.array([], dtype=np.int64)
                train_t = all_t
        n_cams = self.pixel_source.num_cams
        train_indices = (
            np.concatenate([np.arange(t * n_cams, (t + 1) * n_cams) for t in train_t])
            if len(train_t) else np.array([], dtype=np.int64)
        )
        test_indices = (
            np.concatenate([np.arange(t * n_cams, (t + 1) * n_cams) for t in test_t])
            if len(test_t) else np.array([], dtype=np.int64)
        )
        return (torch.from_numpy(train_t).long(),
                torch.from_numpy(test_t).long(),
                train_indices.tolist(),
                test_indices.tolist())


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    """Smoke test.

    Note: this module's file-level `from datasets.base.pixel_source import ...`
    triggers `datasets/__init__.py`, which transitively imports tcnn — requires
    a CUDA-capable PyTorch. For CPU-only smoke testing (e.g. on a login node)
    use the sibling launcher `datasets/_nvidia_av_fog_smoketest.py` which stubs
    out the `datasets` package init before importing. Otherwise:

        conda activate splatad-blackwell
        export PYTHONNOUSERSITE=1
        cd /networkhome/WMGDS/wang3_y/ETH/Code/EmerNeRF
        PYTHONPATH=. python datasets/_nvidia_av_fog_smoketest.py
    """
    import sys
    logging.basicConfig(level=logging.INFO)

    cfg = OmegaConf.create({
        "data_root": "/networkhome/WMGDS/wang3_y/ETH/Dataset/NVIDIA_AV_Fog",
        "dataset": "nvidia_av_fog",
        "scene_idx": "002",
        "start_timestep": 0,
        "end_timestep": 4,             # tiny window for smoke test
        "ray_batch_size": 4096,
        "preload_device": "cpu",
        "pixel_source": {
            "load_size": [960, 1280],
            "downscale": 1,
            "num_cams": 7,
            "test_image_stride": 2,
            "load_rgb": True,
            "load_sky_mask": False,
            "load_dynamic_mask": False,
            "load_features": False,
            "skip_feature_extraction": True,
            "target_feature_dim": 64,
            "feature_model_type": "dinov2_vitb14",
            "feature_extraction_stride": 7,
            "feature_extraction_size": [644, 966],
            "delete_features_after_run": False,
            "sampler": {
                "buffer_downscale": 16,
                "buffer_ratio": 0.0,   # skip building error buffer for smoke
            },
        },
        "lidar_source": {
            "load_lidar": True,
            "only_use_top_lidar": False,
            "truncated_max_range": 80,
            "truncated_min_range": -2,
            "lidar_downsample_factor": 4,
            "lidar_percentile": 0.02,
        },
    })

    ds = NVIDIAAVFogDataset(data_cfg=cfg)

    print("====== SMOKE TEST RESULTS ======")
    print(f"num_cams                  = {ds.num_cams}")
    print(f"num_img_timesteps         = {ds.num_img_timesteps}")
    print(f"num_train_timesteps       = {ds.num_train_timesteps}")
    print(f"num_test_timesteps        = {ds.num_test_timesteps}")
    print(f"pixel_source.num_imgs     = {ds.pixel_source.num_imgs}")
    print(f"pixel_source.images shape = {tuple(ds.pixel_source.images.shape)}")
    print(f"sample image[0] shape     = {tuple(ds.pixel_source.images[0].shape)}")
    print(f"intrinsics[0]             =\n{ds.pixel_source.intrinsics[0]}")
    print(f"cam_to_worlds[0]          =\n{ds.pixel_source.cam_to_worlds[0]}")
    print(f"AABB                      = {ds.aabb.tolist()}")
    if ds.lidar_source is not None:
        n = ds.lidar_source.origins.shape[0]
        print(f"lidar.origins shape       = {tuple(ds.lidar_source.origins.shape)}")
        print(f"lidar.directions shape    = {tuple(ds.lidar_source.directions.shape)}")
        print(f"lidar.ranges shape        = {tuple(ds.lidar_source.ranges.shape)}")
        print(f"lidar.num_timesteps       = {ds.lidar_source.num_timesteps}")
        # transform a few ego-frame points back to verify
        pts = ds.lidar_source.origins[:1] + ds.lidar_source.directions[:1] * ds.lidar_source.ranges[:1]
        print(f"sample world pt           = {pts.numpy().tolist()}")
    print("================================")
    sys.exit(0)
