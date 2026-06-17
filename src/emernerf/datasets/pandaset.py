"""PandaSet dataparser for EmerNeRF.

Mirrors datasets/waymo.py but reads from PandaSet's native layout at
~/ETH/Code/SplatAD/dataset/<scene>/{camera, lidar, meta, annotations}.

Coordinate convention: PandaSet camera poses are stored per-camera per-frame in
the dataset's "world" frame (geo-referenced ECEF-ish). We pick the front camera
as the ego reference and reconstruct:
  - ego_pose[t]   = front_camera_world_pose[t]              (4x4)
  - cam_to_ego[i] = inv(front_world_t0) @ cam_i_world_t0    (4x4, time-invariant)
EmerNeRF's loader then post-multiplies cam_to_ego by OPENCV2DATASET to flip
OpenCV-cam axes (x-right, y-down, z-fwd) → dataset axes (x-fwd, y-left, z-up).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import trange

from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper

logger = logging.getLogger()


PANDASET_CAMERAS_ALL = [
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
]


def _quat_pos_to_4x4(qw: float, qx: float, qy: float, qz: float,
                     px: float, py: float, pz: float) -> np.ndarray:
    """Convert a PandaSet pose entry (quaternion heading + position) to a 4x4 SE(3)."""
    norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
        [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = [px, py, pz]
    return M


def _load_camera_meta(scene_path: Path, cam_name: str):
    cam_dir = scene_path / "camera" / cam_name
    with open(cam_dir / "intrinsics.json") as f:
        K = json.load(f)
    with open(cam_dir / "poses.json") as f:
        poses_raw = json.load(f)
    with open(cam_dir / "timestamps.json") as f:
        ts = json.load(f)
    poses = []
    for p in poses_raw:
        h, q = p["heading"], p["position"]
        poses.append(_quat_pos_to_4x4(h["w"], h["x"], h["y"], h["z"], q["x"], q["y"], q["z"]))
    poses = np.stack(poses, axis=0)  # (T, 4, 4)
    return K, poses, ts


class PandaSetPixelSource(ScenePixelSource):
    # PandaSet front/back are 1080x1920, sides are 1080x1920 too — uniform
    ORIGINAL_SIZE = [[1080, 1920] for _ in range(6)]
    # PandaSet uses standard OpenCV cam frame (x right, y down, z fwd)
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
        # PandaSet has no per-frame dynamic masks or sky masks shipped in usable form
        pixel_data_config.load_dynamic_mask = False
        pixel_data_config.load_sky_mask = False
        super().__init__(pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        # ---- camera list selection mirrors Waymo's num_cams convention ---- #
        # Use the front 5 cams when num_cams==5 (drop back); front 3 / 1 follow.
        if self.num_cams == 1:
            self.camera_list = [0]                  # front
            cam_names = ["front_camera"]
        elif self.num_cams == 3:
            self.camera_list = [1, 0, 2]            # FL, F, FR  (Waymo-order: left, front, right)
            cam_names = ["front_left_camera", "front_camera", "front_right_camera"]
        elif self.num_cams == 5:
            self.camera_list = [3, 1, 0, 2, 4]      # L, FL, F, FR, R
            cam_names = ["left_camera", "front_left_camera", "front_camera", "front_right_camera", "right_camera"]
        elif self.num_cams == 6:
            self.camera_list = [3, 1, 0, 2, 4, 5]
            cam_names = ["left_camera", "front_left_camera", "front_camera", "front_right_camera", "right_camera", "back_camera"]
        else:
            raise NotImplementedError(f"num_cams={self.num_cams} not supported for pandaset")
        # store the resolved physical camera names in the slot order EmerNeRF uses
        self._cam_names = cam_names
        # build_per_cam index → cam_name map (the position in self.camera_list IS the dataset cam_id)
        # For our purposes we just iterate per slot.

        img_filepaths, dynamic_mask_filepaths, sky_mask_filepaths, feat_filepaths = [], [], [], []
        scene_path = Path(self.data_path)
        for t in range(self.start_timestep, self.end_timestep):
            for slot_idx, cam_name in enumerate(cam_names):
                img_filepaths.append(str(scene_path / "camera" / cam_name / f"{t:02d}.jpg"))
                # PandaSet has no dynamic mask files; emit non-existent paths (loader will skip when load_dynamic_mask=False)
                dynamic_mask_filepaths.append("")
                sky_mask_filepaths.append("")
                feat_filepaths.append("")
        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.feat_filepaths = np.array(feat_filepaths)

    def load_calibrations(self):
        # Read per-camera intrinsics + per-camera per-frame poses (in PandaSet world frame).
        scene_path = Path(self.data_path)
        cam_meta = {}  # cam_name → (K, poses[T,4,4])
        for cam_name in PANDASET_CAMERAS_ALL:
            K, poses, _ts = _load_camera_meta(scene_path, cam_name)
            cam_meta[cam_name] = (K, poses)

        # 1) Build per-camera intrinsics in (3,3) form, scaled to load_size.
        _intrinsics = []
        for slot_idx, cam_name in enumerate(self._cam_names):
            K = cam_meta[cam_name][0]
            fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
            fx *= self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[slot_idx][1]
            fy *= self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[slot_idx][0]
            cx *= self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[slot_idx][1]
            cy *= self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[slot_idx][0]
            _intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64))

        # 2) Pick the front_camera at start_timestep as the ego frame.
        front_world = cam_meta["front_camera"][1]   # (T, 4, 4)
        ego_to_world_start = front_world[self.start_timestep]
        inv_start = np.linalg.inv(ego_to_world_start)

        # 3) For each slot camera, compute cam_to_ego (time-invariant under rigid mount).
        cam_to_egos_raw = []
        for cam_name in self._cam_names:
            cam_world_t0 = cam_meta[cam_name][1][self.start_timestep]
            cam_to_ego = inv_start @ cam_world_t0
            cam_to_egos_raw.append(cam_to_ego)
        # PandaSet stores c2w with camera frame already in OpenCV convention
        # (x-right, y-down, z-fwd), which is what EmerNeRF's get_rays() expects.
        # Do NOT post-multiply by OPENCV2DATASET — verified by motion-axis test.
        cam_to_egos = cam_to_egos_raw

        # 5) Per-frame ego_to_world (front_camera_world_pose, rebased to start frame).
        ego_to_worlds, cam_to_worlds, intrinsics, cam_ids, timestamps, timesteps = [], [], [], [], [], []
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_t = inv_start @ front_world[t]
            ego_to_worlds.append(ego_to_world_t)
            for slot_idx in range(len(self._cam_names)):
                cam_ids.append(self.camera_list[slot_idx])
                cam2world = ego_to_world_t @ cam_to_egos[slot_idx]
                cam_to_worlds.append(cam2world)
                intrinsics.append(_intrinsics[slot_idx])
                timestamps.append(t - self.start_timestep)
                timesteps.append(t - self.start_timestep)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        self.cam_ids = torch.from_numpy(np.array(cam_ids, dtype=np.int64)).long()
        self._timestamps = torch.from_numpy(np.array(timestamps, dtype=np.float32)).float()
        self._timesteps = torch.from_numpy(np.array(timesteps, dtype=np.int64)).long()

        # Save the PandaSet-world → canonical (front_camera@t=start) transform so the
        # LiDAR source can rebase its points/poses to the exact same canonical frame
        # without recomputing (and without float drift).
        self.inv_start_ego_to_world = inv_start


class PandaSetLiDARSource:
    """PandaSet LiDAR data parser, mirrors WaymoLiDARSource.

    PandaSet's .pkl.gz lidar files store points **already in PandaSet world frame**
    (sea-level origin, not lidar-local). lidar/poses.json gives the per-frame
    lidar-to-world pose in that same world frame. We rebase both to our canonical
    front-camera@t=start frame so coordinates match the pixel source.
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
        self.inv_start = inv_start_ego_to_world  # (4,4) np.float64
        # placeholders populated by load_data()
        self.origins = self.directions = self.ranges = None
        self.lidar_to_worlds = None
        self._timestamps = self._timesteps = self._normalized_timestamps = None
        self.cached_origins = self.cached_directions = self.cached_ranges = None
        self.cached_normalized_timestamps = None
        self.cached_indices = None
        self._unique_normalized_timestamps = None
        # optional bookkeeping
        self.laser_ids = self.intensities = None
        self.create_all_filelist()
        self.load_data()

    # ------------------------------------------------------------------
    # SceneLidarSource-compatible API surface (we deliberately don't inherit
    # from SceneLidarSource because its abstractmethod set is small and
    # mirroring lets us drop dependence on Waymo-specific fields like flows).
    # ------------------------------------------------------------------

    def create_all_filelist(self):
        self.lidar_filepaths = np.array([
            os.path.join(self.data_path, "lidar", f"{t:02d}.pkl.gz")
            for t in range(self.start_timestep, self.end_timestep)
        ])

    def load_data(self):
        self.load_calibrations()
        self.load_lidar()
        logger.info("[PandaSet-LiDAR] All Lidar Data loaded.")

    def load_calibrations(self):
        with open(os.path.join(self.data_path, "lidar", "poses.json")) as f:
            raw_poses = json.load(f)
        l2ws = []
        for t in range(self.start_timestep, self.end_timestep):
            p = raw_poses[t]
            T_world = _quat_pos_to_4x4(
                p["heading"]["w"], p["heading"]["x"], p["heading"]["y"], p["heading"]["z"],
                p["position"]["x"], p["position"]["y"], p["position"]["z"],
            )
            l2ws.append(self.inv_start @ T_world)
        self.lidar_to_worlds = torch.from_numpy(np.stack(l2ws, axis=0)).float()

    def load_lidar(self):
        import gzip, pickle

        origins, directions, ranges, laser_ids, intensities = [], [], [], [], []
        timestamps, timesteps = [], []

        accumulated = 0
        accumulated_kept = 0
        R = self.inv_start[:3, :3].astype(np.float32)
        t_vec = self.inv_start[:3, 3].astype(np.float32)
        for i, t_global in enumerate(trange(
            self.start_timestep, self.end_timestep, desc="Loading PandaSet LiDAR", dynamic_ncols=True,
        )):
            with gzip.open(self.lidar_filepaths[i], "rb") as f:
                df = pickle.load(f)
            xyz_world = df[["x", "y", "z"]].to_numpy().astype(np.float32)  # (N,3)
            xyz_canon = xyz_world @ R.T + t_vec[None, :]                    # (N,3)
            n = xyz_canon.shape[0]
            accumulated += n
            xyz_t = torch.from_numpy(xyz_canon).float()

            origin_xyz = self.lidar_to_worlds[i][:3, 3]                     # (3,)
            origin_t = origin_xyz.unsqueeze(0).expand(n, -1).contiguous()

            dirs = xyz_t - origin_t
            rng = torch.norm(dirs, dim=-1, keepdim=True)
            safe = rng.clamp_min(1e-6)
            dirs = dirs / safe

            laser_id_t = torch.from_numpy(df["d"].to_numpy()).float()
            intensity_t = torch.from_numpy(df["i"].to_numpy()).float()

            if self.data_cfg.only_use_top_lidar:
                mask = laser_id_t == 1  # PandaSet d=1 → pandar64 top
            else:
                mask = torch.ones(n, dtype=torch.bool)
            # range-based truncation (frame-invariant)
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

            timestep_idx = t_global - self.start_timestep  # match pixel-source convention
            ts_t = torch.full((n_kept,), float(timestep_idx))

            origins.append(origin_t)
            directions.append(dirs)
            ranges.append(rng)
            laser_ids.append(laser_id_t)
            intensities.append(intensity_t)
            timestamps.append(ts_t)
            timesteps.append(ts_t.long())

        logger.info(
            f"[PandaSet-LiDAR] kept {accumulated_kept}/{accumulated} rays "
            f"({100.0*accumulated_kept/accumulated:.1f}%)"
        )
        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self.intensities = torch.cat(intensities, dim=0)
        self._timestamps = torch.cat(timestamps, dim=0).float()
        self._timesteps = torch.cat(timesteps, dim=0).long()

    def to(self, device):
        self.device = device
        for name in ("origins", "directions", "ranges",
                     "_timestamps", "_timesteps", "_normalized_timestamps",
                     "lidar_to_worlds", "laser_ids", "intensities"):
            v = getattr(self, name, None)
            if isinstance(v, Tensor):
                setattr(self, name, v.to(device))
        return self

    # ---- properties used by EmerNeRF base API ----

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
        logger.info(f"[PandaSet-LiDAR] Auto AABB from LiDAR: {aabb}")
        return aabb

    # ---- ray sampling (mirrors SceneLidarSource.sample_uniform_rays + get_train_rays + get_render_rays) ----

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


class PandaSetDataset(SceneDataset):
    dataset: str = "pandaset"

    def __init__(self, data_cfg: OmegaConf) -> None:
        super().__init__(data_cfg)
        # PandaSet scenes are named with 3-digit ids (011, 078, ...); data_cfg.scene_idx can be int.
        scene_id = self.data_cfg.scene_idx
        if isinstance(scene_id, int):
            scene_id_str = f"{scene_id:03d}"
        else:
            scene_id_str = str(scene_id).zfill(3)
        self.data_path = os.path.join(self.data_cfg.data_root, scene_id_str)
        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"

        # PandaSet scenes have 80 frames @ 10 Hz (some have 83). Count via front_camera dir.
        num_frames = len([f for f in os.listdir(os.path.join(self.data_path, "camera", "front_camera")) if f.endswith(".jpg")])
        if self.data_cfg.end_timestep == -1:
            end_timestep = num_frames - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep

        # Build pixel source.
        self.pixel_source = PandaSetPixelSource(
            self.data_cfg.pixel_source,
            self.data_path,
            self.start_timestep,
            self.end_timestep,
            device=self.data_cfg.preload_device,
        )
        self.pixel_source.to(self.data_cfg.preload_device)

        # Optional LiDAR source (PandaSet has 2 sensors: pandar64 top + pandar GT front).
        # Required when supervision.depth.enable=True (EmerNeRF paper default).
        load_lidar = bool(getattr(self.data_cfg.lidar_source, "load_lidar", False))
        if load_lidar:
            self.lidar_source = PandaSetLiDARSource(
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

        # Register normalized timestamps. Pixel + lidar must share the same
        # (tmin, tmax) so model time-axis is consistent across modalities.
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

    def build_data_source(self):
        # not used (handled inline in __init__) but the abstract method needs to exist
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
        # EmerNeRF semantics: timestep-level split. Mirrors waymo.py.
        # SplatAD/NeuRAD paper recipe → train_split_fraction=0.5 (every OTHER frame for training).
        # In EmerNeRF that's test_image_stride=2 (every 2nd frame is test).
        stride = self.data_cfg.pixel_source.test_image_stride
        num_t = self.end_timestep - self.start_timestep
        all_t = np.arange(num_t)
        if stride and stride > 0:
            test_t = all_t[::stride]
            train_t = np.setdiff1d(all_t, test_t)
        else:
            test_t = np.array([], dtype=np.int64)
            train_t = all_t
        # img indices = timestep * num_cams + cam_slot ; we expand
        n_cams = self.pixel_source.num_cams
        train_indices = np.concatenate([np.arange(t * n_cams, (t + 1) * n_cams) for t in train_t]) if len(train_t) else np.array([], dtype=np.int64)
        test_indices = np.concatenate([np.arange(t * n_cams, (t + 1) * n_cams) for t in test_t]) if len(test_t) else np.array([], dtype=np.int64)
        return (torch.from_numpy(train_t).long(),
                torch.from_numpy(test_t).long(),
                train_indices.tolist(),
                test_indices.tolist())
