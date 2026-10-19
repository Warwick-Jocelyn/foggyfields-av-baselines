# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data parser for the NVIDIA_AV_Fog dataset (5 fog-daytime clips derived from NVIDIA NCore).

Two on-disk layouts are supported under ``--data`` (see ``_resolve_clip_root``)::

    <data>/clips/NFF-01__b9ea7a9d/{images,lidar,labels,splits,calib,videos,...}   # live
    <data>/NVIDIA_FOGGY_Field/clips/NFF-01__b9ea7a9d/...                          # live, parent root
    <data>/processed/002_<uuid>_fog_day_rural/processed_pinhole/...               # legacy

Use ``--sequence`` to pick a clip: ``NFF-01``, ``b9ea7a9d``, the full directory name, or
-- legacy layout only -- a leading 3-digit prefix such as ``002``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import pandas as pd
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

# Camera order is fixed; sensor_idx is the index into this tuple.
NVIDIA_AV_FOG_CAMERAS: Tuple[str, ...] = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_front_tele_30fov",
    "camera_rear_tele_30fov",
)

NVIDIA_AV_FOG_LIDAR: str = "lidar_top_360fov"

# The seven cameras are NOT exposed simultaneously. They split into two trigger groups
# with a large, systematic offset from the LiDAR spin centre (measured on the released
# clips, see DATASET_NOTES.md "F1"):
#   group A: front_wide, rear_left, rear_right, front_tele
#   group B: cross_left, cross_right, rear_tele
# Intra-group jitter is +-0.1 ms; the inter-group gap reaches 16.9 ms (NFF-01), which is
# 15.5 cm of ego motion at 32.9 km/h. Never reuse the LiDAR timestamp for a camera.
NVIDIA_AV_FOG_CAMERA_TRIGGER_GROUPS: Tuple[Tuple[str, ...], Tuple[str, ...]] = (
    ("camera_front_wide_120fov", "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_front_tele_30fov"),
    ("camera_cross_left_120fov", "camera_cross_right_120fov", "camera_rear_tele_30fov"),
)

# NCore native rig: +x=left, +y=forward, +z=up.
# ISO 8855:        +x=forward, +y=left,    +z=up.
# Rotation that takes a point's NCore-frame coords to ISO coords:
#     p_iso = R_iso_from_ncore @ p_ncore
# Derivation:
#     x_iso (fwd)  = y_ncore (fwd)
#     y_iso (left) = x_ncore (left)
#     z_iso (up)   = z_ncore (up)
R_ISO_FROM_NCORE = np.array(
    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

# NCore LiDAR sweep `[N,6]` columns: x, y, z, intensity (already 0-1), t_offset_us, ring_id.
LIDAR_SWEEP_COL_X = 0
LIDAR_SWEEP_COL_INTENSITY = 3
LIDAR_SWEEP_COL_T_OFFSET_US = 4
LIDAR_SWEEP_COL_RING_ID = 5

# Allowed NCore obstacle classes. NCore auto-labels v2 surface these for the fog clips:
#   automobile, trailer, heavy_truck, bus, motorcycle, bicycle, ... (pedestrian/two-wheeler rare)
NVIDIA_AV_FOG_ALLOWED_RIGID_CLASSES: Tuple[str, ...] = (
    "automobile",
    "trailer",
    "heavy_truck",
    "bus",
    "other_vehicle",
    "motorcycle",
    "bicycle",
    "train",
    "trolley",
    "tram",
)
NVIDIA_AV_FOG_ALLOWED_DEFORMABLE_CLASSES: Tuple[str, ...] = (
    "pedestrian",
    "person",
    "two_wheeler",
)

# Sensor specs for the NCore 128-line top LiDAR. Beam divergences are not published
# by NVIDIA; we reuse the Pandar64 values as conservative defaults. They only matter
# for `add_missing_points` (which is off by default) and for ray-drop modeling.
HORIZONTAL_BEAM_DIVERGENCE = 3e-3
VERTICAL_BEAM_DIVERGENCE = 1.5e-3


@dataclass
class NvidiaAvFogDataParserConfig(ADDataParserConfig):
    """Config for the NVIDIA_AV_Fog dataparser.

    The dataset is the user's 5-clip foggy multimodal release derived from NVIDIA
    NCore / PhysicalAI-AV. 7 rectified-pinhole cameras + one 128-line spinning LiDAR,
    199 timesteps over ~20s per clip. See ``KNOWN_ISSUES.md`` in the dataset for the
    NCore-rig convention (this dataparser converts to ISO 8855).
    """

    _target: Type = field(default_factory=lambda: NvidiaAvFog)
    data: Path = Path("/home/wang3_y_WMGDS.WMG.WARWICK.AC.UK/Dataset/physicalai_av/NVIDIA_FOGGY_Field")
    """Dataset root. Either a subset root holding ``clips/`` (live layout), a parent holding
    ``NVIDIA_FOGGY_Field/clips`` and/or ``NVIDIA_CLEAR_Field/clips``, or a legacy root
    holding ``processed/<clip>/processed_pinhole``."""
    sequence: str = "NFF-01"
    """Clip selector: ``NFF-01`` / ``NCF-08``, the hash, the full directory name, or a
    legacy 3-digit prefix such as ``002``."""

    cameras: Tuple[
        Literal[
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
            "camera_front_tele_30fov",
            "camera_rear_tele_30fov",
            "none",
            "all",
        ],
        ...,
    ] = NVIDIA_AV_FOG_CAMERAS
    """Which cameras to use. ``("all",)`` expands to all 7."""

    lidars: Tuple[Literal["lidar_top_360fov", "none"], ...] = (NVIDIA_AV_FOG_LIDAR,)
    """Which lidars to use. Currently only the 360° top lidar is present."""

    annotation_interval: float = 0.1
    """Interval between annotations in seconds (10 Hz from LiDAR-driven sync)."""

    min_lidar_dist: Tuple[float, float, float] = (1.0, 2.0, 2.0)
    """Drop points closer than this (sensor frame). Conservative defaults for ego."""

    # ---- F2: rolling shutter ------------------------------------------------------
    rolling_shutter_source: Literal["calib", "config"] = "calib"
    """``calib`` (default) reads the PUBLISHED rolling-shutter readout time from
    ``calib/camera_intrinsics_ftheta.json``: every camera of every clip, fog and clear,
    carries ``"shutter_type": "ROLLING_TOP_TO_BOTTOM"`` and ``"shutter_delay_us": 30559``
    (checked: 224/224 camera-clip entries). That is 30.559 ms of readout inside a 33.331 ms
    frame period -- the frame is very nearly always mid-readout.

    The original config set ``rolling_shutter_time = 0.0`` with a comment claiming the
    value was "not published". It is published; the comment was wrong.

    ``config`` uses ``rolling_shutter_time`` below verbatim; set both to reproduce the
    superseded global-shutter runs (``--rolling-shutter-source config
    --rolling-shutter-time 0.0``)."""

    rolling_shutter_time: float = 0.0
    """Only consulted when ``rolling_shutter_source == "config"``. With ``calib`` this
    field is overwritten at parse time by the published value (0.030559 s)."""

    time_to_center_pixel: float = 0.0
    """Offset from the stored image timestamp to the time the CENTRE row is read out.

    DELIBERATELY LEFT AT 0, AND THIS IS AN OPEN QUESTION -- see the PR body.
    nerfstudio's ray generator computes
        time_offset(row) = (row / height - 0.5) * rolling_shutter_time + time_to_center_pixel
    so a non-zero value is needed exactly when the dataset's ``timestamp`` column means
    the START of the frame rather than its CENTRE. We could not establish which it is:
    the dataset documents neither, and NVIDIA's DriveWorks image-timestamp struct carries
    BOTH a start-of-frame field and a frame-midpoint field, so which one the NCore export
    wrote is a converter detail that is not visible from the released files.

    If you establish that ``timestamp`` is start-of-frame, set this to
    ``shutter_delay_us / 2 * 1e-6 = 0.0152795``. Leaving it at 0 is the correct choice
    under the centre-of-frame reading and is a half-readout (15.3 ms) error under the
    other -- which would be comparable to F1, so it is worth resolving."""

    apply_iso_8855_rotation: bool = False
    """If True, apply ``R_iso_from_ncore`` to all world-frame poses (cam, lidar, cuboid).
    The processed_pinhole/ data is already in ISO 8855 (T_rig_world translation
    grows along world +x = forward; obstacles_3d_world is in the same world frame),
    so the default is now False. KNOWN_ISSUES I-003 refers to obstacles_3d_*ego*.parquet
    only — we read the _world_ variant."""

    splits_json: Literal["reconstruction_all", "nvs_50_50", "nvs_80_20", "nvs_90_10", "auto"] = "auto"
    """If not 'auto', read ``processed_pinhole/splits/<name>.json`` directly. Otherwise
    defer to the parent class' LINSPACE split (which matches ``nvs_50_50`` for
    ``train_split_fraction=0.5``)."""

    include_deformable_actors: bool = False

    # ---- F3: LiDAR point frame ----------------------------------------------------
    lidar_points_frame: Literal["auto", "ego", "sensor"] = "auto"
    """Frame of the xyz columns in ``lidar/sweep_*.npy``.

    ``auto`` (default) READS ``provenance.lidar_frame`` from the clip instead of trusting a
    config default, because the answer differs per clip and is being changed in place:
    the clear clips are already converted to true ego frame and marked ``"ego"``, the fog
    clips are mid-conversion -- some marked ``"ego"``, the rest not marked at all and still
    in the SENSOR frame.

    Getting this wrong is a 1.94 m vertical error: the old default ``"ego"`` applied
    inv(T_lidar_rig) to points that were already in the sensor frame, putting the
    world-frame ground at -1.94 m instead of 0. That is what the published NeuRAD /
    SplatAD numbers were trained on.

    ``ego`` / ``sensor`` force the answer and override the marker (with a warning) -- use
    them for clips that predate the marker."""

    allow_unmarked_lidar_frame: bool = False
    """Opt-in for clips with no ``provenance.lidar_frame``: treat them as legacy SENSOR
    frame. Off by default so an unmarked clip can never be loaded on a guess; the error
    names this flag."""

    check_lidar_frame_geometry: bool = True
    """Cross-check the declared frame against the measured ground plane of one sweep and
    warn if the other hypothesis fits ~1.9 m better. Costs one ``np.load``."""

    camera_timestamp_source: Literal["auto", "per_camera", "lidar"] = "auto"
    """Where the per-image capture time comes from.

    ``per_camera`` (and ``auto`` when the data supports it) reads the REAL per-camera
    timestamps, in this order:
      1. ``calib/camera_timestamps.parquet`` (``timestep_idx`` + one column per camera),
      2. ``videos/<cam>.timestamps.parquet`` column ``timestamp``, matched to each LiDAR
         spin centre by nearest neighbour (exactly the frame-selection rule the dataset
         converter used to pick the rectified frame).
    ``lidar`` reproduces the ORIGINAL, DEFECTIVE behaviour -- every camera inherits the
    LiDAR spin-centre timestamp. Keep it only to re-create pre-fix runs on purpose; it
    costs up to 16.9 ms of timing error (~15.5 cm of ego motion at 32.9 km/h) and it also
    corrupts dynamic-actor pose interpolation, which is done in time.
    ``auto`` = ``per_camera`` if either source exists, else hard error (it never silently
    degrades to the LiDAR timestamp)."""

    use_valid_masks: bool = False
    """If True, attach ``masks/<cam>/valid_mask.png`` (the ftheta->pinhole valid region) as the
    per-image mask so the black rectification corners are excluded from the RGB loss and from
    the model's own eval metrics. Off by default to stay faithful to the published runs."""


@dataclass
class NvidiaAvFog(ADDataParser):
    """NVIDIA_AV_Fog DataParser."""

    config: NvidiaAvFogDataParserConfig

    # Populated in _generate_dataparser_outputs() before the parent runs:
    clip_root: Path = Path()  # processed_pinhole/ for the chosen sequence
    K_per_cam: Dict[str, Dict] = field(default_factory=dict)
    T_camera_rig: Dict[str, np.ndarray] = field(default_factory=dict)   # 4x4, p_cam = T @ p_ego
    T_lidar_rig: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4, p_lidar = T @ p_ego
    T_rig_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))  # [T,4,4] p_world = T @ p_ego
    timestamps_us: np.ndarray = field(default_factory=lambda: np.zeros((0,)))     # [T] microseconds (LiDAR spin centre)
    camera_timestamps_us: Dict[str, np.ndarray] = field(default_factory=dict)    # cam -> [T] microseconds
    lidar_frame: str = "sensor"  # resolved from provenance.lidar_frame; see _resolve_lidar_frame()
    num_timesteps: int = 0
    # Per-camera/lidar metadata for split lookup:
    _cam_timestep: List[int] = field(default_factory=list)
    _lidar_timestep: List[int] = field(default_factory=list)

    # ----- World-frame conventions ------------------------------------------------

    @property
    def _world_rotation(self) -> np.ndarray:
        """4x4 transform: world_iso = R_iso_from_world @ world_ncore (identity if disabled)."""
        if not self.config.apply_iso_8855_rotation:
            return np.eye(4)
        T = np.eye(4)
        T[:3, :3] = R_ISO_FROM_NCORE
        return T

    # ----- Dataset entrypoint ----------------------------------------------------

    def _generate_dataparser_outputs(self, split="train"):
        # Resolve clip directory by prefix match. On ns-eval the reloaded config
        # can come back with data=None (nerfstudio resets the top-level Config.data
        # to None and does not repopulate the dataparser field), so fall back to the
        # config field's default dataset root.
        data_root = self.config.data
        if data_root is None:
            data_root = type(self.config).__dataclass_fields__["data"].default
        self.clip_root = _resolve_clip_root(data_root, self.config.sequence)
        # Load all calibration once.
        self.K_per_cam = json.loads((self.clip_root / "calib" / "K_rectified.json").read_text())
        T_cam_raw = json.loads((self.clip_root / "calib" / "T_camera_rig.json").read_text())
        self.T_camera_rig = {k: _list_to_4x4(v) for k, v in T_cam_raw.items()}
        T_lid_raw = json.loads((self.clip_root / "calib" / "T_lidar_rig.json").read_text())
        self.T_lidar_rig = _list_to_4x4(T_lid_raw[NVIDIA_AV_FOG_LIDAR])
        # Load per-timestep ego pose.
        pose_df = pd.read_parquet(self.clip_root / "calib" / "T_rig_world.parquet")
        pose_df = pose_df.sort_values("timestep_idx").reset_index(drop=True)
        T = len(pose_df)
        Trw = np.tile(np.eye(4)[None, :, :], (T, 1, 1)).astype(np.float64)
        for r in range(3):
            for c in range(4):
                Trw[:, r, c] = pose_df[f"T_rig_world_{r}{c}"].to_numpy()
        self.T_rig_world = Trw
        self.timestamps_us = pose_df["timestamp_us"].to_numpy().astype(np.int64)
        self.num_timesteps = T
        self.camera_timestamps_us = self._load_camera_timestamps()
        self._apply_rolling_shutter_config()
        self.lidar_frame = self._resolve_lidar_frame()
        # If a fixed split file was requested, load timestep indices for it.
        self._train_timesteps, self._eval_timesteps = self._load_split_indices()
        outputs = super()._generate_dataparser_outputs(split)
        if self.config.use_valid_masks:
            # image path is <clip_root>/images/<cam>/frame_XXXX.jpg -> mask is per-camera, time-invariant
            outputs.mask_filenames = [
                self.clip_root / "masks" / Path(f).parent.name / "valid_mask.png" for f in outputs.image_filenames
            ]
            missing = [m for m in set(outputs.mask_filenames) if not m.exists()]
            if missing:
                raise FileNotFoundError(f"use_valid_masks=True but masks are missing: {missing}")
        return outputs

    # ----- Timing -----------------------------------------------------------------

    def _spin_centers_us(self) -> np.ndarray:
        """LiDAR spin-centre timestamp per timestep [T] (microseconds)."""
        lt = self.clip_root / "calib" / "lidar_timestamps.parquet"
        if lt.exists():
            df = pd.read_parquet(lt)
            if "spin_center_us" in df.columns:
                sort_col = "sweep_idx" if "sweep_idx" in df.columns else df.columns[0]
                df = df.sort_values(sort_col)
                return df["spin_center_us"].to_numpy().astype(np.int64)
        # T_rig_world.timestamp_us is the same quantity (the converter writes spin_center there).
        return self.timestamps_us.astype(np.int64)

    def _load_camera_timestamps(self) -> Dict[str, np.ndarray]:
        """Per-camera capture timestamps, [T] microseconds, one entry per camera.

        F1. The original implementation gave every camera the LiDAR spin-centre
        timestamp. The cameras are in fact triggered in two groups (see
        ``NVIDIA_AV_FOG_CAMERA_TRIGGER_GROUPS``) with a systematic offset from the spin
        centre. Measured medians (group A / group B):

            NFF-01  +2.74 / -14.17 ms      NCF-02  -4.71 / -15.75 ms
            NFF-07  +5.59 / -11.31 ms      NCF-08  -0.28 /  -2.31 ms

        Worst inter-group gap 16.91 ms = 15.5 cm of ego motion at 32.9 km/h (~23 cm at
        49 km/h), and it propagates into dynamic-actor pose interpolation, which is
        keyed on time.
        """
        cams = tuple(NVIDIA_AV_FOG_CAMERAS)
        src = self.config.camera_timestamp_source
        spin = self._spin_centers_us()

        if src == "lidar":
            CONSOLE.print(
                "[bold red]NVIDIA_AV_FOG: camera_timestamp_source='lidar' -- every camera is being given the "
                "LiDAR spin-centre timestamp. This is the PRE-FIX (defective) behaviour and is only here to "
                "re-create superseded runs. Up to 16.9 ms / 15.5 cm of camera timing error.[/bold red]"
            )
            return {c: spin.copy() for c in cams}

        ts: Dict[str, np.ndarray] = {}
        origin = "unknown"

        cam_ts_path = self.clip_root / "calib" / "camera_timestamps.parquet"
        if cam_ts_path.exists():
            df = pd.read_parquet(cam_ts_path)
            if "timestep_idx" in df.columns:
                df = df.sort_values("timestep_idx").reset_index(drop=True)
            missing = [c for c in cams if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{cam_ts_path} exists but is missing camera column(s) {missing}. "
                    f"Columns present: {list(df.columns)}"
                )
            if len(df) != self.num_timesteps:
                raise ValueError(
                    f"{cam_ts_path} has {len(df)} rows but the clip has {self.num_timesteps} timesteps."
                )
            ts = {c: df[c].to_numpy().astype(np.int64) for c in cams}
            origin = "calib/camera_timestamps.parquet"
        else:
            video_dir = self.clip_root / "videos"
            per_cam_files = {c: video_dir / f"{c}.timestamps.parquet" for c in cams}
            if not all(f.exists() for f in per_cam_files.values()):
                raise FileNotFoundError(
                    "F1: real per-camera timestamps are required but were not found.\n"
                    f"  looked for {cam_ts_path}\n"
                    f"  and for    {video_dir}/<cam>.timestamps.parquet\n"
                    "Re-run the dataset converter so one of them exists, or pass "
                    "`--camera-timestamp-source lidar` to deliberately reproduce the superseded, "
                    "defective behaviour (up to 16.9 ms of camera timing error)."
                )
            for c, f in per_cam_files.items():
                col = pd.read_parquet(f)["timestamp"].to_numpy().astype(np.int64)
                # Same frame-selection rule the converter used to pick the rectified frame:
                # argmin |camera timestamp - lidar spin_center_us|.
                idx = np.abs(col[None, :] - spin[:, None]).argmin(axis=1)
                ts[c] = col[idx]
            origin = "videos/<cam>.timestamps.parquet (nearest to spin centre)"

        self._log_camera_timing(ts, spin, origin)
        return ts

    def _log_camera_timing(self, ts: Dict[str, np.ndarray], spin: np.ndarray, origin: str) -> None:
        """Sanity-check block: per-camera offset vs the LiDAR spin centre, and the gap
        between the two trigger groups. Also hard-fails on implausible offsets."""
        frame_period_us = 33_331.0  # 30 Hz cameras; measured median inter-frame delta
        CONSOLE.log(f"NVIDIA_AV_FOG camera timestamps: source = {origin}")
        medians: Dict[str, float] = {}
        for c in NVIDIA_AV_FOG_CAMERAS:
            d = (ts[c].astype(np.float64) - spin.astype(np.float64)) / 1000.0  # ms
            medians[c] = float(np.median(d))
            CONSOLE.log(
                f"  {c:30s} offset vs spin centre: median {medians[c]:+8.3f} ms, "
                f"p05 {np.percentile(d, 5):+8.3f}, p95 {np.percentile(d, 95):+8.3f} ms"
            )
            if np.abs(d).max() > frame_period_us / 1000.0:
                raise ValueError(
                    f"Camera {c} timestamp is more than one frame period from the LiDAR spin centre "
                    f"(max |offset| = {np.abs(d).max():.1f} ms). The timestamp source is probably on a "
                    f"different clock than calib/lidar_timestamps.parquet -- refusing to guess."
                )
        ga = [medians[c] for c in NVIDIA_AV_FOG_CAMERA_TRIGGER_GROUPS[0] if c in medians]
        gb = [medians[c] for c in NVIDIA_AV_FOG_CAMERA_TRIGGER_GROUPS[1] if c in medians]
        if ga and gb:
            gap = abs(float(np.median(ga)) - float(np.median(gb)))
            CONSOLE.log(
                f"  trigger-group gap: {gap:.2f} ms "
                f"(group A median {np.median(ga):+.2f}, group B median {np.median(gb):+.2f} ms)"
            )

    def _apply_rolling_shutter_config(self) -> None:
        """F2. Take the rolling-shutter readout time from the published calibration.

        ``calib/camera_intrinsics_ftheta.json`` carries, per camera::

            "shutter_type":    "ROLLING_TOP_TO_BOTTOM"
            "shutter_delay_us": 30559

        nerfstudio applies ``rolling_shutter_time`` to every camera uniformly, so we
        require the seven cameras to agree (they do, in all 32 clips) rather than
        silently averaging them.
        """
        if self.config.rolling_shutter_source == "config":
            CONSOLE.print(
                f"[bold yellow]NVIDIA_AV_FOG: rolling_shutter_source='config' -> "
                f"rolling_shutter_time={self.config.rolling_shutter_time} s is being used verbatim and the "
                f"published 0.030559 s readout is ignored.[/bold yellow]"
            )
            return

        path = self.clip_root / "calib" / "camera_intrinsics_ftheta.json"
        if not path.exists():
            raise FileNotFoundError(
                f"rolling_shutter_source='calib' but {path} is missing. Pass "
                f"`--rolling-shutter-source config --rolling-shutter-time 0.0` to deliberately "
                f"treat the cameras as global-shutter (the superseded behaviour)."
            )
        ftheta = json.loads(path.read_text())
        delays, types = {}, {}
        for cam in NVIDIA_AV_FOG_CAMERAS:
            if cam not in ftheta:
                raise ValueError(f"{path} has no entry for {cam}")
            delays[cam] = float(ftheta[cam]["shutter_delay_us"]) * 1e-6
            types[cam] = str(ftheta[cam].get("shutter_type", "UNKNOWN"))

        bad = {c: t for c, t in types.items() if t != "ROLLING_TOP_TO_BOTTOM"}
        if bad:
            raise ValueError(
                f"Expected every camera to report shutter_type='ROLLING_TOP_TO_BOTTOM' "
                f"(nerfstudio's TOP_TO_BOTTOM ray-time branch assumes it); got {bad}."
            )
        unique = sorted(set(round(v, 9) for v in delays.values()))
        if len(unique) != 1:
            raise ValueError(
                f"Cameras disagree on shutter_delay_us ({delays}). nerfstudio applies one "
                f"rolling_shutter_time to all cameras, so this needs a per-camera implementation "
                f"rather than a silent average."
            )
        self.config.rolling_shutter_time = unique[0]
        frame_period = 1.0 / 30.0
        CONSOLE.log(
            f"NVIDIA_AV_FOG rolling shutter: TOP_TO_BOTTOM, readout {unique[0] * 1e3:.3f} ms "
            f"({100.0 * unique[0] / frame_period:.1f}% of the 33.3 ms frame period); "
            f"time_to_center_pixel = {self.config.time_to_center_pixel} s "
            f"(0 = the image timestamp is assumed to be the CENTRE of the frame; see F2 in the PR)"
        )

    # ----- LiDAR frame --------------------------------------------------------------

    def _resolve_lidar_frame(self) -> str:
        """F3. Decide the frame of ``lidar/sweep_*.npy`` from the clip, not from a default.

        ``provenance.json`` carries ``lidar_frame`` once the clip has been converted::

            {"lidar_frame": "ego",
             "lidar_frame_fix": {"ground_z_before_m": -1.96, "ground_z_after_m": 0.0, ...}}

        Clips written before the fix have no marker and are in the SENSOR frame. Loading
        one as "ego" applies inv(T_lidar_rig) a second time and drops the world-frame
        ground to -1.94 m -- the error behind the superseded published numbers.
        """
        prov_path = self.clip_root / "provenance.json"
        marker = None
        if prov_path.exists():
            try:
                marker = json.loads(prov_path.read_text()).get("lidar_frame")
            except json.JSONDecodeError as e:
                raise ValueError(f"{prov_path} is not valid JSON: {e}") from e
        if marker is not None and marker not in ("ego", "sensor"):
            raise ValueError(
                f"{prov_path}: lidar_frame={marker!r} is not one of 'ego' / 'sensor'. "
                f"Refusing to guess the LiDAR frame."
            )

        forced = self.config.lidar_points_frame
        if forced != "auto":
            if marker is not None and marker != forced:
                CONSOLE.print(
                    f"[bold yellow]NVIDIA_AV_FOG: lidar_points_frame='{forced}' OVERRIDES "
                    f"provenance.lidar_frame='{marker}'. If that is not deliberate you are introducing a "
                    f"{abs(float(self.T_lidar_rig[2, 3])):.2f} m vertical error.[/bold yellow]"
                )
            else:
                CONSOLE.log(f"NVIDIA_AV_FOG LiDAR frame: '{forced}' (forced by config)")
            frame = forced
        elif marker is not None:
            CONSOLE.log(f"NVIDIA_AV_FOG LiDAR frame: '{marker}' (from provenance.lidar_frame)")
            frame = marker
        elif self.config.allow_unmarked_lidar_frame:
            CONSOLE.print(
                "[bold red]NVIDIA_AV_FOG: this clip has no provenance.lidar_frame. "
                "allow_unmarked_lidar_frame=True -> assuming the LEGACY SENSOR frame. "
                "Verify against the ground-plane check below before trusting any metric.[/bold red]"
            )
            frame = "sensor"
        else:
            raise ValueError(
                f"{prov_path} has no 'lidar_frame' marker, so the frame of lidar/sweep_*.npy is unknown.\n"
                f"  This clip predates the frame fix and is almost certainly in the SENSOR frame.\n"
                f"  Pass `--allow-unmarked-lidar-frame True` to accept that reading, or\n"
                f"  `--lidar-points-frame {{ego,sensor}}` to state it outright.\n"
                f"  Guessing costs {abs(float(self.T_lidar_rig[2, 3])):.2f} m of vertical offset."
            )

        if self.config.check_lidar_frame_geometry:
            self._check_lidar_frame_geometry(frame)
        return frame

    def _check_lidar_frame_geometry(self, frame: str) -> None:
        """Warn if the measured ground plane disagrees with the declared frame.

        Ground z is estimated as the 10th percentile of z over points 6-20 m out, which is
        robust to the ego vehicle and to nearby traffic but NOT to a strong road gradient --
        hence a warning, never an error.
        """
        sweep_path = self.clip_root / "lidar" / "sweep_0000.npy"
        if not sweep_path.exists():
            return
        sweep = np.load(sweep_path)
        radius = np.hypot(sweep[:, 0], sweep[:, 1])
        sel = sweep[(radius > 6.0) & (radius < 20.0), 2]
        if sel.size < 1000:
            return
        ground_z = float(np.percentile(sel, 10))
        mount_z = float(self.T_lidar_rig[2, 3])
        err_declared = abs(ground_z - (0.0 if frame == "ego" else -mount_z))
        err_other = abs(ground_z - (-mount_z if frame == "ego" else 0.0))
        CONSOLE.log(
            f"NVIDIA_AV_FOG LiDAR frame check: ground z (p10, 6-20 m) = {ground_z:+.2f} m, "
            f"lidar mount z = {mount_z:+.2f} m; expected ~"
            f"{0.0 if frame == 'ego' else -mount_z:+.2f} m for frame='{frame}'"
        )
        if err_other + 1.0 < err_declared:
            CONSOLE.print(
                f"[bold red]NVIDIA_AV_FOG: the ground plane fits frame="
                f"'{'sensor' if frame == 'ego' else 'ego'}' {err_declared - err_other:.2f} m better than the "
                f"declared '{frame}'. Either provenance.lidar_frame is wrong or this clip is on a steep "
                f"gradient. Check before trusting LiDAR metrics.[/bold red]"
            )

    # ----- ADDataParser overrides -------------------------------------------------

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        cams = tuple(NVIDIA_AV_FOG_CAMERAS) if "all" in self.config.cameras else tuple(self.config.cameras)
        Rwrot = self._world_rotation  # 4x4

        intrinsics: List[np.ndarray] = []
        c2ws: List[np.ndarray] = []
        heights: List[int] = []
        widths: List[int] = []
        times: List[float] = []
        sensor_idxs: List[int] = []
        image_filenames: List[Path] = []
        timestep_record: List[int] = []

        for t in range(self.num_timesteps):
            T_rig_world_t = Rwrot @ self.T_rig_world[t]  # 4x4
            for slot, cam_name in enumerate(cams):
                # F1: the capture time is the CAMERA's own timestamp, not the LiDAR
                # spin centre. See _load_camera_timestamps().
                ts_seconds = float(self.camera_timestamps_us[cam_name][t]) * 1e-6
                K_info = self.K_per_cam[cam_name]
                K = np.array([
                    [K_info["fx"], 0.0,           K_info["cx"]],
                    [0.0,          K_info["fy"],  K_info["cy"]],
                    [0.0,          0.0,           1.0],
                ], dtype=np.float64)
                # T_camera_rig.json stores T_rig_from_camera (i.e. cam->rig) despite the
                # "T_sensor_from_ego" convention string in the sibling lidar file. Verified by
                # physical sanity: file translation column = camera position in rig (+x fwd, +z up)
                # places front cams at (1.95, ~0, 1.32) m (windshield) and rear cams behind axles;
                # inverting puts cameras 1.9 m underground. So use the matrix directly.
                cam_to_rig = self.T_camera_rig[cam_name]
                c2w = T_rig_world_t @ cam_to_rig
                # OpenCV (x-right, y-down, z-fwd) -> NeRFStudio (x-right, y-up, z-back).
                c2w[:3, :3] = c2w[:3, :3] @ OPENCV_TO_NERFSTUDIO

                intrinsics.append(K)
                c2ws.append(c2w)
                widths.append(int(K_info["w"]))
                heights.append(int(K_info["h"]))
                times.append(ts_seconds)
                sensor_idxs.append(slot)
                timestep_record.append(t)
                image_filenames.append(self.clip_root / "images" / cam_name / f"frame_{t:04d}.jpg")

        intrinsics_t = torch.tensor(np.stack(intrinsics), dtype=torch.float32)
        c2ws_t = torch.tensor(np.stack(c2ws), dtype=torch.float32)
        times_t = torch.tensor(times, dtype=torch.float64)
        idxs_t = torch.tensor(sensor_idxs, dtype=torch.int32).unsqueeze(-1)
        heights_t = torch.tensor(heights, dtype=torch.int32).unsqueeze(-1)
        widths_t = torch.tensor(widths, dtype=torch.int32).unsqueeze(-1)

        self._cam_timestep = timestep_record
        # NOTE: do not pass distortion_params here. Images are pinhole-rectified
        # already (K_rectified.json: distortion=none). Passing a zeros tensor
        # actually TRIGGERS the cv2.undistort path in full_images_datamanager —
        # leaving it unset (None) is what causes the path to be skipped.
        cameras = Cameras(
            fx=intrinsics_t[:, 0, 0],
            fy=intrinsics_t[:, 1, 1],
            cx=intrinsics_t[:, 0, 2],
            cy=intrinsics_t[:, 1, 2],
            height=heights_t,
            width=widths_t,
            camera_to_worlds=c2ws_t[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            times=times_t,
            metadata={
                "sensor_idxs": idxs_t,
                "timesteps": torch.tensor(timestep_record, dtype=torch.int32).unsqueeze(-1),
            },
        )
        return cameras, image_filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        Rwrot = self._world_rotation
        # T_lidar_rig.json also stores T_rig_from_lidar (= lidar->rig); see camera comment above.
        # Translation (1.14, 0, 1.94) m = lidar position on rig (roof-mounted, forward of center).
        lidar_to_rig = self.T_lidar_rig
        poses, times, idxs, filenames, timestep_record = [], [], [], [], []
        for t in range(self.num_timesteps):
            T_rig_world_t = Rwrot @ self.T_rig_world[t]
            l2w = T_rig_world_t @ lidar_to_rig
            ts_seconds = float(self.timestamps_us[t]) * 1e-6
            poses.append(l2w)
            times.append(ts_seconds)
            idxs.append(0)  # only one lidar
            timestep_record.append(t)
            filenames.append(self.clip_root / "lidar" / f"sweep_{t:04d}.npy")
        poses_t = torch.tensor(np.stack(poses), dtype=torch.float32)
        times_t = torch.tensor(times, dtype=torch.float64)
        idxs_t = torch.tensor(idxs, dtype=torch.int32).unsqueeze(-1)
        self._lidar_timestep = timestep_record
        lidars = Lidars(
            lidar_to_worlds=poses_t[:, :3, :4],
            lidar_type=LidarType.PANDAR64,  # closest enum; affects ray-drop only
            times=times_t,
            metadata={
                "sensor_idxs": idxs_t,
                "timesteps": torch.tensor(timestep_record, dtype=torch.int32).unsqueeze(-1),
            },
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, filenames

    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[torch.Tensor]:
        """Returns one [N, 5] tensor per lidar frame: x, y, z, intensity, t_offset (sec).

        ADDataParser wants the points in the LIDAR SENSOR frame (``lidar_to_worlds``
        already contains ``T_lidar_rig``). So:
          * ``self.lidar_frame == "sensor"`` -> the points are already there, use as-is;
          * ``self.lidar_frame == "ego"``    -> map ego -> sensor with inv(T_lidar_rig).
        The frame comes from ``provenance.lidar_frame`` (F3), not from a config default.
        """
        # self.T_lidar_rig is T_rig_from_lidar (lidar->rig); invert to get rig->lidar so we can
        # transform sweep points from ego(=rig) frame into actual lidar-sensor frame.
        if self.lidar_frame == "sensor":
            Rrig = torch.eye(3); trig = torch.zeros(3)  # points already in the lidar-sensor frame
        else:
            T_lid_from_rig = torch.tensor(np.linalg.inv(self.T_lidar_rig), dtype=torch.float64)
            Rrig = T_lid_from_rig[:3, :3].float()
            trig = T_lid_from_rig[:3, 3].float()
        point_clouds = []
        for f in filenames:
            sweep = np.load(f)  # [N, 6] float32
            xyz_ego = torch.from_numpy(sweep[:, 0:3]).float()
            # Ego -> lidar-sensor frame (lidar_to_rig = identity-ish but be safe).
            xyz_sensor = (Rrig @ xyz_ego.T).T + trig
            intensity = torch.from_numpy(sweep[:, LIDAR_SWEEP_COL_INTENSITY:LIDAR_SWEEP_COL_INTENSITY + 1]).float()
            t_offset_s = torch.from_numpy(
                sweep[:, LIDAR_SWEEP_COL_T_OFFSET_US:LIDAR_SWEEP_COL_T_OFFSET_US + 1].astype(np.float32) * 1e-6
            )
            pc = torch.cat([xyz_sensor, intensity, t_offset_s], dim=-1)  # [N, 5]
            point_clouds.append(pc)
        # ADDataParser._adjust_times will rebase pc[:, 4] -= 0 if the column is already
        # an offset (which it is — relative to sweep center). Nothing else to do.
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds

    def _get_actor_trajectories(self) -> List[Dict]:
        """Read obstacles_3d_world.parquet and emit per-track trajectories.

        Per-track dict: poses [T,4,4] (world), timestamps [T] (seconds), dims [3] (wlh),
        label (str), stationary (bool), symmetric (bool), deformable (bool).
        """
        df_path = self.clip_root / "labels" / "obstacles_3d_world.parquet"
        if not df_path.exists():
            return []
        df = pd.read_parquet(df_path)
        if not len(df):
            return []
        Rwrot = self._world_rotation[:3, :3]
        twrot = self._world_rotation[:3, 3]

        allowed = set(NVIDIA_AV_FOG_ALLOWED_RIGID_CLASSES)
        if self.config.include_deformable_actors:
            allowed |= set(NVIDIA_AV_FOG_ALLOWED_DEFORMABLE_CLASSES)
        df = df[df["label_class"].str.lower().isin(allowed)]
        if not len(df):
            return []

        ts_seconds = self.timestamps_us.astype(np.float64) * 1e-6
        trajectories: List[Dict] = []
        for track_id, g in df.groupby("track_id"):
            g = g.sort_values("timestep_idx")
            ti = g["timestep_idx"].to_numpy().astype(np.int64)
            cx = g["center_x_w"].to_numpy().astype(np.float64)
            cy = g["center_y_w"].to_numpy().astype(np.float64)
            cz = g["center_z_w"].to_numpy().astype(np.float64)
            qx = g["q_x_w"].to_numpy().astype(np.float64)
            qy = g["q_y_w"].to_numpy().astype(np.float64)
            qz = g["q_z_w"].to_numpy().astype(np.float64)
            qw = g["q_w_w"].to_numpy().astype(np.float64)
            sx = g["size_x"].to_numpy().astype(np.float32).max()  # length (rigid track)
            sy = g["size_y"].to_numpy().astype(np.float32).max()  # width
            sz = g["size_z"].to_numpy().astype(np.float32).max()  # height
            n = len(ti)

            poses = np.tile(np.eye(4)[None, :, :], (n, 1, 1)).astype(np.float64)
            for i in range(n):
                R = _quat_to_R(qx[i], qy[i], qz[i], qw[i])
                # Apply world rotation: p_iso = Rwrot @ p_ncore
                R_world = Rwrot @ R
                t_world = Rwrot @ np.array([cx[i], cy[i], cz[i]]) + twrot
                poses[i, :3, :3] = R_world
                poses[i, :3, 3] = t_world

            timestamps_s = ts_seconds[ti]
            # dims: PandaSet convention is (width, length, height). NCore stores (size_x=length,
            # size_y=width, size_z=height). Reorder to (width, length, height).
            dims = np.array([sy, sx, sz], dtype=np.float32)

            label = str(g["label_class"].iloc[0])
            is_pedestrian = label in NVIDIA_AV_FOG_ALLOWED_DEFORMABLE_CLASSES

            trajectories.append({
                "poses": torch.from_numpy(poses).float(),
                "timestamps": torch.from_numpy(timestamps_s),
                "dims": torch.from_numpy(dims),
                "label": label,
                "stationary": False,
                "symmetric": not is_pedestrian,
                "deformable": is_pedestrian,
            })
        return trajectories

    # ----- Split override --------------------------------------------------------

    def _get_train_eval_indices(self, sensors) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._train_timesteps is None or self.config.splits_json == "auto":
            return super()._get_train_eval_indices(sensors)
        # Map sensor-array indices to their timestep via metadata.
        timesteps = sensors.metadata["timesteps"].squeeze(-1).cpu().numpy()
        train_mask = np.isin(timesteps, self._train_timesteps)
        eval_mask = np.isin(timesteps, self._eval_timesteps)
        train_idx = torch.from_numpy(np.where(train_mask)[0]).long()
        eval_idx = torch.from_numpy(np.where(eval_mask)[0]).long()
        if self.config.max_eval_frames is not None:
            torch.manual_seed(123)
            eval_idx = eval_idx[torch.randperm(len(eval_idx))[: self.config.max_eval_frames]]
        return train_idx, eval_idx

    def _load_split_indices(self):
        if self.config.splits_json == "auto":
            return None, None
        path = self.clip_root / "splits" / f"{self.config.splits_json}.json"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        data = json.loads(path.read_text())
        return np.array(data["train_indices"], dtype=np.int64), np.array(data["eval_indices"], dtype=np.int64)


# ---- Helpers --------------------------------------------------------------------


def _list_to_4x4(rows: List[List[float]]) -> np.ndarray:
    arr = np.array(rows, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4)
        out[:3, :4] = arr
        return out
    raise ValueError(f"Expected 3x4 or 4x4 matrix, got {arr.shape}")


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    R = np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw),     s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw),     1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw),     s * (qy * qz + qx * qw),     1 - s * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


def _resolve_clip_root(data_root: Path, sequence: str) -> Path:
    """Locate a clip directory under ``data_root``, supporting both on-disk layouts.

    Live layout (what the dataset actually ships today)::

        <data_root>/clips/NFF-01__b9ea7a9d/{images,lidar,labels,calib,videos,splits,...}

    ``data_root`` may also be the parent that holds several subsets, in which case both
    ``NVIDIA_FOGGY_Field/clips`` and ``NVIDIA_CLEAR_Field/clips`` are searched::

        <data_root>/NVIDIA_FOGGY_Field/clips/NFF-01__b9ea7a9d/

    Legacy layout (kept working; it is what the published runs used)::

        <data_root>/processed/002_<uuid>_fog_day_rural/processed_pinhole/

    ``sequence`` matches, in order of preference: the exact directory name, the clip id
    before ``__`` (``NFF-01``), the hash after ``__`` (``b9ea7a9d``), or -- legacy only --
    a leading 3-digit prefix (``002``). An ambiguous match is an error, never a guess.
    """
    roots: List[Path] = []
    if (data_root / "clips").is_dir():
        roots.append(data_root / "clips")
    for subset in ("NVIDIA_FOGGY_Field", "NVIDIA_CLEAR_Field"):
        if (data_root / subset / "clips").is_dir():
            roots.append(data_root / subset / "clips")

    candidates: List[Path] = []
    for root in roots:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            clip_id, _, clip_hash = name.partition("__")
            if sequence in (name, clip_id, clip_hash) or clip_id.lower() == sequence.lower():
                candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Sequence '{sequence}' matches multiple clips: {[str(c) for c in candidates]}. "
            "Use the full directory name."
        )
    if roots:
        available = sorted(c.name for root in roots for c in root.iterdir() if c.is_dir())
        if not (data_root / "processed").exists():
            raise ValueError(f"Sequence '{sequence}' not found under {[str(r) for r in roots]}. Available: {available}")

    # ---- legacy processed/<clip>/processed_pinhole/ layout ----
    processed = data_root / "processed"
    if not processed.exists():
        raise FileNotFoundError(
            f"{data_root} has neither a 'clips/' directory (live layout) nor a 'processed/' "
            f"directory (legacy layout)."
        )
    legacy = []
    for child in sorted(processed.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(f"{sequence}_") or sequence in child.name:
            legacy.append(child)
    if not legacy:
        raise ValueError(
            f"Sequence '{sequence}' not found under {processed}. "
            f"Available: {[c.name for c in sorted(processed.iterdir()) if c.is_dir()]}"
        )
    if len(legacy) > 1:
        raise ValueError(
            f"Sequence '{sequence}' matches multiple clips: {[c.name for c in legacy]}. "
            "Use a longer/more specific prefix."
        )
    clip = legacy[0] / "processed_pinhole"
    if not clip.exists():
        raise FileNotFoundError(f"No processed_pinhole/ under {legacy[0]}")
    return clip
