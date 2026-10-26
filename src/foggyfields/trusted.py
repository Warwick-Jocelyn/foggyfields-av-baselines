"""Trusted-return mask (P3, plan §5.4): which observed LiDAR returns may supervise GEOMETRY
(depth / proposal / carving / intensity losses). Halo-suspect returns are excluded from
geometry supervision but stay in the mixture NLL and ray-drop losses.

Phase-A heuristic (before the mixture is warm). A return at range z along a ray with
elevation el (radians, sensor-level approximation el = asin(dir_z_world)) is UNTRUSTED iff

    z < s_halo                                   (inside the near halo shell)
    AND NOT ground-consistent:  |z - h / sin(-el)| <= ground_tol   (for el < 0)

i.e. near returns are kept only where the ground plane explains them. Vertical structure
(cars, walls, poles) closer than s_halo is sacrificed by design during phase A; s_halo is
calibrated offline so that >= 98% of CLEAR-clip returns stay trusted (jobs/p3_mask_calib.py).
After mixture warm-up the mask switches to pi_F > 0.5 (detached), per the plan.
"""
from __future__ import annotations

import torch
from torch import Tensor

H_SENSOR = 2.16          # lidar height above ground, sensor frame (provenance ground_z)


def trusted_returns(z: Tensor, dir_world_z: Tensor, s_halo: float = 5.0,
                    ground_tol: float = 1.0, h_sensor: float = H_SENSOR) -> Tensor:
    """z: [R] observed range (m); dir_world_z: [R] world-frame unit-dir z component.
    Returns bool [R]: True = may supervise geometry."""
    el = torch.asin(dir_world_z.clamp(-1, 1))
    near = z < s_halo
    down = el < -1e-3
    r_ground = torch.where(down, h_sensor / torch.sin(-el.clamp(max=-1e-3)), torch.full_like(z, 1e6))
    ground_ok = (z - r_ground).abs() <= ground_tol
    return ~near | (down & ground_ok)
