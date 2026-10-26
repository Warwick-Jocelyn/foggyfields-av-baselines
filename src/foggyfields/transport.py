"""Sensor transport through the medium (plan v2 §5.3/§5.4, the B1-lite subset).

- expected_depth_overlay: the P1 prototype camera formation I = T(D)*C + (1 - T(D))*A.
  An APPROXIMATION of the ordered compositing (§5.3): correct for a single opaque surface,
  biased where a pixel mixes depths. Always label results produced with it.
- ordered_airlight: the airlight term of §5.3 on the medium grid, with the scene
  transmittance W_scene interpolated so that no airlight is added behind an opaque surface.
- attenuate_surface_echo: LiDAR B1-lite, surface echo only: E_S = g * a_S * T_L(D)^2.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .medium import MediumLUT


def expected_depth_overlay(rgb: Tensor, depth: Tensor, lut: MediumLUT, airlight: Tensor,
                           accumulation: Tensor | None = None, sky_tau: Tensor | None = None) -> Tensor:
    """rgb [R, 3] (clear surface colour), depth [R] (expected ray distance, metres),
    airlight [3] or [R, 3]. If accumulation (alpha in [0, 1]) is given, the (1 - alpha)
    sky remainder gets airlight at sky_tau (default: the LUT's max-range tau) instead of
    the surface depth — NeuRAD's expected depth is biased short exactly there.
    Zero-fog reduction: tau == 0 -> returns rgb exactly (T == 1, 1 - T == 0)."""
    T = lut.transmittance(depth)[:, None]
    A = airlight if airlight.dim() == 2 else airlight[None, :]
    out = T * rgb + (1.0 - T) * A
    if accumulation is not None:
        st = lut.tau[:, -1] if sky_tau is None else sky_tau
        T_sky = torch.exp(-st)[:, None]
        sky = T_sky * rgb + (1.0 - T_sky) * A
        a = accumulation[:, None]
        out = a * out + (1.0 - a) * sky
    return out


def ordered_airlight(lut: MediumLUT, airlight: Tensor,
                     scene_w: Tensor | None = None, scene_s: Tensor | None = None,
                     s_far: float | None = None) -> Tensor:
    """Airlight term of §5.3, integrated EXACTLY against the LUT: the in-scattered airlight
    is integral W_scene(s) * (-dT/ds) * A ds, and W_scene (scene transmittance remaining) is
    piecewise constant between scene samples, so the integral is a finite sum over the
    intervals between consecutive scene-sample depths:

        air = sum_j W_j * [T(b_j) - T(b_{j+1})] * A,   b = [0, s_1, ..., s_S, s_far]

    with W = [1, 1-cumsum(w)_1, ..., 1-cumsum(w)_S]. Behind an opaque surface W == 0, so the
    medium contributes nothing there (wall occlusion is exact, not a quadrature property).
    scene_w [R, S]: compositing weights at ascending distances scene_s [R, S] (any backbone).
    Without scene samples the ray is free space (W == 1) and integrates to s_far
    (default: the LUT's max range). Returns [R, 3]."""
    R = lut.tau.shape[0]
    end = lut.edges[-1].item() if s_far is None else s_far
    if scene_w is None:
        T_end = lut.transmittance(torch.full((R,), end, device=lut.tau.device, dtype=lut.tau.dtype))
        absorbed = (1.0 - T_end)[:, None]
    else:
        far = torch.full_like(scene_s[:, :1], end).maximum(scene_s[:, -1:])
        bounds = torch.cat([torch.zeros_like(scene_s[:, :1]), scene_s, far], dim=1)  # [R, S+2]
        T = lut.transmittance(bounds)                                               # [R, S+2]
        dT = T[:, :-1] - T[:, 1:]                                                   # [R, S+1]
        W = torch.cat([torch.ones_like(scene_w[:, :1]),
                       (1.0 - torch.cumsum(scene_w, dim=1)).clamp(min=0.0)], dim=1)  # [R, S+1]
        absorbed = (W * dT).sum(dim=1, keepdim=True)
    A = airlight if airlight.dim() == 2 else airlight[None, :]
    return absorbed * A


def attenuate_surface_echo(a_s: Tensor, depth: Tensor, lut: MediumLUT, gain: Tensor | float = 1.0) -> Tensor:
    """LiDAR surface echo under round-trip attenuation: E_S = gain * a_S * T_L(D)^2.
    a_s [R] or [R, 1]: the backbone's clear surface amplitude (its rendered intensity).
    Zero-fog reduction: tau == 0, gain == 1 -> returns a_s exactly."""
    T2 = lut.transmittance(depth, two_way=True)
    return gain * a_s * (T2 if a_s.dim() == 1 else T2[:, None])
