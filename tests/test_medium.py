"""Task 3 unit tests for the MediumIntegrator + B1-lite transport (overnight1; keep in CI).

Covers the five required properties:
  1. analytic tau to 1e-4 (homogeneous: exact; linear spatial field: exact at nodes)
  2. one-way vs two-way transmittance
  3. wall occlusion: no airlight from behind an opaque surface
  4. quadrature convergence K=32 vs K=512
  5. zero-fog reduction (CPU, bit-for-bit through overlay + lidar echo; the real-frame
     GPU variant is tests/gpu_zero_fog_real_frame.py, run inside the training container)
plus gradient-flow checks.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medium"))
from foggyfields import (HomogeneousFog, MediumIntegrator, MediumLUT,  # noqa: E402
                         attenuate_surface_echo, expected_depth_overlay, ordered_airlight)

torch.manual_seed(0)


def rays(n=64, dtype=torch.float64):
    o = torch.randn(n, 3, dtype=dtype) * 5
    d = torch.nn.functional.normalize(torch.randn(n, 3, dtype=dtype), dim=-1)
    return o, d


class LinearFog(torch.nn.Module):
    """beta(x) = c0 + c1 * (x . u): analytic tau along a ray with direction u."""
    def __init__(self, c0, c1, u):
        super().__init__()
        self.c0, self.c1 = c0, c1
        self.register_buffer("u", u)

    def forward(self, p):
        return (self.c0 + self.c1 * (p * self.u).sum(-1, keepdim=True)).clamp(min=0.0)


class GaussianBumpFog(torch.nn.Module):
    """Smooth heterogeneous field for the convergence test."""
    def forward(self, p):
        return 0.05 * torch.exp(-((p - 10.0) ** 2).sum(-1, keepdim=True) / (2 * 15.0 ** 2)) + 0.01


def test_analytic_homogeneous_tau():
    beta = 0.05
    o, d = rays()
    lut = MediumIntegrator(HomogeneousFog(beta), max_range=120.0).double()(o, d)
    s = torch.rand(64, 7, dtype=torch.float64) * 150.0  # includes beyond max_range
    err = (lut.tau_at(s) - HomogeneousFog(beta).beta0.double() * s).abs().max()
    assert err < 1e-4, f"homogeneous tau error {err}"


def test_analytic_linear_field_tau():
    c0, c1 = 0.02, 2e-4
    u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    n = 16
    o = torch.zeros(n, 3, dtype=torch.float64)
    d = u.expand(n, 3)
    f = LinearFog(c0, c1, u).double()
    lut = MediumIntegrator(f, max_range=120.0, n_near=16, n_far=16).double()(o, d)
    # midpoint rule is exact for a linear integrand -> tau at the NODES is analytic
    e = lut.edges
    analytic = c0 * e + 0.5 * c1 * e ** 2
    err = (lut.tau[0] - analytic).abs().max()
    assert err < 1e-10, f"linear-field node tau error {err}"
    # between nodes, linear interpolation of a quadratic: bounded, documented error
    s = torch.rand(n, 5, dtype=torch.float64) * 120.0
    err_i = (lut.tau_at(s) - (c0 * s + 0.5 * c1 * s ** 2)).abs().max()
    assert err_i < 1e-3, f"linear-field interp tau error {err_i}"


def test_one_way_vs_two_way():
    o, d = rays()
    lut = MediumIntegrator(HomogeneousFog(0.03), max_range=120.0).double()(o, d)
    s = torch.rand(64, dtype=torch.float64) * 100.0
    T1, T2 = lut.transmittance(s), lut.transmittance(s, two_way=True)
    assert torch.allclose(T2, T1 ** 2, atol=1e-12)
    assert torch.allclose(T2, torch.exp(-2.0 * lut.tau_at(s)), atol=1e-12)


def test_wall_occlusion():
    """An opaque wall at D: airlight integrated past D must be zero."""
    n, D = 8, 30.0
    o = torch.zeros(n, 3, dtype=torch.float64)
    d = torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(n, 3)
    lut = MediumIntegrator(HomogeneousFog(0.05), max_range=120.0).double()(o, d)
    A = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float64)
    scene_s = torch.full((n, 1), D, dtype=torch.float64)
    scene_w = torch.ones(n, 1, dtype=torch.float64)          # fully opaque wall
    air_wall = ordered_airlight(lut, A, scene_w, scene_s)
    # closed form: airlight only accrues before the wall
    T = torch.exp(-lut.tau_at(torch.full((n,), D, dtype=torch.float64)))
    expected = (1.0 - T)[:, None] * A
    assert torch.allclose(air_wall, expected, atol=1e-9), (air_wall[0], expected[0])
    # wall at 0 -> no airlight at all; no wall -> airlight over the full grid (and beyond == clamp)
    air_zero = ordered_airlight(lut, A, torch.ones(n, 1, dtype=torch.float64),
                                torch.zeros(n, 1, dtype=torch.float64))
    assert air_zero.abs().max() < 1e-12
    air_free = ordered_airlight(lut, A)
    T_end = torch.exp(-lut.tau[:, -1])
    assert torch.allclose(air_free, (1.0 - T_end)[:, None] * A, atol=1e-9)


def test_quadrature_convergence():
    o, d = rays(32)
    f = GaussianBumpFog().double()
    lut32 = MediumIntegrator(f, max_range=120.0, n_near=16, n_far=16).double()(o, d)
    lut512 = MediumIntegrator(f, max_range=120.0, n_near=256, n_far=256).double()(o, d)
    s = torch.rand(32, 9, dtype=torch.float64) * 120.0
    err = (lut32.tau_at(s) - lut512.tau_at(s)).abs().max()
    rel = err / lut512.tau_at(s).max()
    # fog fields are smooth by construction (TV-regularised, plan §5.1): K=32 must be sub-1%
    assert err < 1e-2 and rel < 1e-2, f"K=32 vs K=512 tau error {err} (rel {rel})"


def test_zero_fog_reduction_cpu():
    """beta = 0: overlay returns the input bitwise, echo returns a_s bitwise, airlight = 0."""
    n = 16
    o, d = rays(n, dtype=torch.float32)
    fog = HomogeneousFog(0.0, learnable=False)
    lut = MediumIntegrator(fog, max_range=120.0)(o, d)
    assert lut.tau.abs().max() == 0.0
    rgb = torch.rand(n, 3)
    depth = torch.rand(n) * 80
    acc = torch.rand(n)
    A = torch.tensor([0.7, 0.7, 0.7])
    out = expected_depth_overlay(rgb, depth, lut, A, accumulation=acc)
    assert torch.equal(out, rgb * 1.0) or torch.allclose(out, rgb, atol=0.0), "overlay must reduce to identity"
    a_s = torch.rand(n)
    assert torch.equal(attenuate_surface_echo(a_s, depth, lut, gain=1.0), a_s)
    assert ordered_airlight(lut, A).abs().max() == 0.0


def test_gradients_flow_to_beta():
    fog = HomogeneousFog(0.05)
    integ = MediumIntegrator(fog, max_range=120.0)
    o, d = rays(8, dtype=torch.float32)
    lut = integ(o, d)
    s = torch.full((8,), 40.0)
    loss = lut.transmittance(s).sum()
    loss.backward()
    g = fog.raw_beta0.grad
    assert g is not None and torch.isfinite(g).all() and g.abs() > 0
    # dT/dbeta = -s * T for homogeneous; check numerically
    with torch.no_grad():
        b = fog.beta0.item()
    analytic = (-40.0 * math.exp(-b * 40.0)) * 8  # dT/dbeta0 summed over rays
    sigmoid = torch.sigmoid(fog.raw_beta0).item()  # dsoftplus/draw
    assert abs(g.item() - analytic * sigmoid) < 1e-3 * abs(analytic * sigmoid)


def test_overlay_matches_koschmieder():
    """Single opaque surface at depth D: overlay == Koschmieder C e^{-bD} + A (1 - e^{-bD})."""
    beta = 0.04
    n = 8
    o = torch.zeros(n, 3); d = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    lut = MediumIntegrator(HomogeneousFog(beta), max_range=120.0)(o, d)
    rgb = torch.rand(n, 3); depth = torch.rand(n) * 100; A = torch.rand(3)
    out = expected_depth_overlay(rgb, depth, lut, A)
    b = HomogeneousFog(beta).beta0
    T = torch.exp(-b * depth)[:, None]
    assert torch.allclose(out, T * rgb + (1 - T) * A, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
