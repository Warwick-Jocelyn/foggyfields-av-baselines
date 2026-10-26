"""T5 unit tests: FogReturnMixture on an analytic plane scene (homogeneous beta, wall at D)."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medium"))
from foggyfields.medium import HomogeneousFog, MediumIntegrator  # noqa: E402
from foggyfields.mixture import FogReturnMixture, MixtureConfig  # noqa: E402

torch.manual_seed(0)
S0 = 2.0


def plane_scene(beta, D_vals, n_near=64, n_far=64, max_range=120.0):
    n = len(D_vals)
    o = torch.zeros(n, 3, dtype=torch.float64)
    d = torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(n, 3)
    fog = HomogeneousFog(beta, learnable=beta > 0)  # beta=0 -> frozen_zero: EXACT zero
    lut = MediumIntegrator(fog, max_range=max_range,
                           n_near=n_near, n_far=n_far).double()(o, d)
    return lut, torch.as_tensor(D_vals, dtype=torch.float64)


def test_echo_strengths_closed_form():
    beta_cfg = HomogeneousFog(0.03).beta0.item()  # softplus-mapped actual value
    D = [10.0, 30.0, 60.0]
    lut, Dv = plane_scene(0.03, D)
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    a_s = torch.ones(3, dtype=torch.float64)
    E_S, E_F, s_mid = mix.echo_strengths(a_s, Dv, lut)
    for i, d in enumerate(D):
        want = math.exp(-2 * beta_cfg * d) / (d * d + S0 * S0)
        assert abs(E_S[i].item() - want) < 1e-6 * want, (i, E_S[i].item(), want)
    # one fog bin, hand-computed
    k = 5
    ds = (lut.edges[k + 1] - lut.edges[k]).item(); sm = s_mid[k].item()
    tau_mid = 0.5 * (lut.tau[0, k] + lut.tau[0, k + 1]).item()
    want_f = beta_cfg * math.exp(-2 * tau_mid) / (sm * sm + S0 * S0) * ds
    assert abs(E_F[2, k].item() - want_f) < 1e-9, (E_F[2, k].item(), want_f)


def test_dominance_swap_at_predicted_range():
    """Sweep the wall out: beyond an analytic crossover D*, the strongest echo flips
    surface -> fog. D* solves a_s e^{-2 b D}/(D^2+s0^2) = max_k E_F (fog side ~ const in D)."""
    beta = 0.03
    D_sweep = torch.linspace(5.0, 119.0, 300, dtype=torch.float64)
    lut, Dv = plane_scene(beta, D_sweep.tolist())
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    a_s = torch.ones(len(Dv), dtype=torch.float64)  # bright surface: wins while near
    E_S, E_F, s_mid = mix.echo_strengths(a_s, Dv, lut)
    rng, comp = mix.hard_select(E_S, E_F, Dv, s_mid, rule="strongest")
    is_surf = (comp == 0).double()
    # find analytic crossover: first D where max fog echo >= surface echo
    cross_idx = None
    for i in range(len(Dv)):
        if E_F[i].max() >= E_S[i]:
            cross_idx = i
            break
    assert cross_idx is not None and 0 < cross_idx < len(Dv) - 1, "crossover must be interior"
    # selection must be surface strictly before, fog at/after (monotone flip at the crossover)
    assert is_surf[: cross_idx].all(), "surface must win below the crossover"
    assert (comp[cross_idx:] > 0).all(), "fog must win at/after the crossover"
    # fog selection returns a near-range halo point, far from the wall
    assert (rng[cross_idx:] < 15.0).all()


def test_finite_near_range():
    lut, Dv = plane_scene(0.05, [0.1, 0.5, 1.0])
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    a_s = torch.ones(3, dtype=torch.float64)
    E_S, E_F, s_mid = mix.echo_strengths(a_s, Dv, lut)
    pi = mix.selection_probs(E_S, E_F)
    z = Dv.clone()
    nll = mix.nll(z, Dv, pi, s_mid)
    for t in (E_S, E_F, pi, nll):
        assert torch.isfinite(t).all()


def test_occlusion():
    lut, Dv = plane_scene(0.05, [30.0])
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    E_S, E_F, s_mid = mix.echo_strengths(torch.ones(1, dtype=torch.float64), Dv, lut)
    assert (E_F[0, s_mid >= 30.0] == 0).all(), "no fog echo behind the wall"
    # wall at ~0: no fog mass at all
    lut0, Dv0 = plane_scene(0.05, [1e-3])
    E_S0, E_F0, _ = mix.echo_strengths(torch.ones(1, dtype=torch.float64), Dv0, lut0)
    assert E_F0.abs().sum() == 0


def test_zero_fog_reduction():
    lut, Dv = plane_scene(0.0, [20.0, 50.0])
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    a_s = torch.ones(2, dtype=torch.float64)
    E_S, E_F, s_mid = mix.echo_strengths(a_s, Dv, lut)
    assert E_F.abs().sum() == 0, "beta=0 -> no fog echoes"
    pi = mix.selection_probs(E_S, E_F)
    z = Dv + 0.05
    nll = mix.nll(z, Dv, pi, s_mid)
    # with E_F = 0, the return mixture renormalises to the pure surface Gaussian
    sig = mix.cfg.sigma_surface
    pure = 0.5 * ((z - Dv) / sig) ** 2 + math.log(sig) + 0.5 * math.log(2 * math.pi)
    # fog components have pi ~ _EPS -> contribute ~0; allow tiny slack
    assert torch.allclose(nll, pure, atol=1e-6), (nll, pure)


def test_nll_recovers_beta():
    """Sample z from the generative model at known beta; Adam on the nll + drop_nll recovers
    beta on the analytic scene (surface + halo regimes mixed)."""
    torch.manual_seed(1)
    beta_true = 0.04
    n = 4000
    D = (torch.rand(n, dtype=torch.float64) * 80 + 10)
    lut, Dv = plane_scene(beta_true, D.tolist())
    gen = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    a_s = 0.05 + 0.3 * torch.rand(n, dtype=torch.float64)
    E_S, E_F, s_mid = gen.echo_strengths(a_s, Dv, lut)
    pi = gen.selection_probs(E_S, E_F)
    comp = torch.multinomial(pi, 1)[:, 0]
    centers = torch.cat([Dv[:, None], s_mid[None, :].expand(n, -1),
                         torch.zeros(n, 1, dtype=torch.float64)], dim=1)
    z = centers.gather(1, comp[:, None])[:, 0] + 0.1 * torch.randn(n, dtype=torch.float64)
    returned = comp < pi.shape[1] - 1
    # fit
    fog = HomogeneousFog(0.01).double()
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    o = torch.zeros(n, 3, dtype=torch.float64)
    dvec = torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(n, 3)
    integ = MediumIntegrator(fog, max_range=120.0, n_near=64, n_far=64).double()
    opt = torch.optim.Adam(fog.parameters(), lr=0.05)
    for it in range(300):
        lutf = integ(o, dvec)
        E_Sf, E_Ff, smf = mix.echo_strengths(a_s, Dv, lutf)
        pif = mix.selection_probs(E_Sf, E_Ff)
        loss = mix.nll(z[returned], Dv[returned], pif[returned], smf).mean() \
             + mix.drop_nll(pif[~returned]).mean() * (~returned).double().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    beta_fit = fog.beta0.item()
    beta_gen = HomogeneousFog(beta_true).beta0.item()
    assert abs(beta_fit - beta_gen) / beta_gen < 0.05, (beta_fit, beta_gen)


def test_gradient_sign_at_zero():
    """At beta=0+ with halo-like observations, dNLL/dbeta must be negative (push beta up)."""
    n = 200
    D = torch.full((n,), 60.0, dtype=torch.float64)
    z = torch.rand(n, dtype=torch.float64) * 6 + 2   # near-range halo observations
    fog = HomogeneousFog(1e-4).double()
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    integ = MediumIntegrator(fog, max_range=120.0).double()
    o = torch.zeros(n, 3, dtype=torch.float64)
    dvec = torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(n, 3)
    lut = integ(o, dvec)
    E_S, E_F, s_mid = mix.echo_strengths(torch.ones(n, dtype=torch.float64), D, lut)
    pi = mix.selection_probs(E_S, E_F)
    mix_loss = mix.nll(z, D, pi, s_mid).mean()
    mix_loss.backward()
    assert fog.raw_beta0.grad.item() < 0


def test_quadrature_K32_vs_K512():
    for K in [(16, 16), (256, 256)]:
        pass
    lut32, Dv = plane_scene(0.05, [80.0] * 8, n_near=16, n_far=16)
    lut512, _ = plane_scene(0.05, [80.0] * 8, n_near=256, n_far=256)
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    a_s = torch.ones(8, dtype=torch.float64)
    E_S32, E_F32, _ = mix.echo_strengths(a_s, Dv, lut32)
    E_S512, E_F512, _ = mix.echo_strengths(a_s, Dv, lut512)
    assert abs(E_F32.sum() - E_F512.sum()) / E_F512.sum() < 0.01, "total fog mass"
    assert torch.allclose(E_S32, E_S512, rtol=1e-6)


def test_no_range_averaging():
    lut, Dv = plane_scene(0.08, torch.linspace(5, 110, 50).tolist())
    mix = FogReturnMixture(MixtureConfig(s0_near=S0)).double()
    E_S, E_F, s_mid = mix.echo_strengths(torch.rand(50, dtype=torch.float64), Dv, lut)
    rng, comp = mix.hard_select(E_S, E_F, Dv, s_mid, rule="strongest")
    legal = torch.cat([Dv[:, None], s_mid[None, :].expand(50, -1)], dim=1)
    ok = (rng[:, None] == legal).any(dim=1) | torch.isinf(rng)
    assert ok.all(), "hard_select range must be exactly D or a bin midpoint"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
