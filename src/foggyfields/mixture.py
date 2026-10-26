"""Fog-echo return mixture (plan v2 §5.4, operational first-return form).

Per LiDAR ray with clear surface amplitude a_S at depth D, under a medium LUT (built on the
REFERENCE extinction beta_ref; the LiDAR sees tau_L = k_lidar * tau_ref):

    E_S      = a_S * T_L(D)^2 * K(D)                      surface echo
    E_F(s_k) = eta * k_L * beta_mid_k * T_L(s_k)^2 * K(s_k) * ds_k   fog echo per bin, s_k < D
    theta    = noise floor                                 no-return reference

with K(s) = 1 / (s^2 + s0^2) (finite at 0, no bare 1/s^2). Selection probabilities over
{surface, fog bins, no-return}; observed-range likelihood is a Gaussian mixture over the
SELECTED component positions. The only soft object is the likelihood: ranges are NEVER
soft-averaged across components (do-not list §11); inference uses a hard sensor rule.

Logged design decision: the default `prob_mode="softmax_log"` uses pi = softmax(log E / t)
(at t=1: pi proportional to echo strength, scale-invariant, so beta recovery does not depend
on an arbitrary intensity gain). The plan's literal softmax(E/t) is available as
`prob_mode="softmax_raw"` (needs t calibrated to the sensor's power scale).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor, nn

from .medium import MediumLUT

_EPS = 1e-20


@dataclass
class MixtureConfig:
    sigma_surface: float = 0.1      # m, surface echo range noise
    sigma_fog_scale: float = 0.5    # sigma_k = max(sigma_surface, scale * ds_k)
    temperature: float = 1.0
    s0_near: float = 2.0            # K(s) = 1/(s^2 + s0^2)
    k_lidar: float = 1.0            # beta_L = k_lidar * beta_ref (FIXED; A4: not calibratable)
    theta: float = 1e-4             # noise floor, units of the echo strengths
    learn_theta: bool = False
    eta: float = 1.0                # (duplicate kept for clarity) b_L = eta * beta_L
    learn_eta: bool = False         # log-space learnable backscatter-to-extinction ratio
    min_range: float = 0.0          # zero fog echoes below this range (sensor near-field censoring)
    prob_mode: Literal["softmax_log", "softmax_raw"] = "softmax_log"


class FogReturnMixture(nn.Module):
    def __init__(self, config: MixtureConfig | None = None):
        super().__init__()
        self.cfg = config or MixtureConfig()
        log_theta = torch.log(torch.tensor(float(self.cfg.theta)))
        if self.cfg.learn_theta:
            self.log_theta = nn.Parameter(log_theta)
        else:
            self.register_buffer("log_theta", log_theta)
        log_eta = torch.log(torch.tensor(float(self.cfg.eta)))
        if self.cfg.learn_eta:
            self.log_eta = nn.Parameter(log_eta)
        else:
            self.register_buffer("log_eta", log_eta)

    @property
    def theta(self) -> Tensor:
        return self.log_theta.exp()

    @property
    def eta(self) -> Tensor:
        return self.log_eta.exp()

    def _K(self, s: Tensor) -> Tensor:
        return 1.0 / (s * s + self.cfg.s0_near ** 2)

    def echo_strengths(self, a_s: Tensor, D: Tensor, lut: MediumLUT) -> Tuple[Tensor, Tensor, Tensor]:
        """a_s, D: [R]. Returns (E_S [R], E_F [R, K], s_mid [K]). E_F hard-zeroed at s_mid >= D."""
        kL = self.cfg.k_lidar
        e = lut.edges
        s_mid = 0.5 * (e[:-1] + e[1:])                      # [K]
        ds = e[1:] - e[:-1]
        T2_D = torch.exp(-2.0 * kL * lut.tau_at(D))         # [R]
        E_S = a_s * T2_D * self._K(D)
        tau_mid = 0.5 * (lut.tau[:, :-1] + lut.tau[:, 1:])  # [R, K] (linear interp at mids)
        T2_mid = torch.exp(-2.0 * kL * tau_mid)
        assert lut.beta_mid is not None, "LUT built without beta_mid"
        E_F = self.eta * kL * lut.beta_mid * T2_mid * self._K(s_mid)[None, :] * ds[None, :]
        E_F = E_F * (s_mid[None, :] < D[:, None])           # occlusion: no fog echo behind the surface
        if self.cfg.min_range > 0:
            E_F = E_F * (s_mid[None, :] >= self.cfg.min_range)  # near-field censoring
        return E_S, E_F, s_mid

    def selection_logits(self, E_S: Tensor, E_F: Tensor) -> Tensor:
        """[R, 1 + K + 1] logits for {surface, fog bins, no-return}."""
        E = torch.cat([E_S[:, None], E_F, self.theta.expand(E_S.shape[0])[:, None]], dim=1)
        if self.cfg.prob_mode == "softmax_log":
            return torch.log(E + _EPS) / self.cfg.temperature
        return E / self.cfg.temperature

    def selection_probs(self, E_S: Tensor, E_F: Tensor) -> Tensor:
        return torch.softmax(self.selection_logits(E_S, E_F), dim=1)

    def nll(self, z: Tensor, D: Tensor, pi: Tensor, s_mid: Tensor) -> Tensor:
        """-log L(z) on observed returns. z, D: [R]; pi: [R, K+2]; the no-return column is
        excluded and the return components renormalised (z was observed)."""
        sig_S = self.cfg.sigma_surface
        ds = torch.diff(s_mid, prepend=s_mid[:1] - (s_mid[1] - s_mid[0]))
        sig_F = torch.clamp(self.cfg.sigma_fog_scale * ds, min=sig_S)   # [K]
        centers = torch.cat([D[:, None], s_mid[None, :].expand(D.shape[0], -1)], dim=1)  # [R, K+1]
        sigs = torch.cat([torch.full_like(D[:, None], sig_S), sig_F[None, :].expand(D.shape[0], -1)], dim=1)
        pi_ret = pi[:, :-1] / (pi[:, :-1].sum(dim=1, keepdim=True) + _EPS)
        log_norm = -0.5 * ((z[:, None] - centers) / sigs) ** 2 - torch.log(sigs) - 0.9189385332046727
        return -torch.logsumexp(torch.log(pi_ret + _EPS) + log_norm, dim=1)

    def drop_nll(self, pi: Tensor) -> Tensor:
        """-log pi_0 on fired rays with NO return (labels exist: range_image mask)."""
        return -torch.log(pi[:, -1] + _EPS)

    @torch.no_grad()
    def hard_select(self, E_S: Tensor, E_F: Tensor, D: Tensor, s_mid: Tensor,
                    rule: Literal["strongest", "first"] = "strongest") -> Tuple[Tensor, Tensor]:
        """Sensor-rule inference. Returns (range [R], component [R]): component 0 = surface,
        1..K = fog bin, -1 = no return (all strengths below theta). Range is EXACTLY D or a
        bin midpoint — never a blend."""
        E = torch.cat([E_S[:, None], E_F], dim=1)                    # [R, K+1]
        centers = torch.cat([D[:, None], s_mid[None, :].expand(D.shape[0], -1)], dim=1)
        if rule == "strongest":
            comp = E.argmax(dim=1)
        else:  # first: nearest component above the noise floor
            above = E > self.theta
            far = torch.full_like(centers, torch.inf)
            comp = torch.where(above, centers, far).argmin(dim=1)
        rng = centers.gather(1, comp[:, None])[:, 0]
        dropped = E.max(dim=1).values <= self.theta
        comp = torch.where(dropped, torch.full_like(comp, -1), comp)
        rng = torch.where(dropped, torch.full_like(rng, torch.inf), rng)
        return rng, comp

    def intensity(self, E_sel: Tensor, gain: Tensor | float = 1.0) -> Tensor:
        return gain * E_sel
