"""FoggyFields medium: fog extinction fields + the MediumIntegrator (plan v2 §5.2).

The integrator marches a fog field beta(x) [1/m] along rays on a fixed non-uniform grid
(denser in the first `near_range` metres), returns per-ray cumulative optical depth as a
lookup table tau(s_k), and interpolates tau(s) linearly at any depth. Gradients flow to the
field parameters through the interpolation. The scene renderer never samples the fog field.

Backbone-agnostic: works on any (origins, directions) in metric world space — NeuRAD ray
bundles and SplatAD per-pixel rays alike.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class HomogeneousFog(nn.Module):
    """beta(x) = beta_0 >= 0, one scalar (B1 of the plan: delta_phi == 0).

    beta_0 is stored as an unconstrained parameter through softplus so it stays positive;
    `beta0=0` is representable exactly via the `frozen_zero` flag (used by the zero-fog test).
    """

    def __init__(self, beta0: float = 0.02, learnable: bool = True):
        super().__init__()
        if beta0 < 0:
            raise ValueError("beta0 must be >= 0")
        self.frozen_zero = beta0 == 0.0 and not learnable
        raw = self._inv_softplus(torch.tensor(max(float(beta0), 1e-8), dtype=torch.float32))
        self.raw_beta0 = nn.Parameter(raw, requires_grad=learnable)

    @staticmethod
    def _inv_softplus(y: Tensor) -> Tensor:
        return y + torch.log(-torch.expm1(-y))

    @property
    def beta0(self) -> Tensor:
        if self.frozen_zero:
            return torch.zeros((), device=self.raw_beta0.device)
        return nn.functional.softplus(self.raw_beta0)

    def forward(self, positions: Tensor) -> Tensor:  # [..., 3] -> [..., 1]
        return self.beta0.expand(*positions.shape[:-1], 1)


class MediumLUT:
    """Per-ray cumulative optical depth table tau(s_k) with differentiable linear interpolation.

    edges: [K+1] monotonically increasing sample depths (metres), edges[0] == 0.
    tau:   [R, K+1] cumulative optical depth at the edges, tau[:, 0] == 0.
    Beyond edges[-1], tau extends linearly with the last bin's slope (exact for homogeneous
    fog; for a spatial field it assumes beta is constant past max_range). Sky rays therefore
    keep attenuating instead of freezing at tau(max_range).
    """

    def __init__(self, edges: Tensor, tau: Tensor, beta_mid: Tensor | None = None):
        self.edges = edges
        self.tau = tau
        self.beta_mid = beta_mid  # [R, K] field value at bin midpoints (for fog-echo strengths)

    def tau_at(self, s: Tensor) -> Tensor:
        """s: [R] or [R, N] depths (metres) -> tau of the same shape."""
        squeeze = s.dim() == 1
        sq = s[:, None] if squeeze else s
        e, t = self.edges, self.tau
        if sq.shape[0] != t.shape[0]:
            raise ValueError(f"s has {sq.shape[0]} rays but the LUT has {t.shape[0]}")
        idx = torch.searchsorted(e, sq.detach().clamp(min=0.0).contiguous(), right=True).clamp(1, len(e) - 1)
        e_lo, e_hi = e[idx - 1], e[idx]
        t_lo = torch.gather(t, 1, idx - 1)
        t_hi = torch.gather(t, 1, idx)
        w = (sq.clamp(min=0.0) - e_lo) / (e_hi - e_lo)
        out = t_lo + w * (t_hi - t_lo)  # linear inside; w > 1 past the end = linear extension
        return out[:, 0] if squeeze else out

    def transmittance(self, s: Tensor, two_way: bool = False) -> Tensor:
        return torch.exp((-2.0 if two_way else -1.0) * self.tau_at(s))


class MediumIntegrator(nn.Module):
    """Plan v2 §5.2: coarse uniform-per-region marching of the fog field -> per-ray tau LUT.

    Grid: `n_near` bins over [0, near_range] + `n_far` bins over [near_range, max_range]
    (denser near the sensor, where the fog halo lives). Midpoint rule per bin:
    tau(s_{k+1}) = tau(s_k) + beta(o + d * mid_k) * ds_k.
    """

    def __init__(self, field: nn.Module, max_range: float = 120.0,
                 n_near: int = 16, n_far: int = 16, near_range: float = 20.0):
        super().__init__()
        if not (0.0 < near_range < max_range):
            raise ValueError("need 0 < near_range < max_range")
        self.field = field
        near = torch.linspace(0.0, near_range, n_near + 1)
        far = torch.linspace(near_range, max_range, n_far + 1)[1:]
        self.register_buffer("edges", torch.cat([near, far]), persistent=False)

    def forward(self, origins: Tensor, directions: Tensor) -> MediumLUT:
        """origins, directions: [R, 3], directions unit-norm, metric space."""
        e = self.edges.to(origins.device, origins.dtype)
        mids = 0.5 * (e[:-1] + e[1:])                       # [K]
        ds = e[1:] - e[:-1]                                 # [K]
        pos = origins[:, None, :] + directions[:, None, :] * mids[None, :, None]  # [R, K, 3]
        beta = self.field(pos)[..., 0]                      # [R, K]
        tau = torch.cumsum(beta * ds[None, :], dim=1)       # [R, K]
        tau = torch.cat([torch.zeros_like(tau[:, :1]), tau], dim=1)  # [R, K+1]
        return MediumLUT(e, tau, beta_mid=beta)
