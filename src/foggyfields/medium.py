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
        if sq.dtype != e.dtype:  # autocast fp16 queries against fp32 edges
            sq = sq.to(e.dtype)
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

class SpatialFog(nn.Module):
    """Plan §5.1: beta(x) = beta_anchor * exp(eps * tanh(f(x)) - mu), mean-normalised so the
    GLOBAL scale stays at the anchored measurement while the field redistributes it in space.

    f(x) = torch-native multi-resolution dense feature grids (trilinear) + a 2x32 MLP; no tcnn
    dependency, deterministic. mu is the running mean of eps*tanh(f) over the queried support,
    updated with momentum (buffer; frozen in eval), so there is no second global scale.
    tv() returns an L1 total-variation estimate on random grid points (R1 regulariser).
    """

    def __init__(self, beta_anchor: float, aabb, eps: float = 1.5,
                 levels=(16, 32, 64), feat: int = 2, hidden: int = 32,
                 mu_momentum: float = 0.01):
        super().__init__()
        if beta_anchor <= 0:
            raise ValueError("beta_anchor must be > 0")
        self.register_buffer("aabb", torch.as_tensor(aabb, dtype=torch.float32))  # [2, 3]
        self.eps = eps
        self.beta_anchor = beta_anchor
        self.mu_momentum = mu_momentum
        self.grids = nn.ParameterList(
            nn.Parameter(torch.zeros(1, feat, r, r, r)) for r in levels)   # zero-init -> beta == anchor
        self.mlp = nn.Sequential(nn.Linear(feat * len(levels), hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        # small (not zero) last-layer weights: beta starts ~anchor AND gradients flow
        nn.init.normal_(self.mlp[-1].weight, std=1e-3); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("mu", torch.zeros(()))
        self.frozen_zero = False  # API parity with HomogeneousFog

    def _phi(self, x: Tensor) -> Tensor:
        """x [..., 3] world -> eps*tanh(f(x)) [..., 1]."""
        lo, hi = self.aabb[0], self.aabb[1]
        g = ((x - lo) / (hi - lo)).clamp(0, 1) * 2 - 1          # [-1, 1]
        sh = g.shape[:-1]
        g5 = g.reshape(1, -1, 1, 1, 3)
        feats = [torch.nn.functional.grid_sample(
            gr, g5, mode="bilinear", align_corners=True, padding_mode="border"
        ).reshape(gr.shape[1], -1).T for gr in self.grids]
        out = self.mlp(torch.cat(feats, dim=-1))
        return (self.eps * torch.tanh(out)).reshape(*sh, 1)

    def forward(self, positions: Tensor) -> Tensor:
        phi = self._phi(positions)
        if self.training:
            with torch.no_grad():
                self.mu.lerp_(phi.mean().to(self.mu.dtype), self.mu_momentum)  # dtype-safe (autocast fp16 / test fp64)
        return self.beta_anchor * torch.exp(phi - self.mu)

    def _pre(self, x: Tensor) -> Tensor:
        lo, hi = self.aabb[0], self.aabb[1]
        g = ((x - lo) / (hi - lo)).clamp(0, 1) * 2 - 1
        g5 = g.reshape(1, -1, 1, 1, 3)
        feats = [torch.nn.functional.grid_sample(
            gr, g5, mode="bilinear", align_corners=True, padding_mode="border"
        ).reshape(gr.shape[1], -1).T for gr in self.grids]
        return self.mlp(torch.cat(feats, dim=-1))

    def tv(self, n: int = 4096, delta: float = 2.0) -> Tensor:
        """R1': TV smoothness + tanh-saturation penalty. Without the second term the optimizer
        drifts the pre-activation to saturation (tanh' -> 0), freezing the field FLAT while the
        parameters blow up (observed in P3 round 5: grids absmax 7, output constant)."""
        lo, hi = self.aabb[0], self.aabb[1]
        x = lo + (hi - lo) * torch.rand(n, 3, device=lo.device)
        d = torch.nn.functional.normalize(torch.randn(n, 3, device=lo.device), dim=-1) * delta
        tv = (self(x + d) - self(x)).abs().mean() / delta
        act = (self._pre(x) ** 2).mean()
        return tv + 0.1 * act
