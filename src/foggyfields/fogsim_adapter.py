"""Adapters around two vendored LiDAR fog simulators.

Vendored code (see third_party/PINNED.md for pinned commits and API notes):
  - Hahner et al., "Fog Simulation on Real LiDAR Point Clouds for 3D Object
    Detection in Adverse Weather", ICCV 2021.
    third_party/LiDAR_fog_sim @ 1cb9e07
  - Kilic et al., "Lidar Light Scattering Augmentation (LISA)", arXiv 2021.
    third_party/LISA @ ce06a7f

Both adapters share the same contract:

    fog_sweep_*(points_ego, beta, T_lidar_rig, rng_seed) -> np.ndarray

  points_ego  : [N, 6] float32 (x, y, z, intensity01, t_offset_us, ring),
                EGO/rig frame.
  beta        : extinction coefficient in 1/m (our beta_true).
  T_lidar_rig : [4, 4] rig<-lidar (p_rig = T @ p_lidar). Points are moved to
                the SENSOR frame via inv(T) before the sim (both sims compute
                range as ||xyz|| from the sensor origin) and back after.
  rng_seed    : seeds the sim's RNG for reproducibility.

  Returns [M, 6] float32 in the EGO frame, M <= N + (LISA duplicate scatter
  returns). intensity01 is mapped to each sim's units and back to [0, 1].
  t_offset_us and ring are carried through per source point. A point
  scattered to a shorter range keeps its ray direction (both sims scale xyz
  by r'/r in the sensor frame). Points lost to fog are removed.

beta mapping (details in PINNED.md):
  - Hahner: beta == ParameterSet.alpha (attenuation coefficient, 1/m).
    Derived params are passed consistently (mor = ln(20)/alpha,
    backscatter beta = 0.046/mor) because ParameterSet.__init__ computes
    them from the DEFAULT alpha=0.06 before applying kwargs. The soft-
    scatter integral table is only precomputed for alpha in
    {0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.12, 0.15, 0.2}; get_integral_dict
    snaps to the nearest one and we warn if the snap is not exact.
  - LISA: beta == the Mie-derived extinction coefficient alpha
    (atmos_models.alpha_beta), which is linear in the droplet number
    density; we scale the strong-advection-fog N(D) by beta/alpha0 so the
    extinction equals beta exactly (equivalently rho = 20 * beta / alpha0
    in the gamma drop-size distribution, alpha0 ~= 0.0595 1/m at
    lam = 905 nm, m = 1.328).
"""

import sys
import warnings
from pathlib import Path

import numpy as np

_THIRD_PARTY = Path("/capstor/scratch/cscs/yitiwang/foggyfields/third_party")
_HAHNER_DIR = _THIRD_PARTY / "LiDAR_fog_sim"
_LISA_DIR = _THIRD_PARTY / "LISA"

# Hahner integral lookup tables exist only for these alphas (1/m).
HAHNER_AVAILABLE_ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.12, 0.15, 0.2)


# ---------------------------------------------------------------------------
# frame helpers
# ---------------------------------------------------------------------------

def _ego_to_sensor(xyz_ego: np.ndarray, T_lidar_rig: np.ndarray) -> np.ndarray:
    """points_sensor = inv(T_lidar_rig) @ points_ego (T is rig<-lidar)."""
    T_inv = np.linalg.inv(np.asarray(T_lidar_rig, dtype=np.float64))
    return xyz_ego @ T_inv[:3, :3].T + T_inv[:3, 3]


def _sensor_to_ego(xyz_sensor: np.ndarray, T_lidar_rig: np.ndarray) -> np.ndarray:
    T = np.asarray(T_lidar_rig, dtype=np.float64)
    return xyz_sensor @ T[:3, :3].T + T[:3, 3]


def _check_input(points_ego: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_ego)
    if pts.ndim != 2 or pts.shape[1] != 6:
        raise ValueError(f"expected [N, 6] sweep, got {pts.shape}")
    return pts.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Hahner et al. (ICCV 2021)
# ---------------------------------------------------------------------------

def fog_sweep_hahner(points_ego: np.ndarray, beta: float,
                     T_lidar_rig: np.ndarray, rng_seed: int) -> np.ndarray:
    """Fog one sweep with the Hahner et al. ICCV 2021 simulation.

    Entry point used: fog_simulation.simulate_fog(ParameterSet, pc, noise=10)
    with pc = [N, 4] (x, y, z, intensity 0..255) in the SENSOR frame.
    Hard attenuation: I' = round(I * exp(-2 * alpha * r)).
    Soft scatter: a point whose fog response (lookup-table integral of the
    backscattered pulse) exceeds its attenuated return is moved to the fog
    response range (xyz scaled by r'/r) with the fog response as intensity,
    plus uniform range noise (noise=10, variant v1). No points are dropped,
    so N is preserved and metadata is reattached by index.
    """
    pts = _check_input(points_ego)
    beta = float(beta)

    if str(_HAHNER_DIR) not in sys.path:
        sys.path.insert(0, str(_HAHNER_DIR))
    import fog_simulation as _fs

    # Reproducibility: the module-level RNG drives the soft-scatter noise.
    _fs.RNG = np.random.default_rng(seed=int(rng_seed))

    nearest = min(HAHNER_AVAILABLE_ALPHAS, key=lambda a: abs(a - beta))
    if abs(nearest - beta) > 1e-9:
        warnings.warn(
            f"Hahner integral table snaps alpha {beta} -> {nearest}; hard "
            f"attenuation still uses exact beta={beta}.")

    xyz_sensor = _ego_to_sensor(pts[:, :3], T_lidar_rig)

    pc = np.empty((len(pts), 4), dtype=np.float64)
    pc[:, :3] = xyz_sensor
    pc[:, 3] = np.clip(pts[:, 3], 0.0, 1.0) * 255.0  # sim expects 0..255

    # ParameterSet.__init__ derives mor and the backscatter coefficient from
    # the DEFAULT alpha (0.06) before kwargs are applied; pass consistent
    # values (paper: MOR = ln(20)/alpha, beta_backscatter = 0.046/MOR).
    mor = np.log(20.0) / beta
    p = _fs.ParameterSet(alpha=beta, mor=mor, beta=0.046 / mor, gamma=1e-6)

    augmented, _, _ = _fs.simulate_fog(p, pc, noise=10, gain=False,
                                       noise_variant='v1', hard=True,
                                       soft=True)

    out = np.empty((len(pts), 6), dtype=np.float32)
    out[:, :3] = _sensor_to_ego(augmented[:, :3], T_lidar_rig)
    out[:, 3] = np.clip(augmented[:, 3] / 255.0, 0.0, 1.0)
    out[:, 4] = pts[:, 4]  # t_offset_us (N preserved, order preserved)
    out[:, 5] = pts[:, 5]  # ring
    return out


# ---------------------------------------------------------------------------
# LISA (Kilic et al. 2021)
# ---------------------------------------------------------------------------

def _lisa_alpha0_strong_advection_fog() -> float:
    """Mie-derived extinction (1/m) of LISA's unscaled strong-advection fog.

    Instantiating Lisa runs the vendored Bohren-Huffman Mie code (mie.so,
    recompiled for aarch64) over the droplet-diameter grid.
    """
    if str(_LISA_DIR) not in sys.path:
        sys.path.insert(0, str(_LISA_DIR))

    # scipy >= 1.14 removed scipy.integrate.trapz, which the vendored
    # atmos_models.py imports; shim it before the import.
    import scipy.integrate
    if not hasattr(scipy.integrate, "trapz"):
        scipy.integrate.trapz = np.trapezoid

    from pylisa.lisa import Lisa
    from pylisa.atmos_models import alpha_beta

    lisa = Lisa(atm_model='strong_advection_fog')
    alpha0, _ = alpha_beta(lisa.N_model(lisa.D), lisa.D,
                           lisa.q_ext, lisa.q_back)
    return float(alpha0)


_LISA_ALPHA0_CACHE: dict = {}


def _lisa_haze_tracked(pts_3d: np.ndarray, alpha: float):
    """Index-tracking port of Lisa.haze_point_cloud (pylisa/lisa.py:339-424).

    The vendored method concatenates kept/scattered points and cannot carry
    per-point metadata, so the identical math is mirrored here with source
    indices returned. Constants and formulas are verbatim from the vendored
    code (Velodyne HDL64S2: n=0.05, g=0.35, dmin=2), except that the
    extinction coefficient is passed in directly (the vendored code computes
    it from the fog drop-size distribution via Mie theory; we scale that
    distribution so the result equals our beta — see module docstring).

    Returns (pc_out [M, 4], labels [M], src_idx [M] indices into pts_3d).
    Labels: 0 kept (attenuated), 1 cloud scatter, 2 random scatter.
    NOTE (vendored behavior): random-scatter candidates are not excluded
    from the kept set, so a random-scattered point can appear twice (its
    attenuated original plus a near-range noise return).
    """
    n = 0.05
    g = 0.35
    dmin = 2

    d_full = np.sqrt(np.sum(pts_3d[:, :3] ** 2, axis=1))
    detectable = np.where(d_full > dmin)[0]
    d = d_full[detectable]
    pts = pts_3d[detectable]

    beta_usefull = alpha * np.ones(d.shape)
    dmax = -np.divide(np.log(np.divide(n, (pts[:, 3] + g))), 2 * beta_usefull)
    dnew = -np.log(1 - 0.5) / beta_usefull

    probability_lost = 1 - np.exp(-beta_usefull * dmax)
    lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost

    cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter),
                                    np.logical_not(lost))

    # kept (attenuated in place)
    idx_stable = np.where(d < dmax)[0]
    old_points = pts[idx_stable].copy()
    old_points[:, 3] = old_points[:, 3] * np.exp(
        -beta_usefull[idx_stable] * d[idx_stable])

    # cloud scatter: moved to dnew, intensity attenuated to dnew
    cloud_scatter_idx = np.where(np.logical_and(dmax < d, cloud_scatter))[0]
    cs = pts[cloud_scatter_idx].copy()
    scale_c = dnew[cloud_scatter_idx] / d[cloud_scatter_idx]
    cs[:, :3] *= scale_c[:, None]
    cs[:, 3] = cs[:, 3] * np.exp(
        -beta_usefull[cloud_scatter_idx] * dnew[cloud_scatter_idx])

    # random scatter: uniform range in (0, min(dmax, d)], subsampled to 5%
    random_scatter_idx = np.where(random_scatter)[0]
    scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
    drand = np.random.uniform(high=scatter_max[random_scatter_idx])
    drand_idx = np.where(drand > dmin)
    drand = drand[drand_idx]
    random_scatter_idx = random_scatter_idx[drand_idx]
    fraction_random = .05
    subsampled_idx = np.random.choice(
        len(random_scatter_idx), int(fraction_random * len(random_scatter_idx)),
        replace=False)
    drand = drand[subsampled_idx]
    random_scatter_idx = random_scatter_idx[subsampled_idx]
    rs = pts[random_scatter_idx].copy()
    scale_r = drand / d[random_scatter_idx]
    rs[:, :3] *= scale_r[:, None]
    rs[:, 3] = rs[:, 3] * np.exp(-alpha * drand)

    pc_out = np.concatenate((old_points, cs, rs), axis=0)
    labels = np.concatenate((np.zeros(len(idx_stable), dtype=np.int64),
                             np.ones(len(cloud_scatter_idx), dtype=np.int64),
                             2 * np.ones(len(random_scatter_idx), dtype=np.int64)))
    src_idx = np.concatenate((detectable[idx_stable],
                              detectable[cloud_scatter_idx],
                              detectable[random_scatter_idx]))
    return pc_out, labels, src_idx


def fog_sweep_lisa(points_ego: np.ndarray, beta: float,
                   T_lidar_rig: np.ndarray, rng_seed: int) -> np.ndarray:
    """Fog one sweep with LISA (Kilic et al. 2021), fog mode.

    Entry point used: Lisa(atm_model='strong_advection_fog') for the Mie
    extinction calibration, and the fog scattering model of
    Lisa.haze_point_cloud (ported with index tracking; see
    _lisa_haze_tracked). The native Lisa.augment (= augment_avg for fog
    models) only attenuates, jitters ranges and drops weak returns — it
    never creates near-range scatter — so haze_point_cloud is the LISA fog
    path that models backscatter returns.

    Input intensity must be in [0, 1] (LISA's convention already); ranges in
    meters from the sensor origin. Points below the 2 m detectability floor
    and points lost to fog are removed; scattered points keep their ray
    direction. Output may contain duplicate source points (vendored
    behavior: a random-scatter return does not remove the original).
    """
    pts = _check_input(points_ego)
    beta = float(beta)

    if "alpha0" not in _LISA_ALPHA0_CACHE:
        _LISA_ALPHA0_CACHE["alpha0"] = _lisa_alpha0_strong_advection_fog()
    alpha0 = _LISA_ALPHA0_CACHE["alpha0"]
    # Scaling the drop-size distribution N(D) by beta/alpha0 makes the
    # Mie-derived extinction equal beta exactly (alpha is linear in N(D)),
    # so the effective extinction fed to the fog model is beta itself.
    # (Equivalently rho = 20 * beta / alpha0 in nd_strong_advection_fog.)
    _LISA_ALPHA0_CACHE["rho_eff"] = 20.0 * beta / alpha0

    np.random.seed(int(rng_seed))  # LISA uses the global numpy RNG

    xyz_sensor = _ego_to_sensor(pts[:, :3], T_lidar_rig)
    pc = np.empty((len(pts), 4), dtype=np.float64)
    pc[:, :3] = xyz_sensor
    pc[:, 3] = np.clip(pts[:, 3], 0.0, 1.0)  # LISA expects reflectivity 0..1

    pc_out, _labels, src_idx = _lisa_haze_tracked(pc, beta)

    out = np.empty((len(pc_out), 6), dtype=np.float32)
    out[:, :3] = _sensor_to_ego(pc_out[:, :3], T_lidar_rig)
    out[:, 3] = np.clip(pc_out[:, 3], 0.0, 1.0)
    out[:, 4] = pts[src_idx, 4]  # t_offset_us
    out[:, 5] = pts[src_idx, 5]  # ring
    return out
