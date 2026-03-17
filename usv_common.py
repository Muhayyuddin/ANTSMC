#!/usr/bin/env python3
"""
Shared infrastructure for MBZIRC-style USV simulations.

Contains:
  - CONFIG dict with all default parameters
  - Path generators (custom, circular, rectangular, zigzag)
  - Disturbance functions
  - Angle wrapping utility
  - USVDynamics class
  - LOSGuidance class
  - LQR Riccati solver helper (compute_lqr_gains)
  - RK4 integrator
  - Generic simulation runner (run_simulation) that accepts any controller
  - Single-controller plot helper (plot_results)

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# =============================================================================
# Configuration / Default Parameters
# =============================================================================

CONFIG = {
    # --- USV physical parameters ---
    "m_u": 700.0,          # effective surge mass [kg]
    "m_v": 800.0,          # effective sway mass [kg] (including added mass)
    "I_r": 2000.0,         # effective yaw inertia [kg·m²]
    "a_u": 0.1,            # surge damping coefficient [1/s]
    "a_v": 0.15,           # sway damping coefficient [1/s]
    "a_r": 0.2,            # yaw damping coefficient [1/s]
    "L_thruster": 1.348,   # half-distance between thrusters [m]
    "F_min": -1000.0,      # minimum thruster force [N]
    "F_max": 1000.0,       # maximum thruster force [N]

    # --- Environmental disturbances (acceleration units) ---
    # Constant bias (wind + current mean)
    "d_u_const": 0.10,     # constant surge disturbance [m/s²]
    "d_v_const": 0.06,     # constant sway disturbance [m/s²]
    "d_r_const": 0.04,     # constant yaw disturbance  [rad/s²]
    # Time-varying wave component
    "d_u_wave_amp": 0.05,  # wave-induced surge amplitude [m/s²]
    "d_v_wave_amp": 0.04,  # wave-induced sway amplitude [m/s²]
    "d_r_wave_amp": 0.02,  # wave-induced yaw amplitude [rad/s²]
    "d_wave_freq": 0.5,    # wave frequency [rad/s]
    # Ocean current (body-frame effect via kinematics)
    "V_current": 0.50,      # current speed [m/s]
    "beta_current": np.deg2rad(45.0),  # current direction [rad] (NE)
    # Wind gusts
    "F_wind_u": 300.0,     # mean wind force surge [N]  (≈ 0.43 m/s² on 700 kg)
    "F_wind_v": 200.0,     # mean wind force sway [N]   (≈ 0.25 m/s² on 800 kg)
    "F_wind_r": 150.0,     # mean wind moment yaw [N·m] (≈ 0.075 rad/s² on 2000 kg·m²)
    "wind_gust_amp_u": 120.0,  # gust amplitude surge [N]
    "wind_gust_amp_v": 80.0,   # gust amplitude sway [N]
    "wind_gust_amp_r": 60.0,   # gust amplitude yaw [N·m]
    "wind_gust_freq": 0.3,    # gust frequency [rad/s]

    # --- Sensor noise ---
    # Additive white Gaussian noise on measured states
    "sensor_noise_enabled": True,
    "noise_std_pos": 0.5,     # position (x, y) noise std [m] — typical DGPS
    "noise_std_psi": 0.02,    # heading noise std [rad] (~1.1°) — typical IMU
    "noise_std_u": 0.05,      # surge speed noise std [m/s]
    "noise_std_v": 0.05,      # sway speed noise std [m/s]
    "noise_std_r": 0.005,     # yaw rate noise std [rad/s]

    # --- Model uncertainty ---
    # Multiplicative uncertainty on plant parameters (used by dynamics)
    "param_uncertainty_enabled": True,
    "uncertainty_m_u": 0.10,   # ±10% mass uncertainty
    "uncertainty_m_v": 0.10,   # ±10% sway mass uncertainty
    "uncertainty_I_r": 0.10,   # ±10% inertia uncertainty
    "uncertainty_a_u": 0.15,   # ±15% surge damping uncertainty
    "uncertainty_a_v": 0.15,   # ±15% sway damping uncertainty
    "uncertainty_a_r": 0.15,   # ±15% yaw damping uncertainty

    # --- LOS guidance ---
    "Delta": 15.0,         # look-ahead distance [m]
    "k_y": 1.0,            # cross-track gain
    "u_min": 0.5,          # minimum desired speed [m/s]
    "u_max": 2.0,          # maximum desired speed [m/s]
    "beta": 0.05,          # speed–error decay rate
    "use_ilos": False,     # enable integral LOS
    "k_I": 0.05,           # integral LOS gain

    # --- LQR weighting ---
    # Bryson's rule (Bryson & Ho, 1975): Q_ii = 1/max_error_i², R_jj = 1/max_control_j²
    # State: [tilde_u, r, y_e, chi_e]
    # With 3-DOF sway coupling + sensor noise, cross-track is harder to reject.
    # Increased Q33 (y_e weight) and Q44 (chi_e weight) to improve
    # disturbance rejection at the cost of higher control effort.
    #   max acceptable: e_u<0.15 → Q11=1/0.15²≈44 → 50
    #   max acceptable: r<0.13 → Q22=1/0.13²≈59 → 80
    #   max acceptable: y_e<0.1 → Q33=1/0.1²=100 → 400 (aggressive)
    #   max acceptable: chi_e<0.08 → Q44=1/0.08²≈156 → 200
    # R unchanged: R11=0.001 (surge ~unconstrained), R22=0.008 (yaw moderate)
    "Q": np.diag([50.0, 80.0, 400.0, 200.0]),
    "R": np.diag([0.001, 0.005]),
    "u_d_nom": 1.5,        # nominal design speed for LQR [m/s]

    # --- LQR-SMC parameters ---
    # K_s switching gains: Utkin (2009) — must exceed max disturbance bound
    # Surge: max disturbance ≈ 0.43 m/s² × m_u = 301 N → K_s11 = 150
    # Yaw: max disturbance ≈ 0.075 rad/s² × I_r = 150 N·m → K_s22 = 250
    "K_s": np.diag([150.0, 250.0]),
    "eps": np.array([0.5, 0.5]),

    # --- Simulation ---
    "dt": 0.05,
    "t_final": 300.0,
    "state0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # [x, y, psi, u, v, r]

    # --- Waypoints (default custom path) ---
    "waypoints": np.array([
        [0.0,   0.0],
        [80.0,  0.0],
        [120.0, 60.0],
        [200.0, 60.0],
        [240.0, 0.0],
        [300.0, 0.0],
    ]),

    # --- Path type selector ---
    "path_type": "custom",
}


# =============================================================================
# Path Generators
# =============================================================================

def generate_circular_path(
    center: tuple = (100.0, 100.0),
    radius: float = 60.0,
    n_points: int = 64,
) -> np.ndarray:
    """Generate waypoints approximating a circular path.

    Parameters
    ----------
    center : (float, float)
    radius : float [m]
    n_points : int

    Returns
    -------
    ndarray, shape (n_points + 1, 2)
    """
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    return np.column_stack([xs, ys])


def generate_rectangular_path(
    origin: tuple = (0.0, 0.0),
    width: float = 200.0,
    height: float = 80.0,
) -> np.ndarray:
    """Generate waypoints for a rectangular closed-loop path.

    Parameters
    ----------
    origin : (float, float)
    width, height : float [m]

    Returns
    -------
    ndarray, shape (5, 2)
    """
    ox, oy = origin
    return np.array([
        [ox,         oy],
        [ox + width, oy],
        [ox + width, oy + height],
        [ox,         oy + height],
        [ox,         oy],
    ])


def generate_zigzag_path(
    start: tuple = (0.0, 0.0),
    n_zigs: int = 5,
    zig_length: float = 60.0,
    zig_amplitude: float = 40.0,
) -> np.ndarray:
    """Generate waypoints for a zigzag (sawtooth) path.

    Parameters
    ----------
    start : (float, float)
    n_zigs : int
    zig_length, zig_amplitude : float [m]

    Returns
    -------
    ndarray, shape (n_zigs + 1, 2)
    """
    sx, sy = start
    pts = [[sx, sy]]
    for i in range(1, n_zigs + 1):
        x = sx + i * zig_length
        y = sy + zig_amplitude * (1 if i % 2 == 1 else -1)
        pts.append([x, y])
    return np.array(pts)


def get_waypoints_for_path_type(path_type: str) -> np.ndarray:
    """Return waypoints for the requested path type."""
    if path_type == "circular":
        return generate_circular_path()
    elif path_type == "rectangular":
        return generate_rectangular_path()
    elif path_type == "zigzag":
        return generate_zigzag_path()
    elif path_type == "custom":
        return CONFIG["waypoints"]
    else:
        raise ValueError(f"Unknown path type: {path_type!r}")


# =============================================================================
# JONSWAP Wave Spectrum — Stochastic Disturbance Model
# =============================================================================

class JONSWAPDisturbance:
    """Stochastic wave disturbance based on the JONSWAP spectrum.

    Generates wave-induced force/moment time series by superposing N
    harmonic components whose amplitudes are drawn from the JONSWAP
    spectral density, with uniformly random phases.

    The JONSWAP spectrum (Hasselmann et al., 1973) is:
        S(ω) = (αg²/ω⁵) exp(-5/4 (ωp/ω)⁴) γ^exp(-(ω-ωp)²/(2σ²ωp²))

    where:
        αg  = generalised Phillips constant ≈ 0.0081
        ωp  = 2π/Tp  peak frequency
        γ   = peak enhancement factor (3.3 typical for North Sea)
        σ   = 0.07 if ω ≤ ωp, 0.09 if ω > ωp

    Reference: DNV-RP-C205 "Environmental Conditions and Environmental
    Loads" (2019); Fossen (2011), Ch. 8.

    Parameters
    ----------
    Hs : float
        Significant wave height [m].
    Tp : float
        Peak period [s].
    gamma : float
        Peak enhancement factor (default 3.3).
    N : int
        Number of spectral components (default 50).
    seed : int
        Random seed for reproducibility.
    force_scale_u : float
        Conversion factor from wave elevation [m] to surge force
        acceleration [m/s²] (RAO-like transfer).
    force_scale_v : float
        Same for sway.
    moment_scale_r : float
        Same for yaw moment acceleration [rad/s²].
    """

    def __init__(self, Hs, Tp, gamma=3.3, N=50, seed=42,
                 force_scale_u=0.065, force_scale_v=0.048,
                 moment_scale_r=0.024):
        self.Hs = Hs
        self.Tp = Tp
        self.gamma = gamma
        self.N = N
        self.force_scale_u = force_scale_u
        self.force_scale_v = force_scale_v
        self.moment_scale_r = moment_scale_r

        # Frequency range: 0.2ωp to 3.0ωp
        omega_p = 2.0 * np.pi / Tp
        self.omega_p = omega_p
        omega_min = 0.2 * omega_p
        omega_max = 3.0 * omega_p
        self.omegas = np.linspace(omega_min, omega_max, N)
        d_omega = self.omegas[1] - self.omegas[0]

        # Compute JONSWAP spectral density at each component
        alpha_pm = 0.0081  # Phillips constant
        g = 9.81
        S = np.zeros(N)
        for i, w in enumerate(self.omegas):
            sigma = 0.07 if w <= omega_p else 0.09
            # PM base spectrum
            S_pm = (alpha_pm * g**2 / w**5) * np.exp(-1.25 * (omega_p / w)**4)
            # JONSWAP peak enhancement
            r_exp = np.exp(-0.5 * ((w - omega_p) / (sigma * omega_p))**2)
            S[i] = S_pm * gamma**r_exp

        # Normalise to match Hs: Hs = 4 sqrt(m0), m0 = ∫S dω
        m0 = np.sum(S) * d_omega
        Hs_computed = 4.0 * np.sqrt(m0) if m0 > 0 else 1e-6
        scale_factor = (Hs / Hs_computed)**2
        S *= scale_factor

        # Component amplitudes
        self.amplitudes = np.sqrt(2.0 * S * d_omega)

        # Random phases (3 independent sets for surge, sway, yaw)
        rng = np.random.RandomState(seed)
        self.phases_u = rng.uniform(0, 2 * np.pi, N)
        self.phases_v = rng.uniform(0, 2 * np.pi, N)
        self.phases_r = rng.uniform(0, 2 * np.pi, N)

    def wave_elevation(self, t):
        """Compute stochastic wave elevation η(t) [m]."""
        return np.sum(self.amplitudes * np.sin(self.omegas * t + self.phases_u))

    def force_u(self, t):
        """Surge wave force acceleration [m/s²]."""
        return self.force_scale_u * np.sum(
            self.amplitudes * np.sin(self.omegas * t + self.phases_u))

    def force_v(self, t):
        """Sway wave force acceleration [m/s²]."""
        return self.force_scale_v * np.sum(
            self.amplitudes * np.sin(self.omegas * t + self.phases_v))

    def moment_r(self, t):
        """Yaw moment acceleration [rad/s²]."""
        return self.moment_scale_r * np.sum(
            self.amplitudes * np.sin(self.omegas * t + self.phases_r))


# Singleton JONSWAP instance (created on demand)
_jonswap_instance = None


def get_jonswap_instance():
    """Return (and cache) the JONSWAP disturbance generator."""
    global _jonswap_instance
    if _jonswap_instance is None:
        # Default: SS3-equivalent (Hs=1.0m, Tp=6s)
        _jonswap_instance = JONSWAPDisturbance(Hs=1.0, Tp=6.0, seed=42)
    return _jonswap_instance


def set_jonswap_params(Hs, Tp, gamma=3.3, seed=42):
    """Configure JONSWAP disturbance for a specific sea state."""
    global _jonswap_instance
    _jonswap_instance = JONSWAPDisturbance(Hs=Hs, Tp=Tp, gamma=gamma, seed=seed)


def disturbance_u_jonswap(t, **kwargs):
    """Surge disturbance using JONSWAP spectrum + wind + current bias."""
    c = CONFIG
    # Stochastic wave component (replaces deterministic sinusoid)
    d = get_jonswap_instance().force_u(t)
    # Constant bias (current drag)
    d += c["d_u_const"]
    # Wind force → acceleration
    d += (c["F_wind_u"] + c["wind_gust_amp_u"] * np.sin(
        c["wind_gust_freq"] * t + 1.2)) / c["m_u"]
    return d


def disturbance_v_jonswap(t, **kwargs):
    """Sway disturbance using JONSWAP spectrum + wind + current bias."""
    c = CONFIG
    d = get_jonswap_instance().force_v(t)
    d += c["d_v_const"]
    d += (c["F_wind_v"] + c["wind_gust_amp_v"] * np.sin(
        c["wind_gust_freq"] * t + 0.8)) / c["m_v"]
    return d


def disturbance_r_jonswap(t, **kwargs):
    """Yaw disturbance using JONSWAP spectrum + wind + current bias."""
    c = CONFIG
    d = get_jonswap_instance().moment_r(t)
    d += c["d_r_const"]
    d += (c["F_wind_r"] + c["wind_gust_amp_r"] * np.sin(
        c["wind_gust_freq"] * t + 2.5)) / c["I_r"]
    return d

# =============================================================================
# Disturbance Functions — Unified Realistic Model
# =============================================================================

def disturbance_u(t: float, **kwargs) -> float:
    """Surge disturbance at time *t* [m/s²].

    Combines:
      1. Constant bias (ocean current mean + drag offset)
      2. Wave-induced oscillation
      3. Wind force (mean + gusts) / mass

    All controllers face the SAME disturbance for fair comparison.
    """
    c = CONFIG
    # Constant bias
    d = c["d_u_const"]
    # Wave component
    d += c["d_u_wave_amp"] * np.sin(c["d_wave_freq"] * t)
    # Wind force → acceleration
    d += (c["F_wind_u"] + c["wind_gust_amp_u"] * np.sin(c["wind_gust_freq"] * t + 1.2)) / c["m_u"]
    return d


def disturbance_v(t: float, **kwargs) -> float:
    """Sway disturbance at time *t* [m/s²].

    Combines:
      1. Constant bias (lateral current + wind)
      2. Wave-induced oscillation (90° phase shift from surge)
      3. Wind force (mean + gusts) / sway mass
    """
    c = CONFIG
    d = c["d_v_const"]
    d += c["d_v_wave_amp"] * np.sin(c["d_wave_freq"] * t + 1.57)
    d += (c["F_wind_v"] + c["wind_gust_amp_v"] * np.sin(c["wind_gust_freq"] * t + 0.8)) / c["m_v"]
    return d


def disturbance_r(t: float, **kwargs) -> float:
    """Yaw disturbance at time *t* [rad/s²].

    Combines:
      1. Constant bias
      2. Wave-induced oscillation
      3. Wind moment (mean + gusts) / inertia
    """
    c = CONFIG
    d = c["d_r_const"]
    d += c["d_r_wave_amp"] * np.sin(c["d_wave_freq"] * t + 0.7)
    d += (c["F_wind_r"] + c["wind_gust_amp_r"] * np.sin(c["wind_gust_freq"] * t + 2.5)) / c["I_r"]
    return d


# =============================================================================
# Angle wrapping
# =============================================================================

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
# USV Dynamics
# =============================================================================

class USVDynamics:
    """3-DOF (surge + sway + yaw) USV dynamics plus planar kinematics.

    State vector: [x, y, psi, u, v, r]
    Control: (tau_u, tau_r) — sway is unactuated (underactuated USV).

    Dynamics:
      x_dot   = u cos(psi) - v sin(psi) + V_c cos(beta_c)
      y_dot   = u sin(psi) + v cos(psi) + V_c sin(beta_c)
      psi_dot = r
      u_dot   = -a_u u + (m_v/m_u) v r + b_u tau_u + d_u(t)
      v_dot   = -a_v v - (m_u/m_v) u r + d_v(t)
      r_dot   = -a_r r + ((m_u - m_v)/I_r) u v + b_r tau_r + d_r(t)

    The Coriolis-like coupling terms arise from the added-mass asymmetry
    and capture sideslip dynamics critical for realistic USV behaviour.
    """

    def __init__(
        self,
        m_u: float = None, m_v: float = None, I_r: float = None,
        a_u: float = None, a_v: float = None, a_r: float = None,
        L: float = None,
        F_min: float = None, F_max: float = None,
        dist_u_func=None, dist_v_func=None, dist_r_func=None,
        apply_uncertainty: bool = None,
        rng_seed: int = 42,
    ):
        self.m_u = m_u if m_u is not None else CONFIG["m_u"]
        self.m_v = m_v if m_v is not None else CONFIG["m_v"]
        self.I_r = I_r if I_r is not None else CONFIG["I_r"]
        self.a_u = a_u if a_u is not None else CONFIG["a_u"]
        self.a_v = a_v if a_v is not None else CONFIG["a_v"]
        self.a_r = a_r if a_r is not None else CONFIG["a_r"]
        self.L = L if L is not None else CONFIG["L_thruster"]
        self.F_min = F_min if F_min is not None else CONFIG["F_min"]
        self.F_max = F_max if F_max is not None else CONFIG["F_max"]
        self.dist_u = dist_u_func if dist_u_func is not None else disturbance_u
        self.dist_v = dist_v_func if dist_v_func is not None else disturbance_v
        self.dist_r = dist_r_func if dist_r_func is not None else disturbance_r

        # Apply parametric uncertainty to the TRUE plant
        # Controllers still use nominal values from CONFIG
        apply_unc = apply_uncertainty if apply_uncertainty is not None else CONFIG.get("param_uncertainty_enabled", False)
        if apply_unc:
            rng = np.random.RandomState(rng_seed)
            self.m_u *= (1.0 + CONFIG["uncertainty_m_u"] * (2.0 * rng.rand() - 1.0))
            self.m_v *= (1.0 + CONFIG["uncertainty_m_v"] * (2.0 * rng.rand() - 1.0))
            self.I_r *= (1.0 + CONFIG["uncertainty_I_r"] * (2.0 * rng.rand() - 1.0))
            self.a_u *= (1.0 + CONFIG["uncertainty_a_u"] * (2.0 * rng.rand() - 1.0))
            self.a_v *= (1.0 + CONFIG["uncertainty_a_v"] * (2.0 * rng.rand() - 1.0))
            self.a_r *= (1.0 + CONFIG["uncertainty_a_r"] * (2.0 * rng.rand() - 1.0))

        self.b_u = 1.0 / self.m_u
        self.b_r = 1.0 / self.I_r

    def state_derivative(self, state: np.ndarray, control: tuple, t: float) -> np.ndarray:
        """Time derivative of [x, y, psi, u, v, r]."""
        assert state.shape == (6,)
        x, y, psi, u, v, r = state
        tau_u, tau_r = control
        d_u = self.dist_u(t)
        d_v = self.dist_v(t)
        d_r = self.dist_r(t)
        V_c = CONFIG.get("V_current", 0.0)
        beta_c = CONFIG.get("beta_current", 0.0)

        # Coriolis-like coupling terms from added-mass asymmetry
        coriolis_u = (self.m_v / self.m_u) * v * r
        coriolis_v = -(self.m_u / self.m_v) * u * r
        coriolis_r = ((self.m_u - self.m_v) / self.I_r) * u * v

        return np.array([
            u * np.cos(psi) - v * np.sin(psi) + V_c * np.cos(beta_c),
            u * np.sin(psi) + v * np.cos(psi) + V_c * np.sin(beta_c),
            r,
            -self.a_u * u + coriolis_u + self.b_u * tau_u + d_u,
            -self.a_v * v + coriolis_v + d_v,                # unactuated
            -self.a_r * r + coriolis_r + self.b_r * tau_r + d_r,
        ])

    def map_tau_to_thrusters(self, tau_u: float, tau_r: float):
        """Map (tau_u, tau_r) -> saturated (F_L, F_R)."""
        F_L = 0.5 * tau_u + tau_r / (2.0 * self.L)
        F_R = 0.5 * tau_u - tau_r / (2.0 * self.L)
        F_L = float(np.clip(F_L, self.F_min, self.F_max))
        F_R = float(np.clip(F_R, self.F_min, self.F_max))
        return F_L, F_R

    def map_thrusters_to_tau(self, F_L: float, F_R: float):
        """Map (F_L, F_R) -> (tau_u, tau_r)."""
        return float(F_L + F_R), float(self.L * (F_L - F_R))


# =============================================================================
# LOS Guidance
# =============================================================================

class LOSGuidance:
    """Line-of-Sight guidance law for waypoint path following.

    Returns (u_d, psi_d, y_e, gamma) each step.
    """

    def __init__(
        self, waypoints: np.ndarray,
        Delta: float = None, k_y: float = None,
        u_min: float = None, u_max: float = None,
        beta: float = None,
        use_ilos: bool = None, k_I: float = None,
    ):
        self.waypoints = np.asarray(waypoints, dtype=float)
        assert self.waypoints.ndim == 2 and self.waypoints.shape[1] == 2
        self.n_wps = self.waypoints.shape[0]
        self.Delta = Delta if Delta is not None else CONFIG["Delta"]
        self.k_y = k_y if k_y is not None else CONFIG["k_y"]
        self.u_min = u_min if u_min is not None else CONFIG["u_min"]
        self.u_max = u_max if u_max is not None else CONFIG["u_max"]
        self.beta = beta if beta is not None else CONFIG["beta"]
        self.use_ilos = use_ilos if use_ilos is not None else CONFIG["use_ilos"]
        self.k_I = k_I if k_I is not None else CONFIG["k_I"]
        self.segment_idx = 0
        self.y_e_int = 0.0
        self._precompute_segments()

    def _precompute_segments(self):
        n_seg = self.n_wps - 1
        self.seg_tangent = np.zeros((n_seg, 2))
        self.seg_normal = np.zeros((n_seg, 2))
        self.seg_length = np.zeros(n_seg)
        self.seg_gamma = np.zeros(n_seg)
        for i in range(n_seg):
            diff = self.waypoints[i + 1] - self.waypoints[i]
            length = np.linalg.norm(diff)
            self.seg_length[i] = length
            t_vec = diff / max(length, 1e-12)
            self.seg_tangent[i] = t_vec
            self.seg_normal[i] = np.array([-t_vec[1], t_vec[0]])
            self.seg_gamma[i] = np.arctan2(diff[1], diff[0])

    def update(self, x: float, y: float, psi: float, dt: float):
        """Returns (u_d, psi_d, y_e, gamma)."""
        idx = min(self.segment_idx, self.n_wps - 2)
        p = np.array([x, y])
        p_i = self.waypoints[idx]
        t_vec = self.seg_tangent[idx]
        n_vec = self.seg_normal[idx]
        seg_len = self.seg_length[idx]
        dp = p - p_i
        s_along = float(t_vec @ dp)
        y_e = float(n_vec @ dp)

        # Segment switching
        switch_margin = 2.0
        if s_along > seg_len - switch_margin and idx < self.n_wps - 2:
            self.segment_idx += 1
            idx = self.segment_idx
            p_i = self.waypoints[idx]
            t_vec = self.seg_tangent[idx]
            n_vec = self.seg_normal[idx]
            seg_len = self.seg_length[idx]
            dp = p - p_i
            s_along = float(t_vec @ dp)
            y_e = float(n_vec @ dp)

        gamma = self.seg_gamma[idx]
        s_los = s_along + self.Delta
        p_los = p_i + s_los * t_vec
        psi_los = np.arctan2(p_los[1] - y, p_los[0] - x)

        if self.use_ilos:
            self.y_e_int += y_e * dt
            tilde_y_e = y_e + self.k_I * self.y_e_int
        else:
            tilde_y_e = y_e

        psi_d = wrap_angle(psi_los - np.arctan2(self.k_y * tilde_y_e, self.Delta))
        u_d = self.u_min + (self.u_max - self.u_min) * np.exp(-self.beta * y_e ** 2)
        return u_d, psi_d, y_e, gamma


# =============================================================================
# LQR Riccati Helper
# =============================================================================

def compute_lqr_gains(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Solve CARE and return (P, K, G).

    K = R^{-1} B^T P   (optimal gain)
    G = B^T P           (sliding manifold matrix)
    """
    P = solve_continuous_are(A, B, Q, R)
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P
    G = B.T @ P
    return P, K, G


def build_AB(a_u: float, a_r: float, b_u: float, b_r: float, u_d: float):
    """Construct the linearised error-state matrices (A, B) for given u_d.

    State: x_ctrl = [tilde_u, r, y_e, chi_e]
    Input: u_ctrl = [tau_u, tau_r]
    """
    A = np.array([
        [-a_u,  0.0,  0.0,  0.0],
        [ 0.0, -a_r,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  u_d],
        [ 0.0,  1.0,  0.0,  0.0],
    ])
    B = np.array([
        [b_u,  0.0],
        [0.0,  b_r],
        [0.0,  0.0],
        [0.0,  0.0],
    ])
    return A, B


# =============================================================================
# Saturation (boundary-layer sign)
# =============================================================================

def sigma(s: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Component-wise saturation function.

    sigma_i = s_i/eps_i  if |s_i| <= eps_i
            = sign(s_i)  otherwise
    """
    result = np.zeros_like(s)
    for i in range(s.shape[0]):
        if np.abs(s[i]) <= eps[i]:
            result[i] = s[i] / eps[i]
        else:
            result[i] = np.sign(s[i])
    return result


# =============================================================================
# RK4 Integrator
# =============================================================================

def rk4_step(f, t: float, state: np.ndarray, dt: float, control: tuple) -> np.ndarray:
    """One RK4 step: f(state, control, t) -> state_dot."""
    k1 = f(state, control, t)
    k2 = f(state + 0.5 * dt * k1, control, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, control, t + 0.5 * dt)
    k4 = f(state + dt * k3, control, t + dt)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# =============================================================================
# Generic Simulation Runner
# =============================================================================

def run_simulation(controller, config: dict = None, controller_name: str = ""):
    """Run a time-domain simulation with any controller that exposes
    ``compute_control(u, r, y_e, psi, psi_d, u_d, gamma) -> (tau_u, tau_r)``.

    Parameters
    ----------
    controller : object
        Must have a ``compute_control`` method with the signature above.
    config : dict, optional
        Overrides for CONFIG entries.
    controller_name : str
        Label stored in the returned dict for plotting.

    Returns
    -------
    dict – time histories.
    """
    cfg = {**CONFIG}
    if config is not None:
        cfg.update(config)

    # Resolve waypoints
    path_type = cfg.get("path_type", "custom")
    waypoints = get_waypoints_for_path_type(path_type)
    if config is not None and "waypoints" in config:
        waypoints = config["waypoints"]
    cfg["waypoints"] = waypoints

    # Auto initial state — 6-DOF: [x, y, psi, u, v, r]
    wp0, wp1 = waypoints[0], waypoints[1]
    init_heading = np.arctan2(wp1[1] - wp0[1], wp1[0] - wp0[0])
    if config is None or "state0" not in config:
        cfg["state0"] = np.array([wp0[0], wp0[1], init_heading, 0.0, 0.0, 0.0])

    dyn = USVDynamics(
        m_u=cfg["m_u"], m_v=cfg["m_v"], I_r=cfg["I_r"],
        a_u=cfg["a_u"], a_v=cfg["a_v"], a_r=cfg["a_r"],
        L=cfg["L_thruster"],
        F_min=cfg["F_min"], F_max=cfg["F_max"],
    )
    los = LOSGuidance(
        waypoints=cfg["waypoints"],
        Delta=cfg["Delta"], k_y=cfg["k_y"],
        u_min=cfg["u_min"], u_max=cfg["u_max"],
        beta=cfg["beta"],
        use_ilos=cfg["use_ilos"], k_I=cfg["k_I"],
    )

    dt = cfg["dt"]
    N = int(cfg["t_final"] / dt)
    state = cfg["state0"].copy()

    hist = {k: np.zeros(N) for k in
            ["t", "x", "y", "psi", "u", "v", "r", "y_e", "chi_e",
             "psi_d", "u_d", "tau_u", "tau_r", "F_L", "F_R"]}

    # Sensor noise RNG (fixed seed per simulation for reproducibility)
    noise_enabled = cfg.get("sensor_noise_enabled", False)
    noise_rng = np.random.RandomState(12345)

    def ode_rhs(s, ctrl, t_now):
        return dyn.state_derivative(s, ctrl, t_now)

    for k in range(N):
        t = k * dt
        x, y, psi, u, v, r = state

        # --- Add sensor noise to measurements fed to guidance and controller ---
        if noise_enabled:
            x_meas = x + noise_rng.randn() * cfg["noise_std_pos"]
            y_meas = y + noise_rng.randn() * cfg["noise_std_pos"]
            psi_meas = psi + noise_rng.randn() * cfg["noise_std_psi"]
            u_meas = u + noise_rng.randn() * cfg["noise_std_u"]
            r_meas = r + noise_rng.randn() * cfg["noise_std_r"]
        else:
            x_meas, y_meas, psi_meas, u_meas, r_meas = x, y, psi, u, r

        # LOS guidance uses noisy measurements
        u_d, psi_d, y_e, gamma = los.update(x_meas, y_meas, psi_meas, dt)

        # Controller uses noisy measurements
        tau_u, tau_r = controller.compute_control(
            u_meas, r_meas, y_e, psi_meas, psi_d, u_d, gamma)
        F_L, F_R = dyn.map_tau_to_thrusters(tau_u, tau_r)
        tau_u_act, tau_r_act = dyn.map_thrusters_to_tau(F_L, F_R)

        # Log TRUE states (not noisy) for performance evaluation
        # but y_e is from guidance (affected by noisy position)
        # Compute true y_e for logging
        idx = min(los.segment_idx, los.n_wps - 2)
        dp_true = np.array([x, y]) - los.waypoints[idx]
        y_e_true = float(los.seg_normal[idx] @ dp_true)

        hist["t"][k] = t
        hist["x"][k] = x;    hist["y"][k] = y
        hist["psi"][k] = psi; hist["u"][k] = u; hist["v"][k] = v; hist["r"][k] = r
        hist["y_e"][k] = y_e_true; hist["chi_e"][k] = wrap_angle(psi - gamma)
        hist["psi_d"][k] = psi_d; hist["u_d"][k] = u_d
        hist["tau_u"][k] = tau_u_act; hist["tau_r"][k] = tau_r_act
        hist["F_L"][k] = F_L; hist["F_R"][k] = F_R

        state = rk4_step(ode_rhs, t, state, dt, (tau_u_act, tau_r_act))
        state[2] = wrap_angle(state[2])

    hist["waypoints"] = cfg["waypoints"]
    hist["path_type"] = path_type
    hist["controller"] = controller_name
    return hist


# =============================================================================
# Single-controller plot helper
# =============================================================================

def plot_results(data: dict):
    """5-panel plot set for one simulation run."""
    t = data["t"]
    wps = data["waypoints"]
    label = data.get("controller", "")
    ptype = data.get("path_type", "custom").capitalize()
    title_prefix = f"{label} – {ptype}" if label else ptype

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(wps[:, 0], wps[:, 1], "rs--", ms=5, label="Reference")
    ax1.plot(data["x"], data["y"], "b-", lw=1, label="USV path")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_title(f"Trajectory – {title_prefix}")
    ax1.legend(); ax1.set_aspect("equal", adjustable="datalim"); ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, data["y_e"], "b-", lw=1)
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("y_e [m]")
    ax2.set_title(f"Cross-Track Error – {title_prefix}"); ax2.grid(True)

    fig3, (a3a, a3b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a3a.plot(t, np.degrees(data["psi"]), "b-", label=r"$\psi$")
    a3a.plot(t, np.degrees(data["psi_d"]), "r--", label=r"$\psi_d$")
    a3a.set_ylabel("Heading [deg]"); a3a.set_title(f"Heading – {title_prefix}")
    a3a.legend(); a3a.grid(True)
    a3b.plot(t, np.degrees(data["chi_e"]), "b-")
    a3b.set_xlabel("Time [s]"); a3b.set_ylabel("χ_e [deg]"); a3b.grid(True)

    fig4, (a4a, a4b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a4a.plot(t, data["tau_u"], "b-", lw=1); a4a.set_ylabel("τ_u [N]")
    a4a.set_title(f"Control Inputs – {title_prefix}"); a4a.grid(True)
    a4b.plot(t, data["tau_r"], "r-", lw=1); a4b.set_xlabel("Time [s]")
    a4b.set_ylabel("τ_r [N·m]"); a4b.grid(True)

    fig5, (a5a, a5b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a5a.plot(t, data["F_L"], "b-", lw=1, label="F_L")
    a5a.set_ylabel("F_L [N]"); a5a.set_title(f"Thrusters – {title_prefix}")
    a5a.legend(); a5a.grid(True)
    a5b.plot(t, data["F_R"], "r-", lw=1, label="F_R")
    a5b.set_xlabel("Time [s]"); a5b.set_ylabel("F_R [N]")
    a5b.legend(); a5b.grid(True)
    plt.tight_layout()


def print_summary(data: dict):
    """Print one-line performance summary."""
    label = data.get("controller", "Controller")
    ptype = data.get("path_type", "custom")
    ye = data["y_e"]
    ce = data["chi_e"]
    print(f"  [{label:10s} | {ptype:12s}]  "
          f"Max|y_e|={np.max(np.abs(ye)):7.3f} m   "
          f"RMS y_e={np.sqrt(np.mean(ye**2)):7.3f} m   "
          f"Max|χ_e|={np.degrees(np.max(np.abs(ce))):6.2f}°")
