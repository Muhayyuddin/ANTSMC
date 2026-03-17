#!/usr/bin/env python3
"""
Active Disturbance Rejection Controller (ADRC) for MBZIRC-style USV.

ADRC uses an Extended State Observer (ESO) to estimate and cancel the
total disturbance, combined with a PD state-feedback controller:

  Plant per channel:
    ẏ_i = -a_i y_i + b_i τ_i + δ_i(t)

  ESO (2nd-order, Gao's bandwidth parameterisation):
    x̂̇_{i,1} = -a_i x̂_{i,1} + b_i τ_i + x̂_{i,2} + β₁(y_i - x̂_{i,1})
    x̂̇_{i,2} = β₂(y_i - x̂_{i,1})
    where β₁ = 2ω_o,  β₂ = ω_o²

  Control law (model-based + disturbance cancellation):
    τ_i = (1/b_i)·[ -k_p_i·e_i - k_d_i·ė_i + a_i·y_i - x̂_{i,2} ]

  This is the standard ADRC formulation (Han, 1999; Gao, 2003).

Usage (standalone):
    python usv_adrc_sim.py

Dependencies: numpy, matplotlib, usv_common
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_common import (
    CONFIG, wrap_angle,
    run_simulation, plot_results, print_summary,
)


# =============================================================================
# Per-channel ESO (2nd order)
# =============================================================================

class ESO2:
    """2nd-order Extended State Observer with Gao's bandwidth parameterisation.

    For plant:  ẏ = -a·y + b·τ + δ(t)
    ESO:
        x̂̇₁ = -a·x̂₁ + b·τ + x̂₂ + 2ω_o·(y - x̂₁)
        x̂̇₂ = ω_o²·(y - x̂₁)

    Returns x̂₂ = δ̂ (estimated total disturbance).

    Uses RK4 integration for numerical stability at large dt.
    (Euler is unstable when β₂·dt = ω_o²·dt > 1, which occurs
    at ω_o = 15 rad/s, dt = 0.05 s → β₂·dt = 11.25.)
    """

    def __init__(self, a_i, b_i, omega_o):
        self.a_i = a_i
        self.b_i = b_i
        self.omega_o = omega_o
        self.beta1 = 2.0 * omega_o
        self.beta2 = omega_o ** 2
        self.xhat1 = 0.0   # state estimate
        self.xhat2 = 0.0   # disturbance estimate

    def reset(self, y0=0.0):
        self.xhat1 = y0
        self.xhat2 = 0.0

    def _eso_rhs(self, xhat, y_i, tau_i):
        """ESO continuous-time derivative."""
        e = y_i - xhat[0]
        dxhat1 = -self.a_i * xhat[0] + self.b_i * tau_i + xhat[1] + self.beta1 * e
        dxhat2 = self.beta2 * e
        return np.array([dxhat1, dxhat2])

    def update(self, y_i, tau_i, dt):
        """One RK4 step. Returns δ̂ = x̂₂."""
        xhat = np.array([self.xhat1, self.xhat2])
        k1 = self._eso_rhs(xhat, y_i, tau_i)
        k2 = self._eso_rhs(xhat + 0.5 * dt * k1, y_i, tau_i)
        k3 = self._eso_rhs(xhat + 0.5 * dt * k2, y_i, tau_i)
        k4 = self._eso_rhs(xhat + dt * k3, y_i, tau_i)
        xhat = xhat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.xhat1 = xhat[0]
        self.xhat2 = xhat[1]
        return self.xhat2


# =============================================================================
# ADRC Controller Defaults
# =============================================================================

ADRC_DEFAULTS = {
    # ── Tuning methodology (Gao, 2003; Herbst, 2013) ──
    # The ESO observer bandwidth ω_o should be 3–5× the controller
    # closed-loop bandwidth ω_c (Gao's rule of thumb).  With sensor
    # noise, higher ω_o amplifies noise in the disturbance estimate.
    #
    # Controller bandwidth: ω_c = √(kp) ≈ 0.84 rad/s (surge), 0.71 (yaw)
    # Recommended: ω_o = 3–5 × ω_c ≈ 2.5–4.2 rad/s
    # We use ω_o = 4.0 (surge) and 3.5 (yaw) — a moderate 5× and 5×.
    # Previous ω_o = 15 was ~20× controller bandwidth → excessive noise
    # amplification and ESO transient-fighting on path switches.
    #
    # PD gains designed for second-order response:
    #   ω_n = √(kp), ζ = kd / (2·ω_n) ≈ 0.7
    #
    # ESO disturbance estimate is clamped to the maximum physically
    # realisable disturbance to prevent transient overshoot.

    # ESO bandwidths (reduced from 15 → 4.0/3.5 for noise robustness)
    "omega_o_u": 4.0,      # surge ESO bandwidth [rad/s]
    "omega_o_r": 3.5,      # yaw ESO bandwidth [rad/s]

    # PD gains — surge channel (acceleration-space: τ = (1/b)·[...])
    # ω_n = sqrt(kp_u) ≈ 0.84 rad/s, ζ = kd_u / (2·ω_n) ≈ 0.7
    "kp_u": 0.7,           # proportional gain on surge error
    "kd_u": 1.2,           # derivative/damping gain

    # PD gains — yaw channel (acceleration-space)
    # ω_n = sqrt(kp_psi) ≈ 0.71 rad/s, ζ = kd_r / (2·ω_n) ≈ 0.7
    "kp_psi": 0.5,         # proportional gain on heading error
    "kd_r": 1.0,           # derivative gain on yaw rate error
    "ky_e": 0.12,          # cross-track error gain (slightly reduced)
                            # Effective: I_r * ky_e = 2000*0.12 = 240 N·m/m

    # r_d computation
    "tau_f": 0.3,           # filter time constant [s] (increased for noise)
    "r_d_max": 1.5,         # max desired yaw rate [rad/s]

    # Disturbance estimate clamp (prevents ESO transient overshoot)
    "delta_hat_max_u": 0.5,   # max |δ̂_u| [m/s²] — physical max ≈ 0.43
    "delta_hat_max_r": 0.15,  # max |δ̂_r| [rad/s²] — physical max ≈ 0.075
}


# =============================================================================
# ADRC Controller
# =============================================================================

class ADRCController:
    """Active Disturbance Rejection Controller for USV surge + yaw.

    Two independent ADRC channels:

    **Surge**:
        e_u = u - u_d
        τ_u = (1/b_u)·[-kp_u·e_u + a_u·u_d - δ̂_u]

    **Yaw**:
        e_ψ = wrap(ψ - ψ_d)
        r_d  = d(ψ_d)/dt   (finite difference, filtered)
        τ_r = (1/b_r)·[-kp_ψ·e_ψ - kd_r·(r - r_d) - ky_e·y_e + a_r·r_d - δ̂_r]

    ESOs estimate the total disturbance δ̂ per channel.
    The yaw channel uses yaw rate error (r − r_d) instead of raw r, so that
    the desired turn rate on curved paths is not penalised.
    """

    def __init__(
        self,
        a_u=None, a_r=None, b_u=None, b_r=None,
        omega_o_u=None, omega_o_r=None,
        kp_u=None, kd_u=None,
        kp_psi=None, kd_r=None, ky_e=None,
    ):
        self.a_u = a_u if a_u is not None else CONFIG["a_u"]
        self.a_r = a_r if a_r is not None else CONFIG["a_r"]
        self.b_u = b_u if b_u is not None else 1.0 / CONFIG["m_u"]
        self.b_r = b_r if b_r is not None else 1.0 / CONFIG["I_r"]

        oo_u = omega_o_u if omega_o_u is not None else ADRC_DEFAULTS["omega_o_u"]
        oo_r = omega_o_r if omega_o_r is not None else ADRC_DEFAULTS["omega_o_r"]

        self.kp_u = kp_u if kp_u is not None else ADRC_DEFAULTS["kp_u"]
        self.kd_u = kd_u if kd_u is not None else ADRC_DEFAULTS["kd_u"]
        self.kp_psi = kp_psi if kp_psi is not None else ADRC_DEFAULTS["kp_psi"]
        self.kd_r = kd_r if kd_r is not None else ADRC_DEFAULTS["kd_r"]
        self.ky_e = ky_e if ky_e is not None else ADRC_DEFAULTS["ky_e"]

        # Disturbance clamp limits
        self.delta_hat_max_u = ADRC_DEFAULTS["delta_hat_max_u"]
        self.delta_hat_max_r = ADRC_DEFAULTS["delta_hat_max_r"]

        # r_d computation
        self.tau_f_rd = ADRC_DEFAULTS["tau_f"]
        self.r_d_max = ADRC_DEFAULTS["r_d_max"]

        # ESOs
        self.eso_u = ESO2(self.a_u, self.b_u, oo_u)
        self.eso_r = ESO2(self.a_r, self.b_r, oo_r)

        # Previous applied torques for ESO (post-allocation)
        self._tau_u_prev = 0.0
        self._tau_r_prev = 0.0
        self._dt = CONFIG["dt"]

        # Yaw rate reference from psi_d differentiation
        self._psi_d_prev = None
        self._r_d = 0.0

    def compute_control(
        self,
        u: float, r: float, y_e: float,
        psi: float, psi_d: float, u_d: float, gamma: float,
    ):
        """Compute ADRC control.

        Returns
        -------
        (tau_u, tau_r) : tuple of floats
        """
        dt = self._dt

        # Compute desired yaw rate from psi_d changes (filtered finite difference)
        if self._psi_d_prev is not None:
            r_d_raw = wrap_angle(psi_d - self._psi_d_prev) / max(dt, 1e-12)
            r_d_raw = np.clip(r_d_raw, -self.r_d_max, self.r_d_max)
            # First-order low-pass filter
            alpha_f = dt / (self.tau_f_rd + dt)
            self._r_d = self._r_d + alpha_f * (r_d_raw - self._r_d)
        else:
            self._r_d = 0.0
        self._psi_d_prev = psi_d
        r_d = self._r_d

        # ESO updates (use previous actual torques for correct estimation)
        delta_hat_u = self.eso_u.update(u, self._tau_u_prev, dt)
        delta_hat_r = self.eso_r.update(r, self._tau_r_prev, dt)

        # Clamp disturbance estimates to physical limits (Herbst, 2013)
        # Prevents ESO transient overshoot during path switches
        delta_hat_u = float(np.clip(delta_hat_u,
                                    -self.delta_hat_max_u, self.delta_hat_max_u))
        delta_hat_r = float(np.clip(delta_hat_r,
                                    -self.delta_hat_max_r, self.delta_hat_max_r))

        # Surge channel
        e_u = u - u_d
        tau_u = (1.0 / self.b_u) * (
            -self.kp_u * e_u
            - self.kd_u * e_u          # derivative term (e_u ≈ ė for first-order plant)
            + self.a_u * u_d           # model-based: cancel damping at desired speed
            - delta_hat_u              # ESO disturbance rejection
        )

        # Yaw channel — uses (r - r_d) so desired turn rate is not penalised
        e_psi = wrap_angle(psi - psi_d)
        e_r = r - r_d
        tau_r = (1.0 / self.b_r) * (
            -self.kp_psi * e_psi
            - self.kd_r * e_r          # yaw rate ERROR (not raw r)
            - self.ky_e * y_e
            + self.a_r * r_d           # model-based: cancel damping at desired yaw rate
            - delta_hat_r              # ESO disturbance rejection
        )

        # Store for next ESO step
        self._tau_u_prev = tau_u
        self._tau_r_prev = tau_r

        return float(tau_u), float(tau_r)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MBZIRC USV — ADRC Controller")
    print("=" * 60)

    path_types = ["custom", "circular", "rectangular", "zigzag"]
    t_final_map = {
        "custom": 300.0, "circular": 350.0,
        "rectangular": 350.0, "zigzag": 300.0,
    }

    ctrl = ADRCController()

    for ptype in path_types:
        print(f"\n{'─' * 50}")
        print(f"  Path: {ptype.upper()}")
        print(f"{'─' * 50}")

        data = run_simulation(
            controller=ctrl,
            config={"path_type": ptype, "t_final": t_final_map[ptype]},
            controller_name="ADRC",
        )
        print_summary(data)

    print("\nDone.")
