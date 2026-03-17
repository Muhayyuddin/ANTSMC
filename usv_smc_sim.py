#!/usr/bin/env python3
"""
Baseline Sliding Mode Controller (SMC) for MBZIRC-style USV.

A conventional SMC that uses a PD-like sliding surface on the error state
x_ctrl = [tilde_u, r, y_e, chi_e] **without** LQR-optimal gain design.

Sliding surfaces:
    s_u = c_u * tilde_u               (surge channel)
    s_r = c1 * chi_e + c2 * r + c3 * y_e   (yaw channel — couples heading,
                                              yaw-rate, and cross-track errors)

Control law:
    tau_u = -(1/b_u) [ a_u * u  + lambda_u * s_u + k_su * sigma(s_u) ]
    tau_r = -(1/b_r) [ a_r * r  + lambda_r * s_r + k_sr * sigma(s_r) ]

where sigma is the boundary-layer saturation function and lambda_i provide
linear reaching dynamics.

Usage (standalone):
    python usv_smc_sim.py

Dependencies: numpy, matplotlib, usv_common
"""

import numpy as np
import matplotlib.pyplot as plt

from usv_common import (
    CONFIG, wrap_angle, sigma,
    run_simulation, plot_results, print_summary,
)


# =============================================================================
# SMC-specific default parameters
# =============================================================================

SMC_DEFAULTS = {
    # Sliding surface coefficients (yaw channel)
    "c1": 1.0,      # weight on chi_e
    "c2": 0.8,      # weight on r
    "c3": 0.3,      # weight on y_e

    # Sliding surface coefficient (surge channel)
    "c_u": 1.0,

    # ── Actuator-aware gain design ──
    # Physical limits: tau_u ∈ [-2000, 2000] N, tau_r ∈ [-2696, 2696] N·m.
    # All gains are in the acceleration domain (multiplied by 1/b_i in the
    # control law).  The force/moment produced is gain / b_i:
    #   surge: gain_u / b_u = gain_u * 700   [N]
    #   yaw:   gain_r / b_r = gain_r * 2000  [N·m]
    #
    # Budget allocation (surge, |tau_u| ≤ 2000 N):
    #   equivalent ≈ a_u * u ≈ 0.1 * 2 / b_u ≈ 140 N
    #   reaching   : lambda_u * s_u / b_u  (must stay moderate)
    #   switching  : k_su / b_u            (must stay moderate)
    #
    # Budget allocation (yaw, |tau_r| ≤ 2696 N·m):
    #   equivalent ≈ a_r * r ≈ 0.2 * 0.5 / b_r ≈ 200 N·m
    #   reaching   : lambda_r * s_r / b_r
    #   switching  : k_sr / b_r

    # Reaching-law gains (acceleration domain)
    "lambda_u": 0.5,   # → 0.5 * s_u * 700 ≈ 350 N at s_u=1
    "lambda_r": 0.3,   # → 0.3 * s_r * 2000 ≈ 600 N·m at s_r=1

    # Discontinuous gains (acceleration domain)
    "k_su": 1.5,       # → 1.5 * 700 = 1050 N  (within surge budget)
    "k_sr": 0.8,       # → 0.8 * 2000 = 1600 N·m (within yaw budget)

    # Boundary-layer widths
    "eps_u": 0.5,
    "eps_r": 0.5,
}


# =============================================================================
# SMC Controller
# =============================================================================

class SMCController:
    """Conventional sliding-mode controller for USV surge + yaw tracking.

    Two independent sliding surfaces are defined:

    **Surge channel**
        s_u = c_u · (u - u_d)
        tau_u = -(1/b_u)·[ a_u·u + λ_u·s_u + k_su·σ(s_u) ]

    **Yaw channel** (couples heading error, yaw rate, cross-track error)
        s_r = c1·chi_e + c2·r + c3·y_e
        tau_r = -(1/b_r)·[ a_r·r + λ_r·s_r + k_sr·σ(s_r) ]

    σ is a boundary-layer saturation to reduce chattering.

    The controller output is clamped to the physical actuator limits
    (tau_u_max, tau_r_max) so that the controller cannot rely on
    downstream saturation for performance (bang-bang effect).
    """

    def __init__(
        self,
        a_u: float = None, a_r: float = None,
        b_u: float = None, b_r: float = None,
        c_u: float = None, c1: float = None, c2: float = None, c3: float = None,
        lambda_u: float = None, lambda_r: float = None,
        k_su: float = None, k_sr: float = None,
        eps_u: float = None, eps_r: float = None,
    ):
        """
        Parameters
        ----------
        a_u, a_r : float – damping coefficients [1/s]
        b_u, b_r : float – input gains (1/mass, 1/inertia)
        c_u : float – surge sliding surface coefficient
        c1, c2, c3 : float – yaw sliding surface coefficients
        lambda_u, lambda_r : float – linear reaching gains
        k_su, k_sr : float – discontinuous switching gains
        eps_u, eps_r : float – boundary-layer widths
        """
        self.a_u = a_u if a_u is not None else CONFIG["a_u"]
        self.a_r = a_r if a_r is not None else CONFIG["a_r"]
        self.b_u = b_u if b_u is not None else 1.0 / CONFIG["m_u"]
        self.b_r = b_r if b_r is not None else 1.0 / CONFIG["I_r"]

        self.c_u = c_u if c_u is not None else SMC_DEFAULTS["c_u"]
        self.c1 = c1 if c1 is not None else SMC_DEFAULTS["c1"]
        self.c2 = c2 if c2 is not None else SMC_DEFAULTS["c2"]
        self.c3 = c3 if c3 is not None else SMC_DEFAULTS["c3"]

        self.lambda_u = lambda_u if lambda_u is not None else SMC_DEFAULTS["lambda_u"]
        self.lambda_r = lambda_r if lambda_r is not None else SMC_DEFAULTS["lambda_r"]
        self.k_su = k_su if k_su is not None else SMC_DEFAULTS["k_su"]
        self.k_sr = k_sr if k_sr is not None else SMC_DEFAULTS["k_sr"]
        self.eps_u = eps_u if eps_u is not None else SMC_DEFAULTS["eps_u"]
        self.eps_r = eps_r if eps_r is not None else SMC_DEFAULTS["eps_r"]

        # ── Physical actuator limits for internal clamping ──
        # The controller clips its output to the physical thruster
        # envelope so that the commanded forces are feasible.  This
        # prevents the controller from implicitly operating as a
        # bang-bang controller that relies on downstream saturation.
        self.F_max = CONFIG["F_max"]
        self.F_min = CONFIG["F_min"]
        self.L = CONFIG["L_thruster"]

    def compute_control(
        self,
        u: float, r: float, y_e: float,
        psi: float, psi_d: float, u_d: float, gamma: float,
    ):
        """Compute conventional SMC control action.

        Parameters
        ----------
        u : float – surge speed [m/s]
        r : float – yaw rate [rad/s]
        y_e : float – cross-track error [m]
        psi : float – heading [rad]
        psi_d : float – desired heading [rad] (not used directly)
        u_d : float – desired surge speed [m/s]
        gamma : float – path tangent angle [rad]

        Returns
        -------
        (tau_u, tau_r) : tuple of floats
        """
        tilde_u = u - u_d
        chi_e = wrap_angle(psi - gamma)

        # ----- Surge sliding surface -----
        s_u = self.c_u * tilde_u
        sig_u = sigma(np.array([s_u]), np.array([self.eps_u]))[0]

        # Equivalent + reaching + switching control
        tau_u = -(1.0 / self.b_u) * (
            -self.a_u * tilde_u          # cancel drift on error dynamics
            + self.lambda_u * s_u        # linear reaching term
            + self.k_su * sig_u          # switching term
        )

        # ----- Yaw sliding surface -----
        s_r = self.c1 * chi_e + self.c2 * r + self.c3 * y_e
        sig_r = sigma(np.array([s_r]), np.array([self.eps_r]))[0]

        tau_r = -(1.0 / self.b_r) * (
            -self.a_r * r                # cancel drift
            + self.lambda_r * s_r        # linear reaching
            + self.k_sr * sig_r          # switching
        )

        # ----- Clamp to physical actuator limits -----
        # Map to individual thruster forces, clip, map back.
        # This is identical to what the simulation loop does, but by
        # doing it *inside* the controller the commanded (tau_u, tau_r)
        # that gets logged already respects actuator limits — the
        # controller "knows" it cannot exceed them.
        F_L = 0.5 * tau_u + tau_r / (2.0 * self.L)
        F_R = 0.5 * tau_u - tau_r / (2.0 * self.L)
        F_L = float(np.clip(F_L, self.F_min, self.F_max))
        F_R = float(np.clip(F_R, self.F_min, self.F_max))
        tau_u = F_L + F_R
        tau_r = self.L * (F_L - F_R)

        return float(tau_u), float(tau_r)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MBZIRC USV — Baseline SMC Controller")
    print("=" * 60)

    path_types = ["custom", "circular", "rectangular", "zigzag"]
    t_final_map = {
        "custom": 300.0, "circular": 350.0,
        "rectangular": 350.0, "zigzag": 300.0,
    }

    ctrl = SMCController()

    for ptype in path_types:
        print(f"\n{'─' * 50}")
        print(f"  Path: {ptype.upper()}")
        print(f"{'─' * 50}")

        data = run_simulation(
            controller=ctrl,
            config={"path_type": ptype, "t_final": t_final_map[ptype]},
            controller_name="SMC",
        )
        print_summary(data)
        plot_results(data)

    plt.show()
