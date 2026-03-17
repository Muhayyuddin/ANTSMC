#!/usr/bin/env python3
"""
Adaptive Nonlinear Terminal Sliding Mode Controller (ANTSMC) for USV.

Novel controller with three mechanisms over conventional SMC,
WITHOUT ESO to avoid transient-fighting on path switches.

1. Terminal sliding surface:  s_r = c1*chi_e + c2*r + c3*|y_e|^alpha*sign(y_e)
2. Power-rate reaching law:   lambda*|s|^p*sign(s) instead of lambda*s
3. Adaptive switching gain:   k(t) = k0 + k_adapt(t), where dk/dt = mu*|s|

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
# ANTSMC Defaults
# =============================================================================

ANTSMC_DEFAULTS = {
    # Terminal sliding surface (yaw channel)
    "c1": 1.0,          # weight on chi_e
    "c2": 0.9,          # weight on r  (yaw-rate damping)
    "c3": 0.3,          # weight on terminal y_e
    "alpha": 0.6,       # terminal power (0 < alpha < 1)

    # Surge surface
    "c_u": 1.0,

    # Power-rate reaching law
    "lambda_u": 0.5,    # surge reaching gain
    "lambda_r": 0.3,    # yaw reaching gain
    "p_r": 0.85,        # power-rate exponent (0 < p < 1)

    # Switching gains
    "k_su": 1.5,        # surge
    "k_sr0": 0.5,       # yaw base — adaptive gain adds up to k_adapt_max more

    # Adaptive switching gain (yaw)
    "mu_r": 3.0,         # adaptation rate
    "leak_r": 0.5,       # leakage rate (increased for 3-DOF robustness)
    "k_adapt_max": 0.8,  # max adaptive increment (total k_sr up to 1.3)

    # Boundary-layer widths
    "eps_u": 0.5,
    "eps_r": 0.6,        # wider boundary layer for smoother 3-DOF response

    # Cross-track integral
    "ki_ye": 0.02,       # reduced for 3-DOF sway robustness
    "int_ye_max": 3.0,   # tighter clamp for 3-DOF

    # Thrust allocation priority
    "surge_priority": 0.3,  # minimum fraction of thrust for surge
}


# =============================================================================
# ANTSMC Controller
# =============================================================================

class ANTSMCController:
    """Adaptive Nonlinear Terminal SMC for USV path following.

    Novel mechanisms over conventional SMC:
    1. Terminal (fractional-power) sliding surface for variable gain
    2. Power-rate reaching law for faster convergence near surface
    3. Adaptive switching gain for disturbance-proportional robustness

    No ESO - avoids transient artifacts during rapid path changes.
    """

    def __init__(self, **kwargs):
        self.p = {**ANTSMC_DEFAULTS, **kwargs}

        self.a_u = CONFIG["a_u"]
        self.a_r = CONFIG["a_r"]
        self.b_u = 1.0 / CONFIG["m_u"]
        self.b_r = 1.0 / CONFIG["I_r"]

        # Physical actuator limits
        self.F_max = CONFIG["F_max"]
        self.F_min = CONFIG["F_min"]
        self.L = CONFIG["L_thruster"]

        # Derived thrust limits
        self.tau_u_max = 2.0 * self.F_max  # both thrusters full forward
        self.tau_r_max = 2.0 * self.L * self.F_max  # differential max

        # Internal state
        self._int_ye = 0.0
        self._k_adapt_r = 0.0
        self._dt = CONFIG["dt"]
        self._prev_ye = 0.0

    @staticmethod
    def _sat(s, eps):
        if abs(s) <= eps:
            return s / eps
        return np.sign(s)

    @staticmethod
    def _terminal(x, alpha):
        """Modified terminal transformation with crossover.

        For |x| <= 1:  |x|^alpha * sign(x)  (boosted gain for small errors)
        For |x| >  1:  sign(x) * (1 + (|x| - 1))  = x  (linear for large)

        This gives the finite-time convergence benefit of the terminal
        surface for small errors without REDUCING gain at large errors
        during transients (which would slow down recovery on zigzag).
        """
        ax = abs(x)
        if ax <= 1.0:
            return np.sign(x) * (ax ** alpha)
        else:
            # Continuous at |x|=1: 1^alpha = 1
            # Slope at |x|=1: alpha * 1^(alpha-1) = alpha for terminal
            #                 but we use slope=1 for linear extension
            return np.sign(x) * (1.0 + (ax - 1.0))

    @staticmethod
    def _power_rate(s, p_exp):
        """Power-rate reaching: |s|^p * sign(s)"""
        return np.sign(s) * (abs(s) ** p_exp)

    def compute_control(self, u, r, y_e, psi, psi_d, u_d, gamma):
        p = self.p
        dt = self._dt

        # Cross-track error rate (sway-induced drift detection)
        ye_dot = (y_e - self._prev_ye) / dt
        self._prev_ye = y_e

        # --- Sway-resilience: detect large-error regime ---
        # When |y_e| is large, the vessel is likely affected by sway coupling
        # at a path corner. Reduce integral contribution and adaptive gain
        # to allow the controller to prioritize heading recovery.
        large_error = abs(y_e) > 5.0

        # Cross-track integral with conditional integration
        if not large_error:
            self._int_ye += y_e * dt
            self._int_ye = np.clip(self._int_ye, -p["int_ye_max"],
                                   p["int_ye_max"])
        else:
            # Exponential decay to prevent integral-induced lock-up
            decay = 0.9 if abs(y_e) < 15.0 else 0.8
            self._int_ye *= decay

        # === SURGE CHANNEL (standard SMC) ===
        tilde_u = u - u_d
        s_u = p["c_u"] * tilde_u
        sat_u = self._sat(s_u, p["eps_u"])

        tau_u = -(1.0 / self.b_u) * (
            -self.a_u * tilde_u
            + p["lambda_u"] * s_u
            + p["k_su"] * sat_u
        )

        # === YAW CHANNEL (terminal + power-rate + adaptive) ===
        chi_e = wrap_angle(psi - gamma)

        # Terminal sliding surface
        ye_term = self._terminal(y_e, p["alpha"])
        s_r = (p["c1"] * chi_e
               + p["c2"] * r
               + p["c3"] * ye_term
               + p["ki_ye"] * self._int_ye)

        # Adaptive gain with error-proportional leakage
        # Large errors → fast leakage to prevent runaway gain accumulation
        effective_leak = p["leak_r"] * (1.0 + 0.2 * min(abs(y_e), 20.0))
        self._k_adapt_r += (p["mu_r"] * abs(s_r)
                            - effective_leak * self._k_adapt_r) * dt
        self._k_adapt_r = float(np.clip(self._k_adapt_r,
                                        0.0, p["k_adapt_max"]))
        k_sr = p["k_sr0"] + self._k_adapt_r

        # Power-rate reaching + adaptive switching
        pr_term = self._power_rate(s_r, p["p_r"])
        sat_r = self._sat(s_r, p["eps_r"])

        tau_r = -(1.0 / self.b_r) * (
            -self.a_r * r
            + p["lambda_r"] * pr_term
            + k_sr * sat_r
        )

        # === THRUST ALLOCATION WITH SURGE PRIORITY ===
        F_L = 0.5 * tau_u + tau_r / (2.0 * self.L)
        F_R = 0.5 * tau_u - tau_r / (2.0 * self.L)
        F_L = float(np.clip(F_L, self.F_min, self.F_max))
        F_R = float(np.clip(F_R, self.F_min, self.F_max))

        # Surge priority enforcement: if speed is critically low and
        # yaw demand is consuming all thrust, scale yaw to preserve surge
        tau_u_actual = F_L + F_R
        if u < u_d * 0.4 and tau_u > 0 and tau_u_actual < p["surge_priority"] * self.tau_u_max:
            # Limit yaw torque to free thrust for surge
            tau_r_limit = (1.0 - p["surge_priority"]) * self.tau_r_max
            tau_r = float(np.clip(tau_r, -tau_r_limit, tau_r_limit))
            F_L = 0.5 * tau_u + tau_r / (2.0 * self.L)
            F_R = 0.5 * tau_u - tau_r / (2.0 * self.L)
            F_L = float(np.clip(F_L, self.F_min, self.F_max))
            F_R = float(np.clip(F_R, self.F_min, self.F_max))

        tau_u = F_L + F_R
        tau_r = self.L * (F_L - F_R)

        # Anti-windup: decay integral when near actuator limits
        if abs(F_L) >= self.F_max * 0.95 or abs(F_R) >= self.F_max * 0.95:
            self._int_ye *= 0.97

        return float(tau_u), float(tau_r)


# Alias for backward compatibility with run_all imports
NTSMCESOController = ANTSMCController


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MBZIRC USV - ANTSMC (Adaptive Nonlinear Terminal SMC)")
    print("  Novel controller - NO ESO")
    print("=" * 60)

    path_types = ["custom", "circular", "rectangular", "zigzag"]
    t_final_map = {
        "custom": 300.0, "circular": 350.0,
        "rectangular": 350.0, "zigzag": 300.0,
    }

    ctrl = ANTSMCController()

    for ptype in path_types:
        print(f"\n--- Path: {ptype.upper()} ---")
        data = run_simulation(
            controller=ctrl,
            config={"path_type": ptype, "t_final": t_final_map[ptype]},
            controller_name="ANTSMC",
        )
        print_summary(data)

    print("\nDone.")
