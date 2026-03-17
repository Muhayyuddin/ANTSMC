#!/usr/bin/env python3
"""
Adaptive Sliding Mode Controller (ASMC) for USV — ablation baseline.

This controller uses SMC's conventional **linear** sliding surface
(no terminal fractional-power term) combined with ANTSMC's **adaptive
switching gain** mechanism.  It serves as an ablation baseline to
isolate the contribution of the terminal sliding surface from the
adaptive gain.

    SMC:    linear surface + fixed gain        → baseline
    ASMC:   linear surface + adaptive gain     → this file  (ablation)
    ANTSMC: terminal surface + adaptive gain   → proposed

Sliding surfaces:
    s_u = c_u * tilde_u                          (surge — identical to SMC)
    s_r = c1 * chi_e + c2 * r + c3 * y_e         (yaw — LINEAR, identical to SMC)

Yaw control law (adaptive gain, no terminal surface, no power-rate reaching):
    tau_r = -(1/b_r) [ -a_r*r + lambda_r*s_r + k_sr(t)*sat(s_r) ]

    where k_sr(t) = k_sr0 + k_adapt(t),
          dk_adapt/dt = mu*|s_r| - leak(y_e)*k_adapt   (same as ANTSMC)

The reaching law uses the standard linear term  λ·s  (NOT the power-rate
|s|^p·sign(s) from ANTSMC), so the ONLY difference between ASMC and SMC
is the adaptive switching gain.

Usage (standalone):
    python usv_asmc_sim.py

Dependencies: numpy, matplotlib, usv_common
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_common import (
    CONFIG, wrap_angle, sigma,
    run_simulation, plot_results, print_summary,
)


# =============================================================================
# ASMC Defaults — same gains as ANTSMC where applicable, but LINEAR surface
# =============================================================================

ASMC_DEFAULTS = {
    # Linear sliding surface (yaw channel) — SAME as SMC
    "c1": 1.0,          # weight on chi_e  (same as SMC)
    "c2": 0.8,          # weight on r      (same as SMC)
    "c3": 0.3,          # weight on y_e    (LINEAR — same as SMC)

    # Surge surface
    "c_u": 1.0,

    # Reaching gains — LINEAR reaching (same as SMC)
    "lambda_u": 0.5,    # surge reaching gain  (same as SMC)
    "lambda_r": 0.3,    # yaw reaching gain    (same as SMC)

    # Switching gains
    "k_su": 1.5,        # surge (fixed, same as SMC)
    "k_sr0": 0.8,       # yaw base (same as SMC's fixed gain)

    # Adaptive switching gain (yaw) — same mechanism as ANTSMC
    "mu_r": 3.0,         # adaptation rate  (same as ANTSMC)
    "leak_r": 0.5,       # base leakage rate (same as ANTSMC)
    "k_adapt_max": 0.2,  # max adaptive increment (total k_sr up to 1.0)
                          # lower than ANTSMC (0.8) because without the terminal
                          # surface, large adaptive gains cause overshoot at corners

    # Boundary-layer widths — same as SMC
    "eps_u": 0.5,
    "eps_r": 0.5,
}


# =============================================================================
# ASMC Controller
# =============================================================================

class ASMCController:
    """Adaptive SMC with LINEAR sliding surface (ablation baseline).

    Structurally identical to SMCController except that k_sr is adaptive:
        k_sr(t) = k_sr0 + k_adapt(t)
    where dk_adapt/dt = mu*|s_r| - leak(y_e)*k_adapt  (same as ANTSMC).

    Everything else (sliding surface, reaching law, saturation, thrust
    allocation) is exactly the same as SMC.
    """

    def __init__(self, **kwargs):
        p = {**ASMC_DEFAULTS, **kwargs}
        self.c1 = p["c1"]
        self.c2 = p["c2"]
        self.c3 = p["c3"]
        self.c_u = p["c_u"]
        self.lambda_u = p["lambda_u"]
        self.lambda_r = p["lambda_r"]
        self.k_su = p["k_su"]
        self.k_sr0 = p["k_sr0"]
        self.eps_u = p["eps_u"]
        self.eps_r = p["eps_r"]

        # Adaptive gain parameters (from ANTSMC)
        self.mu_r = p["mu_r"]
        self.leak_r = p["leak_r"]
        self.k_adapt_max = p["k_adapt_max"]

        self.a_u = CONFIG["a_u"]
        self.a_r = CONFIG["a_r"]
        self.b_u = 1.0 / CONFIG["m_u"]
        self.b_r = 1.0 / CONFIG["I_r"]

        # Physical actuator limits
        self.F_max = CONFIG["F_max"]
        self.F_min = CONFIG["F_min"]
        self.L = CONFIG["L_thruster"]

        # Internal state
        self._k_adapt_r = 0.0
        self._dt = CONFIG["dt"]

    def compute_control(self, u, r, y_e, psi, psi_d, u_d, gamma):
        dt = self._dt
        tilde_u = u - u_d
        chi_e = wrap_angle(psi - gamma)

        # === SURGE (identical to SMC) ===
        s_u = self.c_u * tilde_u
        sig_u = sigma(np.array([s_u]), np.array([self.eps_u]))[0]

        tau_u = -(1.0 / self.b_u) * (
            -self.a_u * tilde_u
            + self.lambda_u * s_u
            + self.k_su * sig_u
        )

        # === YAW (SMC surface + adaptive gain) ===
        s_r = self.c1 * chi_e + self.c2 * r + self.c3 * y_e
        sig_r = sigma(np.array([s_r]), np.array([self.eps_r]))[0]

        # Adaptive gain update — same law as ANTSMC
        effective_leak = self.leak_r * (1.0 + 0.2 * min(abs(y_e), 20.0))
        self._k_adapt_r += (self.mu_r * abs(s_r)
                            - effective_leak * self._k_adapt_r) * dt
        self._k_adapt_r = float(np.clip(self._k_adapt_r,
                                        0.0, self.k_adapt_max))
        k_sr = self.k_sr0 + self._k_adapt_r

        tau_r = -(1.0 / self.b_r) * (
            -self.a_r * r
            + self.lambda_r * s_r       # linear reaching (same as SMC)
            + k_sr * sig_r              # adaptive switching
        )

        # === THRUST ALLOCATION (identical to SMC) ===
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
    print("  MBZIRC USV - ASMC (Adaptive SMC — Ablation Baseline)")
    print("  Linear surface + adaptive gain (no terminal surface)")
    print("=" * 60)

    path_types = ["custom", "circular", "rectangular", "zigzag"]
    t_final_map = {
        "custom": 300.0, "circular": 350.0,
        "rectangular": 350.0, "zigzag": 300.0,
    }

    ctrl = ASMCController()

    for ptype in path_types:
        print(f"\n--- Path: {ptype} ---")
        data = run_simulation(
            controller=ctrl,
            config={
                "path_type": ptype,
                "t_final": t_final_map[ptype],
                "disturbance_scale": 0.35,
            },
            controller_name=f"ASMC",
        )
        print_summary(data)

    print("\nDone.")
