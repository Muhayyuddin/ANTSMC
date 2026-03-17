#!/usr/bin/env python3
"""
Baseline LQR Controller for MBZIRC-style USV.

Pure LQR state-feedback on the error state x_ctrl = [tilde_u, r, y_e, chi_e].

Control law:  u_ctrl = -K @ x_ctrl

where K = R^{-1} B^T P is the optimal LQR gain from the continuous-time
algebraic Riccati equation.

No sliding-mode augmentation — this serves as the "LQR-only" baseline
for comparison with the hybrid LQR–SMC and the pure SMC controllers.

Usage (standalone):
    python usv_lqr_sim.py

Dependencies: numpy, scipy, matplotlib, usv_common
"""

import numpy as np
import matplotlib.pyplot as plt

from usv_common import (
    CONFIG, build_AB, compute_lqr_gains, wrap_angle,
    run_simulation, plot_results, print_summary,
    get_waypoints_for_path_type,
)


# =============================================================================
# LQR Controller
# =============================================================================

class LQRController:
    """LQR state-feedback controller with integral augmentation.

    Error state:
        x_ctrl = [tilde_u, r, y_e, chi_e]^T
        tilde_u = u - u_d
        chi_e   = wrap(psi - gamma)

    Control:
        [tau_u, tau_r]^T = -K @ x_ctrl - K_i * ∫y_e dt

    The gain K is computed from the continuous-time algebraic Riccati
    equation for the linearised surge–yaw error dynamics.

    A small integral term on cross-track error is added to reject
    constant sway disturbance bias that pure state feedback cannot
    eliminate (Anderson & Moore, 1990, §5.5).
    """

    def __init__(
        self,
        a_u: float = None, a_r: float = None,
        b_u: float = None, b_r: float = None,
        Q: np.ndarray = None, R: np.ndarray = None,
        u_d_nom: float = None,
    ):
        self.a_u = a_u if a_u is not None else CONFIG["a_u"]
        self.a_r = a_r if a_r is not None else CONFIG["a_r"]
        self.b_u = b_u if b_u is not None else 1.0 / CONFIG["m_u"]
        self.b_r = b_r if b_r is not None else 1.0 / CONFIG["I_r"]
        self.Q = Q if Q is not None else CONFIG["Q"].copy()
        self.R = R if R is not None else CONFIG["R"].copy()
        self.u_d_nom = u_d_nom if u_d_nom is not None else CONFIG["u_d_nom"]

        # Build linearised model and solve Riccati
        self.A, self.B = build_AB(self.a_u, self.a_r, self.b_u, self.b_r, self.u_d_nom)
        self.P, self.K, self.G = compute_lqr_gains(self.A, self.B, self.Q, self.R)

        # Integral gain for cross-track bias rejection
        # Ki_ye acts on ∫y_e dt and feeds into yaw channel only
        # Gain chosen to be small enough not to cause oscillation
        # but sufficient to reject constant sway disturbance
        self._Ki_ye = 15.0        # [N·m / (m·s)]
        self._int_ye = 0.0
        self._int_ye_max = 5.0    # anti-windup clamp [m·s]
        self._int_gate = 8.0      # only integrate when |y_e| < gate [m]
        self._dt = CONFIG["dt"]

    def compute_control(
        self,
        u: float, r: float, y_e: float,
        psi: float, psi_d: float, u_d: float, gamma: float,
    ):
        tilde_u = u - u_d
        chi_e = wrap_angle(psi - gamma)
        x_ctrl = np.array([tilde_u, r, y_e, chi_e])
        assert x_ctrl.shape == (4,)

        u_ctrl = -self.K @ x_ctrl

        # Integral augmentation on cross-track error (yaw channel only)
        # Conditional integration to prevent windup at corners
        if abs(y_e) < self._int_gate:
            self._int_ye += y_e * self._dt
        else:
            self._int_ye *= 0.95  # decay when far off-track
        self._int_ye = np.clip(self._int_ye, -self._int_ye_max, self._int_ye_max)

        tau_u = float(u_ctrl[0])
        tau_r = float(u_ctrl[1]) - self._Ki_ye * self._int_ye

        return tau_u, tau_r


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MBZIRC USV — Baseline LQR Controller")
    print("=" * 60)

    path_types = ["custom", "circular", "rectangular", "zigzag"]
    t_final_map = {
        "custom": 300.0, "circular": 350.0,
        "rectangular": 350.0, "zigzag": 300.0,
    }

    ctrl = LQRController()

    for ptype in path_types:
        print(f"\n{'─' * 50}")
        print(f"  Path: {ptype.upper()}")
        print(f"{'─' * 50}")

        data = run_simulation(
            controller=ctrl,
            config={"path_type": ptype, "t_final": t_final_map[ptype]},
            controller_name="LQR",
        )
        print_summary(data)
        plot_results(data)

    plt.show()
