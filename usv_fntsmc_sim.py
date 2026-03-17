#!/usr/bin/env python3
"""
Fast Non-Singular Terminal Sliding Mode Controller (FNTSMC) for USV.

Implementation based on:
    Fan, Y., Liu, B., Wang, G., & Mu, D. (2021).
    "Adaptive Fast Non-Singular Terminal Sliding Mode Path Following
    Control for an Underactuated Unmanned Surface Vehicle with
    Uncertainties and Unknown Disturbances."
    Sensors, 21(22), 7454. https://doi.org/10.3390/s21227454

Key features from Fan et al. (2021):
1. Fast Non-Singular Terminal Sliding Mode (FNTSM) surface:
       s = e_dot + alpha*e + beta*zeta(e)
   with piecewise zeta(e) = sig^a(e) for |e|>=phi, linear for |e|<phi
   providing finite-time convergence without singularity.

2. Finite-Time Lumped Disturbance Observer (FTDO):
       M*nu_hat_dot = -lam1*sqrt(L)*sig^{1/2}(M*nu_tilde) + F_hat + tau
       F_hat_dot    = -lam2*L*sgn(M*nu_tilde)
   to estimate combined model uncertainty + external disturbance.

3. Two-layer adaptive switching gain:
       k_dot(t) = -rho(t)*sgn(delta(t)),  rho(t) = r0 + r(t)
       r_dot(t) = gamma*|delta(t)|
   for real-time gain adaptation without known disturbance bounds.

4. Auxiliary dynamic system for actuator saturation compensation.

Adaptation notes:
  Fan2021 uses their "Lanxin" USV (m11=215, m33=80 kg*m^2).
  Our MBZIRC USV has m_u=700 kg, I_r=2000 kg*m^2 (heavier vessel).
  Parameters are re-tuned accordingly while preserving the algorithm.

  The yaw sliding surface uses chi_e (heading error relative to
  path tangent) and couples y_e like all other controllers in this
  framework, to ensure a fair comparison under the same guidance law.

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
# FNTSMC Defaults
# =============================================================================
# Fan2021 original params (Section 5.2):
#   alpha_psi=4, beta_psi=0.7, eta_r=2, a=97/99, phi=0.01, L_obs=2000
#   K_er=-500, K_r=0.0001, alpha_u=3, beta_u=0.5, eta_u=0.1
#
# Re-tuned for MBZIRC USV (heavier vessel, different dynamics).
# =============================================================================

FNTSMC_DEFAULTS = {
    # -- FNTSM surface: yaw channel --
    # s_r = c1*chi_e + c2*r + c3*zeta(y_e) where zeta is the FNTSM
    # piecewise terminal function from Fan2021 Eq.(8)
    "c1":         1.0,     # weight on chi_e
    "c2":         0.8,     # weight on r (yaw-rate damping)
    "c3":         0.3,     # weight on terminal y_e

    # FNTSM terminal function parameters (Fan2021)
    "a_fntsm":    0.98,    # terminal exponent (Fan2021: 97/99)
    "phi":        0.01,    # piecewise switchover threshold

    # -- Surge surface: standard SMC --
    "c_u":        1.0,

    # -- Reaching law --
    "lambda_u":   0.5,     # surge reaching gain
    "lambda_r":   0.3,     # yaw reaching gain

    # -- Fixed switching gains --
    "eta_r":      0.8,     # fixed reaching gain, yaw (Fan2021: eta_r=2)
    "eta_u":      0.5,     # fixed reaching gain, surge

    # -- Two-layer adaptive gain: yaw (Fan2021 Eq.13-14) --
    #   k_dot(t) = -rho(t)*sgn(delta(t)),  delta = |s_r| - eps
    #   r_dot(t) = gamma*|delta|   when delta > 0
    #   rho(t)   = r0 + r(t)
    "gamma_r":    0.5,     # 2nd layer adaptation rate
    "r0_r":       0.3,     # base rho for yaw
    "eps_adapt":  0.01,    # epsilon in delta = |s| - eps
    "k_adapt_max": 0.4,   # max adaptive gain (safety clamp)

    # -- Finite-Time Disturbance Observer (FTDO, Fan2021 Eq.30-31) --
    #   M*nu_hat_dot = -lam1*sqrt(L)*sig^{1/2}(M*nu_tilde) + F_hat + tau
    #   F_hat_dot    = -lam2*L*sgn(M*nu_tilde)
    "ftdo_L":       180.0,   # observer gain (Fan2021: 2000, reduced)
    "ftdo_lambda1": 1.0,     # 1st observer coefficient
    "ftdo_lambda2": 0.8,     # 2nd observer coefficient

    # -- Auxiliary dynamic system (Fan2021 Eq.42) --
    "K_er":       200.0,   # auxiliary decay rate, yaw
    "K_eu":       200.0,   # auxiliary decay rate, surge
    "xi_r":       0.01,    # activation threshold, yaw
    "xi_u":       0.01,    # activation threshold, surge
    "K_comp_r":   0.01,    # saturation compensation gain, yaw
    "K_comp_u":   0.01,    # saturation compensation gain, surge

    # -- Boundary-layer widths (for smooth switching) --
    "eps_u":      0.5,
    "eps_r":      0.5,
}


# =============================================================================
# FNTSMC Controller
# =============================================================================

class FNTSMCController:
    """Fast Non-Singular Terminal SMC from Fan et al. (2021).

    Architecture:
      - FNTSM piecewise terminal function zeta(e) in the yaw surface
      - Finite-time lumped disturbance observer (FTDO)
      - Two-layer adaptive switching gain (unknown disturbance bounds)
      - Auxiliary dynamic system for actuator saturation compensation

    Yaw sliding surface (adapted for our framework):
        s_r = c1*chi_e + c2*r + c3*zeta(y_e)

    where zeta(e) is the Fan2021 FNTSM piecewise function:
        zeta(e) = |e|^a * sign(e)   if |e| >= phi
        zeta(e) = phi^(a-1) * e     if |e| < phi

    Control law (yaw):
        tau_r = -(1/b_r)*[-a_r*r + lambda_r*s_r + (eta_r + k_adapt)*sat(s_r)]
                - F_hat_r + K_comp*e_aux
    """

    def __init__(self, **kwargs):
        p = {**FNTSMC_DEFAULTS, **kwargs}
        self.p = p

        # Model parameters
        self.m_u = CONFIG["m_u"]       # surge mass (m11)
        self.m_r = CONFIG["I_r"]       # yaw inertia (m33)
        self.a_u = CONFIG["a_u"]
        self.a_r = CONFIG["a_r"]
        self.b_u = 1.0 / CONFIG["m_u"]
        self.b_r = 1.0 / CONFIG["I_r"]

        # Physical actuator limits
        self.F_max = CONFIG["F_max"]
        self.F_min = CONFIG["F_min"]
        self.L = CONFIG["L_thruster"]

        self._dt = CONFIG["dt"]

        # --- Internal state ---
        # Two-layer adaptive gain
        self._k_adapt = 0.3      # adaptive gain (yaw)
        self._r_adapt = 0.0      # 2nd-layer state

        # FTDO states (yaw)
        self._ftdo_r_nu_hat = 0.0    # estimated yaw rate
        self._ftdo_r_F_hat = 0.0     # estimated lumped disturbance

        # FTDO states (surge)
        self._ftdo_u_nu_hat = 0.0
        self._ftdo_u_F_hat = 0.0

        # Auxiliary dynamics
        self._e_aux_r = 0.0
        self._e_aux_u = 0.0

        self._first_call = True

    # -----------------------------------------------------------------
    #  Fan2021 FNTSM piecewise function zeta(e) -- Eq.(8)
    # -----------------------------------------------------------------

    def _zeta(self, e):
        """FNTSM piecewise terminal function from Fan2021 Eq.(8).

        For |e| >= phi:  zeta(e) = |e|^a * sign(e)   (terminal convergence)
        For |e| <  phi:  zeta(e) = phi^(a-1) * e      (linear, no singularity)

        Continuity at +/-phi:  phi^a = phi^(a-1) * phi  (check)
        """
        a = self.p["a_fntsm"]
        phi = self.p["phi"]
        if abs(e) >= phi:
            return np.sign(e) * (abs(e) ** a)
        else:
            return (phi ** (a - 1.0)) * e

    # -----------------------------------------------------------------
    #  Boundary-layer saturation
    # -----------------------------------------------------------------

    @staticmethod
    def _sat(s, eps):
        """Boundary-layer saturation sigma(s, eps)."""
        if abs(s) <= eps:
            return s / eps
        return np.sign(s)

    # -----------------------------------------------------------------
    #  Finite-Time Disturbance Observer (Fan2021 Eq.30-31)
    # -----------------------------------------------------------------

    def _update_ftdo(self, channel, nu_meas, tau_applied, dt):
        """Finite-time lumped disturbance observer.

        Estimates F in:  M*nu_dot = F + tau

        Observer equations (Fan2021 Eq.30):
            M*nu_hat_dot = -lam1*sqrt(L)*|M*nu_tilde|^{1/2}*sgn(M*nu_tilde)
                           + F_hat + tau
            F_hat_dot    = -lam2*L*sgn(M*nu_tilde)

        where nu_tilde = nu - nu_hat  (observer error).
        """
        lam1 = self.p["ftdo_lambda1"]
        lam2 = self.p["ftdo_lambda2"]
        L_obs = self.p["ftdo_L"]

        if channel == 'r':
            M = self.m_r
            nu_hat = self._ftdo_r_nu_hat
            F_hat = self._ftdo_r_F_hat
        else:
            M = self.m_u
            nu_hat = self._ftdo_u_nu_hat
            F_hat = self._ftdo_u_F_hat

        # Observer error
        nu_tilde = nu_meas - nu_hat
        M_nu_tilde = M * nu_tilde

        # sig^{1/2}(x) = |x|^{1/2} * sign(x)
        sig_half = np.sign(M_nu_tilde) * np.sqrt(abs(M_nu_tilde))

        # nu_hat_dot = (1/M) * [-lam1*sqrt(L)*sig^{1/2}(M*nu_tilde) + F_hat + tau]
        nu_hat_dot = (1.0 / M) * (
            lam1 * np.sqrt(L_obs) * sig_half + F_hat + tau_applied
        )

        # F_hat_dot = lam2*L*sgn(M*nu_tilde)   (smooth approx)
        eps_sign = 0.1
        if abs(M_nu_tilde) <= eps_sign:
            sgn_approx = M_nu_tilde / eps_sign
        else:
            sgn_approx = np.sign(M_nu_tilde)
        F_hat_dot = lam2 * L_obs * sgn_approx

        # Euler integration
        nu_hat_new = nu_hat + nu_hat_dot * dt
        F_hat_new = F_hat + F_hat_dot * dt

        # Safety clamp (prevent runaway)
        F_hat_new = float(np.clip(F_hat_new, -5000.0, 5000.0))

        if channel == 'r':
            self._ftdo_r_nu_hat = nu_hat_new
            self._ftdo_r_F_hat = F_hat_new
        else:
            self._ftdo_u_nu_hat = nu_hat_new
            self._ftdo_u_F_hat = F_hat_new

        return F_hat_new

    # -----------------------------------------------------------------
    #  Two-layer adaptive gain (Fan2021 Eq.13-14)
    # -----------------------------------------------------------------

    def _update_adaptive_gain(self, s, dt):
        """Two-layer adaptive switching gain (yaw channel).

        Layer 1:  k_dot(t) = -rho(t) * sgn(delta(t))
        Layer 2:  r_dot(t) = gamma * |delta(t)|   when delta > 0
        where:    rho(t) = r0 + r(t)
                  delta(t) = |s(t)| - eps
        """
        p = self.p
        gamma = p["gamma_r"]
        r0 = p["r0_r"]
        eps_a = p["eps_adapt"]
        k_max = p["k_adapt_max"]

        # delta = |s| - eps
        delta = abs(s) - eps_a

        # Layer 2 update
        if delta > 0:
            self._r_adapt += gamma * delta * dt
        self._r_adapt = float(np.clip(self._r_adapt, 0.0, 10.0))

        # rho(t) = r0 + r(t)
        rho = r0 + self._r_adapt

        # Layer 1 update: k_dot = -rho * sgn(delta)
        if abs(delta) <= 0.01:
            sgn_delta = delta / 0.01
        else:
            sgn_delta = np.sign(delta)
        self._k_adapt += (-rho * sgn_delta) * dt
        self._k_adapt = float(np.clip(self._k_adapt, 0.0, k_max))

        return self._k_adapt

    # -----------------------------------------------------------------
    #  Auxiliary dynamic system (Fan2021 Eq.42/57)
    # -----------------------------------------------------------------

    def _update_auxiliary(self, channel, s, delta_tau, dt):
        """Auxiliary dynamics for actuator saturation compensation.

        e_dot = -K_e*e - (s*dtau + 0.5*dtau^2)/(e + dtau)  when |e| >= xi
              = 0                                             otherwise

        where dtau = tau_actual - tau_commanded (saturation error).
        """
        if channel == 'r':
            e_aux = self._e_aux_r
            K_e = self.p["K_er"]
            xi = self.p["xi_r"]
        else:
            e_aux = self._e_aux_u
            K_e = self.p["K_eu"]
            xi = self.p["xi_u"]

        if abs(e_aux) >= xi:
            denom = e_aux + delta_tau
            if abs(denom) < 1e-10:
                denom = 1e-10 * (1.0 if denom >= 0 else -1.0)
            e_dot = (-K_e * e_aux
                     - (s * delta_tau + 0.5 * delta_tau**2) / denom)
        else:
            e_dot = 0.0

        e_aux_new = float(np.clip(e_aux + e_dot * dt, -10.0, 10.0))

        if channel == 'r':
            self._e_aux_r = e_aux_new
        else:
            self._e_aux_u = e_aux_new
        return e_aux_new

    # -----------------------------------------------------------------
    #  Main control law
    # -----------------------------------------------------------------

    def compute_control(self, u, r, y_e, psi, psi_d, u_d, gamma):
        """Compute FNTSMC control action (Fan2021 Eq.43/58).

        Parameters
        ----------
        u     : surge speed [m/s]
        r     : yaw rate [rad/s]
        y_e   : cross-track error [m]
        psi   : heading [rad]
        psi_d : desired heading [rad] (from LOS guidance)
        u_d   : desired surge speed [m/s]
        gamma : path tangent angle [rad]

        Returns
        -------
        (tau_u, tau_r) : control forces
        """
        p = self.p
        dt = self._dt

        if self._first_call:
            self._ftdo_r_nu_hat = r
            self._ftdo_u_nu_hat = u
            self._first_call = False

        # ==========================================================
        #  YAW CHANNEL -- FNTSM surface + FTDO + adaptive gain
        # ==========================================================

        chi_e = wrap_angle(psi - gamma)

        # -- FNTSM sliding surface --
        # s_r = c1*chi_e + c2*r + c3*zeta(y_e)
        #
        # The FNTSM piecewise terminal function zeta(y_e) replaces
        # the linear y_e term used in conventional SMC, providing
        # faster finite-time convergence for small errors and
        # non-singular behaviour (Fan2021 Lemma 2).
        zeta_ye = self._zeta(y_e)
        s_r = p["c1"] * chi_e + p["c2"] * r + p["c3"] * zeta_ye

        # -- FTDO: estimated lumped disturbance (from previous step) --
        F_hat_r = self._ftdo_r_F_hat

        # -- Two-layer adaptive gain update --
        k_adapt = self._update_adaptive_gain(s_r, dt)
        k_total = p["eta_r"] + k_adapt

        # -- Saturation function --
        sat_r = self._sat(s_r, p["eps_r"])

        # -- Control law (Fan2021 Eq.43 adapted) --
        # tau_r = -(1/b_r)*[-a_r*r + lambda_r*s_r + k_total*sat(s_r)]
        #         - F_hat_r + K_comp*e_aux
        #
        # F_hat_r is the estimated lumped disturbance in force units.
        # We feed it forward for disturbance compensation.
        tau_r0 = -(1.0 / self.b_r) * (
            -self.a_r * r                  # cancel known damping
            + p["lambda_r"] * s_r          # reaching
            + k_total * sat_r              # adaptive switching
        ) - F_hat_r + p["K_comp_r"] * self._e_aux_r

        # ==========================================================
        #  SURGE CHANNEL -- standard SMC + FTDO
        # ==========================================================

        tilde_u = u - u_d
        s_u = p["c_u"] * tilde_u
        sat_u = self._sat(s_u, p["eps_u"])

        F_hat_u = self._ftdo_u_F_hat

        tau_u0 = -(1.0 / self.b_u) * (
            -self.a_u * tilde_u
            + p["lambda_u"] * s_u
            + p["eta_u"] * sat_u
        ) - F_hat_u + p["K_comp_u"] * self._e_aux_u

        # ==========================================================
        #  THRUST ALLOCATION + SATURATION
        # ==========================================================

        F_L = 0.5 * tau_u0 + tau_r0 / (2.0 * self.L)
        F_R = 0.5 * tau_u0 - tau_r0 / (2.0 * self.L)
        F_L_sat = float(np.clip(F_L, self.F_min, self.F_max))
        F_R_sat = float(np.clip(F_R, self.F_min, self.F_max))

        tau_u = F_L_sat + F_R_sat
        tau_r = self.L * (F_L_sat - F_R_sat)

        # ==========================================================
        #  AUXILIARY DYNAMICS UPDATE (saturation compensation)
        # ==========================================================

        delta_tau_r = tau_r - tau_r0
        delta_tau_u = tau_u - tau_u0
        self._update_auxiliary('r', s_r, delta_tau_r, dt)
        self._update_auxiliary('u', s_u, delta_tau_u, dt)

        # ==========================================================
        #  FTDO UPDATE (for next step)
        # ==========================================================

        self._update_ftdo('r', r, tau_r, dt)
        self._update_ftdo('u', u, tau_u, dt)

        return float(tau_u), float(tau_r)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MBZIRC USV -- FNTSMC (Fan et al., 2021)")
    print("  Adaptive Fast Non-Singular Terminal Sliding Mode Control")
    print("=" * 60)

    path_types = ["custom", "circular", "rectangular", "zigzag"]
    t_final_map = {
        "custom": 300.0, "circular": 350.0,
        "rectangular": 350.0, "zigzag": 300.0,
    }

    for ptype in path_types:
        print("\n--- Path: {} ---".format(ptype.upper()))
        ctrl = FNTSMCController()
        data = run_simulation(
            controller=ctrl,
            config={
                "path_type": ptype,
                "t_final": t_final_map[ptype],
                "disturbance_scale": 0.35,
            },
            controller_name="FNTSMC",
        )
        print_summary(data)

    print("\nDone.")
