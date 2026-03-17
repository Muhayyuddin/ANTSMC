#!/usr/bin/env python3
"""
JONSWAP Wave Spectrum Validation Study
=======================================

Runs all 6 controllers under stochastic JONSWAP wave disturbances
(replacing deterministic sinusoidal wave components) to validate
that the controller ranking is robust to stochastic excitation.

Three scenarios are tested:
  1. SS1-equivalent: Hs=0.1m, Tp=4.0s (calm)
  2. SS2-equivalent: Hs=0.5m, Tp=5.0s (smooth)
  3. SS3-equivalent: Hs=1.0m, Tp=6.0s (slight)

Note: SS4 is excluded — small USVs are not recommended to operate
above Sea State 3 per IMO guidelines.

Generates:
  - Comparison table (JONSWAP vs deterministic) for all 3 sea states
  - Time-series plot of JONSWAP wave elevation
  - Bar chart comparing RMS y_e under both disturbance models
  - JONSWAP spectrum plot

Reference: DNV-RP-C205 (2019); Fossen (2011), Ch. 8.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from usv_common import (
    CONFIG, run_simulation, USVDynamics,
    JONSWAPDisturbance, set_jonswap_params,
    disturbance_u, disturbance_v, disturbance_r,
    disturbance_u_jonswap, disturbance_v_jonswap, disturbance_r_jonswap,
)
from usv_run_all import set_disturbance_scale, restore_disturbances

# Import all controllers
from usv_lqr_sim import LQRController
from usv_smc_sim import SMCController
from usv_asmc_sim import ASMCController
from usv_adrc_sim import ADRCController
from usv_fntsmc_sim import FNTSMCController
from usv_ntsmc_eso_sim import ANTSMCController

CONTROLLERS = [
    ("LQR",     LQRController),
    ("SMC",     SMCController),
    ("ASMC",    ASMCController),
    ("ADRC",    ADRCController),
    ("FNTSMC",  FNTSMCController),
    ("ANTSMC",  ANTSMCController),
]

PATHS = ["custom", "circular", "rectangular", "zigzag"]
T_FINAL_MAP = {"custom": 300, "circular": 350, "rectangular": 350, "zigzag": 300}

# JONSWAP sea-state mapping (WMO → Hs, Tp)
# SS1: calm, SS2: smooth, SS3: slight — covers the full operational envelope
JONSWAP_SCENARIOS = {
    "SS1": {"Hs": 0.1,  "Tp": 4.0, "det_scale": 0.10, "label": "SS1 (Hs=0.1m, Tp=4.0s)"},
    "SS2": {"Hs": 0.5,  "Tp": 5.0, "det_scale": 0.35, "label": "SS2 (Hs=0.5m, Tp=5.0s)"},
    "SS3": {"Hs": 1.0,  "Tp": 6.0, "det_scale": 0.60, "label": "SS3 (Hs=1.0m, Tp=6.0s)"},
}


def run_deterministic(ss_name, path_type):
    """Run all controllers with deterministic disturbance at given SS."""
    # Ensure JONSWAP functions are NOT in use (clean module state)
    import usv_common as _uc
    _uc.disturbance_u = disturbance_u
    _uc.disturbance_v = disturbance_v
    _uc.disturbance_r = disturbance_r

    scale = JONSWAP_SCENARIOS[ss_name]["det_scale"]
    set_disturbance_scale(scale)
    results = {}
    for name, ctrl_cls in CONTROLLERS:
        ctrl = ctrl_cls()
        cfg = {"path_type": path_type, "t_final": T_FINAL_MAP[path_type]}
        data = run_simulation(ctrl, cfg, name)
        ye = np.array(data["y_e"])
        t = np.array(data["t"])
        mask_ss = t > 30
        rms = np.sqrt(np.mean(ye[mask_ss] ** 2))
        results[name] = rms
    restore_disturbances()
    return results


def run_jonswap(ss_name, path_type):
    """Run all controllers with JONSWAP stochastic disturbance."""
    params = JONSWAP_SCENARIOS[ss_name]
    # Set wind/current scale (same as deterministic)
    set_disturbance_scale(params["det_scale"])
    # Configure JONSWAP
    set_jonswap_params(Hs=params["Hs"], Tp=params["Tp"], seed=42)

    results = {}
    for name, ctrl_cls in CONTROLLERS:
        ctrl = ctrl_cls()
        # Create dynamics with JONSWAP disturbance functions
        dynamics = USVDynamics(
            dist_u_func=disturbance_u_jonswap,
            dist_v_func=disturbance_v_jonswap,
            dist_r_func=disturbance_r_jonswap,
        )
        cfg = {"path_type": path_type, "t_final": T_FINAL_MAP[path_type]}
        data = run_simulation(ctrl, cfg, name, dynamics=dynamics)
        ye = np.array(data["y_e"])
        t = np.array(data["t"])
        mask_ss = t > 30
        rms = np.sqrt(np.mean(ye[mask_ss] ** 2))
        results[name] = rms
    restore_disturbances()
    return results


def plot_jonswap_spectrum(save_path="plots/jonswap_spectrum.png"):
    """Plot deterministic vs JONSWAP spectral density side by side for SS3."""
    from usv_run_all import set_disturbance_scale, restore_disturbances

    params = JONSWAP_SCENARIOS["SS3"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # --- (a) Deterministic sinusoidal wave spectrum ---
    ax = axes[0]
    # Set SS3 scale to read the scaled wave amplitudes
    set_disturbance_scale(params["det_scale"])
    from usv_common import CONFIG as _cfg
    # The deterministic wave is a single-frequency sinusoid at d_wave_freq
    omega_det = _cfg["d_wave_freq"]  # 0.5 rad/s (scaled from CONFIG)
    # Wave amplitudes (surge acceleration)
    A_u = _cfg["d_u_wave_amp"]  # scaled amplitude [m/s²]
    # Convert acceleration amplitude to equivalent "wave elevation" amplitude
    # via the same force_scale_u used in JONSWAP: a = force_scale_u * η → η = a / force_scale_u
    force_scale_u = 0.065  # same as JONSWAPDisturbance default
    A_eta = A_u / force_scale_u  # equivalent wave elevation amplitude [m]
    # The spectral density of A*sin(ω₀t) is S(ω) = (A²/2) δ(ω - ω₀)
    # We represent this as a narrow Gaussian spike for visualisation
    omegas_det = np.linspace(0.01, 3.5, 1000)
    spike_width = 0.015  # narrow Gaussian width [rad/s]
    S_det = (A_eta**2 / 2) * (1.0 / (spike_width * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((omegas_det - omega_det) / spike_width)**2)

    ax.fill_between(omegas_det, S_det, alpha=0.4, color="steelblue")
    ax.plot(omegas_det, S_det, "b-", linewidth=1.5, label="Sinusoidal")
    ax.axvline(omega_det, color="r", linestyle="--", alpha=0.7,
               label=f"$\\omega_0$ = {omega_det:.2f} rad/s")
    ax.set_xlabel("$\\omega$ [rad/s]", fontsize=11)
    ax.set_ylabel("$S(\\omega)$ [m²·s/rad]", fontsize=11)
    ax.set_title("(a) Deterministic Sinusoidal", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3.5])
    restore_disturbances()

    # --- (b) JONSWAP spectrum ---
    ax = axes[1]
    j = JONSWAPDisturbance(Hs=params["Hs"], Tp=params["Tp"])
    omega_p = j.omega_p
    omegas_jon = np.linspace(0.1, 3.0 * omega_p, 500)

    alpha_pm = 0.0081
    g = 9.81
    gamma = 3.3
    S_jon = np.zeros_like(omegas_jon)
    for i, w in enumerate(omegas_jon):
        sigma = 0.07 if w <= omega_p else 0.09
        S_pm = (alpha_pm * g ** 2 / w ** 5) * np.exp(-1.25 * (omega_p / w) ** 4)
        r_exp = np.exp(-0.5 * ((w - omega_p) / (sigma * omega_p)) ** 2)
        S_jon[i] = S_pm * gamma ** r_exp

    # Normalise to Hs
    dw = omegas_jon[1] - omegas_jon[0]
    m0 = np.sum(S_jon) * dw
    S_jon *= (params["Hs"] / (4.0 * np.sqrt(m0))) ** 2

    ax.fill_between(omegas_jon, S_jon, alpha=0.4, color="coral")
    ax.plot(omegas_jon, S_jon, color="orangered", linewidth=1.5, label="JONSWAP")
    ax.axvline(omega_p, color="r", linestyle="--", alpha=0.7,
               label=f"$\\omega_p$ = {omega_p:.2f} rad/s")
    ax.set_xlabel("$\\omega$ [rad/s]", fontsize=11)
    ax.set_title(f"(b) JONSWAP ($H_s$={params['Hs']} m, $T_p$={params['Tp']} s, "
                 f"$\\gamma$={gamma})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper center", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3.5])

    fig.suptitle("Wave Spectra Comparison — Sea State 3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_wave_timeseries(save_path="plots/jonswap_wave_timeseries.png"):
    """Plot deterministic vs JONSWAP wave elevation time series for SS3."""
    from usv_run_all import set_disturbance_scale, restore_disturbances

    params = JONSWAP_SCENARIOS["SS3"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    t = np.linspace(0, 300, 6000)

    # --- (a) Deterministic sinusoidal time series ---
    ax = axes[0]
    set_disturbance_scale(params["det_scale"])
    from usv_common import CONFIG as _cfg
    omega_det = _cfg["d_wave_freq"]
    A_u = _cfg["d_u_wave_amp"]
    force_scale_u = 0.065
    A_eta = A_u / force_scale_u  # equivalent wave elevation [m]
    eta_det = A_eta * np.sin(omega_det * t)
    restore_disturbances()

    ax.plot(t, eta_det, "b-", linewidth=2.0, alpha=0.9, label="Sinusoidal")
    ax.axhline( A_eta, color="r", linewidth=1.5, linestyle="--", alpha=0.5,
               label=f"Amplitude = {A_eta:.3f} m")
    ax.axhline(-A_eta, color="r", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("$\\eta(t)$ [m]", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title("(a) Deterministic Sinusoidal Wave",
                 fontsize=20, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- (b) JONSWAP stochastic time series ---
    ax = axes[1]
    
    omega_p = 2 * np.pi / params["Tp"]
    omegas = np.linspace(0.1, 3.0 * omega_p, 500)
    alpha_pm = 0.0081
    g = 9.81
    gamma = 3.3
    S = np.zeros_like(omegas)
    for i, w in enumerate(omegas):
        sigma = 0.07 if w <= omega_p else 0.09
        S_pm = (alpha_pm * g ** 2 / w ** 5) * np.exp(-1.25 * (omega_p / w) ** 4)
        r_exp = np.exp(-0.5 * ((w - omega_p) / (sigma * omega_p)) ** 2)
        S[i] = S_pm * gamma ** r_exp
    
    rng = np.random.default_rng(seed=42)
    # Generate elevation logic inline from JONSWAPDisturbance
    phases = rng.uniform(0, 2 * np.pi, len(omegas))
    dw = omegas[1] - omegas[0]
    A_i = np.sqrt(2 * S * dw)
    eta_stoch = np.zeros_like(t)
    for i in range(len(omegas)):
        eta_stoch += A_i[i] * np.cos(omegas[i] * t + phases[i])
    
    ax.plot(t, eta_stoch, "b-", linewidth=2.0, alpha=0.9, label="JONSWAP")
    ax.axhline( params["Hs"]/2, color="g", linewidth=1.5, linestyle="--", alpha=0.5,
               label=f"Ideal Amp = {params['Hs']/2:.3f} m")
    ax.axhline(-params["Hs"]/2, color="g", linewidth=1.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("$\\eta(t)$ [m]", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_title(f"(b) JONSWAP Wave (Hs={params['Hs']}m, Tp={params['Tp']}s)",
                 fontsize=20, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    
    eps_path = save_path.replace("plots/", "plots-eps/").replace(".png", ".eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")

    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Saved: {eps_path}")


def plot_comparison_bars(results_det, results_jon, save_path="plots/jonswap_comparison.png"):
    """Bar chart comparing deterministic vs JONSWAP RMS y_e."""
    ctrl_names = [c[0] for c in CONTROLLERS]
    ss_names = list(JONSWAP_SCENARIOS.keys())

    fig, axes = plt.subplots(1, len(ss_names), figsize=(14, 5))
    if len(ss_names) == 1:
        axes = [axes]

    x = np.arange(len(ctrl_names))
    width = 0.35

    for ax, ss_name in zip(axes, ss_names):
        det_vals = [results_det[ss_name].get(c, 0) for c in ctrl_names]
        jon_vals = [results_jon[ss_name].get(c, 0) for c in ctrl_names]
        
        bars1 = ax.bar(x - width / 2, det_vals, width, label="Deterministic",
                       color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + width / 2, jon_vals, width, label="JONSWAP",
                       color="coral", alpha=0.8)

        ax.set_xlabel("Controller")
        ax.set_ylabel("RMS $y_e$ (ss) [m]")
        ax.set_title(f"{JONSWAP_SCENARIOS[ss_name]['label']}")
        ax.set_xticks(x)
        ax.set_xticklabels(ctrl_names, fontsize=9)
        ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis="y")

        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Controller Performance: Deterministic vs JONSWAP Wave Disturbance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    eps_path = save_path.replace("plots/", "plots-eps/").replace(".png", ".eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("JONSWAP WAVE SPECTRUM VALIDATION STUDY")
    print("=" * 70)

    # Check if run_simulation accepts dynamics parameter
    # We need to check the signature
    import inspect
    sig = inspect.signature(run_simulation)
    has_dynamics = "dynamics" in sig.parameters

    if not has_dynamics:
        print("\n[INFO] run_simulation does not accept 'dynamics' parameter.")
        print("       Will patch USVDynamics disturbance functions directly.\n")

    # Generate spectrum and wave plots
    print("\n--- Generating JONSWAP spectrum plot ---")
    plot_jonswap_spectrum()

    print("\n--- Generating wave time-series plot ---")
    plot_wave_timeseries()

    # Run simulations
    results_det = {}  # {ss_name: {ctrl_name: avg_rms}}
    results_jon = {}

    for ss_name, params in JONSWAP_SCENARIOS.items():
        print(f"\n{'=' * 50}")
        print(f"  {params['label']}")
        print(f"{'=' * 50}")

        det_avgs = {c: 0.0 for c in [n for n, _ in CONTROLLERS]}
        jon_avgs = {c: 0.0 for c in [n for n, _ in CONTROLLERS]}

        for path in PATHS:
            print(f"\n  Path: {path}")

            # --- Deterministic ---
            print("    [Deterministic]", end=" ")
            det = run_deterministic(ss_name, path)
            for cn, val in det.items():
                det_avgs[cn] += val / len(PATHS)
                print(f"{cn}={val:.3f}", end="  ")
            print()

            # --- JONSWAP ---
            print("    [JONSWAP]      ", end=" ")
            set_disturbance_scale(params["det_scale"])
            set_jonswap_params(Hs=params["Hs"], Tp=params["Tp"], seed=42)

            for name, ctrl_cls in CONTROLLERS:
                ctrl = ctrl_cls()
                # Temporarily swap disturbance functions in CONFIG-reading dynamics
                from usv_common import USVDynamics as _Dyn
                dyn = _Dyn(
                    dist_u_func=disturbance_u_jonswap,
                    dist_v_func=disturbance_v_jonswap,
                    dist_r_func=disturbance_r_jonswap,
                )
                cfg = {"path_type": path, "t_final": T_FINAL_MAP[path]}
                if has_dynamics:
                    data = run_simulation(ctrl, cfg, name, dynamics=dyn)
                else:
                    # Monkey-patch: replace the dynamics in run_simulation
                    # by temporarily swapping disturbance functions
                    import usv_common as _uc
                    orig_du = _uc.disturbance_u
                    orig_dv = _uc.disturbance_v
                    orig_dr = _uc.disturbance_r
                    _uc.disturbance_u = disturbance_u_jonswap
                    _uc.disturbance_v = disturbance_v_jonswap
                    _uc.disturbance_r = disturbance_r_jonswap
                    try:
                        data = run_simulation(ctrl, cfg, name)
                    finally:
                        _uc.disturbance_u = orig_du
                        _uc.disturbance_v = orig_dv
                        _uc.disturbance_r = orig_dr

                ye = np.array(data["y_e"])
                t_arr = np.array(data["t"])
                mask_ss = t_arr > 30
                rms = np.sqrt(np.mean(ye[mask_ss] ** 2))
                jon_avgs[name] += rms / len(PATHS)
                print(f"{name}={rms:.3f}", end="  ")

            restore_disturbances()
            print()

        results_det[ss_name] = det_avgs
        results_jon[ss_name] = jon_avgs

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Average RMS y_e [m] across all 4 paths")
    print("=" * 70)
    ctrl_names = [c[0] for c in CONTROLLERS]
    header = f"{'Controller':>10s}"
    for ss_name in JONSWAP_SCENARIOS:
        header += f"  {'Det-'+ss_name:>10s}  {'JON-'+ss_name:>10s}  {'Δ%':>6s}"
    print(header)
    print("-" * len(header))

    for cn in ctrl_names:
        row = f"{cn:>10s}"
        for ss_name in JONSWAP_SCENARIOS:
            d = results_det[ss_name][cn]
            j = results_jon[ss_name][cn]
            pct = 100 * (j - d) / d if d > 0 else 0
            row += f"  {d:10.4f}  {j:10.4f}  {pct:+5.1f}%"
        print(row)

    # Check ranking preservation
    print("\n--- Ranking Comparison ---")
    for ss_name in JONSWAP_SCENARIOS:
        det_rank = sorted(ctrl_names, key=lambda c: results_det[ss_name][c])
        jon_rank = sorted(ctrl_names, key=lambda c: results_jon[ss_name][c])
        match = det_rank == jon_rank
        print(f"  {ss_name}: Deterministic ranking = {det_rank}")
        print(f"  {ss_name}: JONSWAP      ranking = {jon_rank}")
        print(f"  {ss_name}: Rankings {'MATCH ✓' if match else 'DIFFER ✗'}")
        # Check top-2
        print(f"  {ss_name}: #1 Det={det_rank[0]}, #1 JON={jon_rank[0]}, "
              f"#2 Det={det_rank[1]}, #2 JON={jon_rank[1]}")
        print()

    # Generate comparison bar chart
    print("--- Generating comparison bar chart ---")
    plot_comparison_bars(results_det, results_jon)

    # Save results to file
    with open("results_jonswap.txt", "w") as f:
        f.write("JONSWAP Wave Spectrum Validation Results\n")
        f.write("=" * 50 + "\n\n")
        for ss_name in JONSWAP_SCENARIOS:
            f.write(f"\n{JONSWAP_SCENARIOS[ss_name]['label']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Controller':>10s}  {'Deterministic':>13s}  {'JONSWAP':>10s}  {'Δ%':>6s}\n")
            for cn in ctrl_names:
                d = results_det[ss_name][cn]
                j = results_jon[ss_name][cn]
                pct = 100 * (j - d) / d if d > 0 else 0
                f.write(f"{cn:>10s}  {d:13.4f}  {j:10.4f}  {pct:+5.1f}%\n")
    print("  Saved: results_jonswap.txt")

    print("\n✓ JONSWAP validation study complete.")


if __name__ == "__main__":
    main()
