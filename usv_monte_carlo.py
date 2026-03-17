#!/usr/bin/env python3
"""
Monte Carlo Robustness Study for USV Controllers
==================================================

Runs N_TRIALS simulations per controller with randomly varied:
  - Plant parameters: mass ±20%, damping ±30% (drawn uniformly per trial)
  - Disturbance scaling: ±50% variation around SS3 baseline
  - Sensor noise seed: varied per trial

All five controllers share the same plant realisation per trial
(fair comparison). Controllers still use nominal CONFIG values.

Produces:
  - Box-plot of RMS y_e across trials (per controller)
  - Bar chart of mean ± std RMS y_e
  - Convergence plot (running mean vs trial count)
  - Summary table with statistics (mean, std, CV, min, max)
  - Results saved to results_montecarlo.txt

Usage:
    python usv_monte_carlo.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from usv_common import (
    CONFIG, wrap_angle,
    get_waypoints_for_path_type,
    USVDynamics, LOSGuidance,
)
from usv_run_all import (
    CONTROLLERS, COLORS, LINE_STYLES, LINE_WIDTHS,
    set_disturbance_scale, restore_disturbances,
)

# =============================================================================
# Monte Carlo Configuration
# =============================================================================

N_TRIALS = 30              # number of MC trials
PATH_TYPES = ["custom", "rectangular"]  # paths for MC study
T_FINAL_MAP = {"custom": 300.0, "rectangular": 350.0}

# Sea states to evaluate
SEA_STATES = {
    "ss1": {"scale": 0.10, "label": "SS1 — Calm (Hs ≈ 0.1 m)"},
    "ss2": {"scale": 0.35, "label": "SS2 — Smooth (Hs ≈ 0.5 m)"},
    "ss3": {"scale": 0.60, "label": "SS3 — Slight (Hs ≈ 1.0 m)"},
}

# Uncertainty ranges (wider than nominal for robustness testing)
MC_UNCERTAINTY = {
    "m_u": 0.20,   # ±20% mass
    "m_v": 0.20,
    "I_r": 0.20,
    "a_u": 0.30,   # ±30% damping
    "a_v": 0.30,
    "a_r": 0.30,
}

# Disturbance variation: scale ∈ [0.75×, 1.25×] of BASE_DIST_SCALE
DIST_VARIATION = 0.25  # ±25% of base scale

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_montecarlo.txt")


# =============================================================================
# MC Simulation Runner
# =============================================================================

def run_mc_simulation(controller, ctrl_name, waypoints, plant_params,
                      dist_scale, noise_seed, path_type, t_final):
    """Run a single MC simulation with given plant parameters."""
    cfg = {**CONFIG}
    cfg["path_type"] = path_type
    cfg["t_final"] = t_final
    cfg["waypoints"] = waypoints

    # Set disturbance scale
    set_disturbance_scale(dist_scale)

    # Initial state
    wp0, wp1 = waypoints[0], waypoints[1]
    init_heading = np.arctan2(wp1[1] - wp0[1], wp1[0] - wp0[0])
    state = np.array([wp0[0], wp0[1], init_heading, 0.0, 0.0, 0.0])

    # Create dynamics with perturbed plant (uncertainty applied manually)
    dyn = USVDynamics(
        m_u=plant_params["m_u"], m_v=plant_params["m_v"],
        I_r=plant_params["I_r"],
        a_u=plant_params["a_u"], a_v=plant_params["a_v"],
        a_r=plant_params["a_r"],
        L=cfg["L_thruster"],
        F_min=cfg["F_min"], F_max=cfg["F_max"],
        apply_uncertainty=False,  # already perturbed
    )
    los = LOSGuidance(
        waypoints=waypoints,
        Delta=cfg["Delta"], k_y=cfg["k_y"],
        u_min=cfg["u_min"], u_max=cfg["u_max"],
        beta=cfg["beta"],
        use_ilos=cfg["use_ilos"], k_I=cfg["k_I"],
    )

    dt = cfg["dt"]
    N = int(t_final / dt)
    noise_rng = np.random.RandomState(noise_seed)

    ye_arr = np.zeros(N)
    tau_u_arr = np.zeros(N)
    tau_r_arr = np.zeros(N)
    t_arr = np.zeros(N)

    def ode_rhs(s, ctrl, t_now):
        return dyn.state_derivative(s, ctrl, t_now)

    for k in range(N):
        t = k * dt
        x, y, psi, u, v, r = state

        # Add sensor noise
        x_m = x + noise_rng.randn() * cfg["noise_std_pos"]
        y_m = y + noise_rng.randn() * cfg["noise_std_pos"]
        psi_m = psi + noise_rng.randn() * cfg["noise_std_psi"]
        u_m = u + noise_rng.randn() * cfg["noise_std_u"]
        r_m = r + noise_rng.randn() * cfg["noise_std_r"]

        u_d, psi_d, y_e, gamma = los.update(x_m, y_m, psi_m, dt)
        tau_u, tau_r = controller.compute_control(
            u_m, r_m, y_e, psi_m, psi_d, u_d, gamma)
        F_L, F_R = dyn.map_tau_to_thrusters(tau_u, tau_r)
        tau_u_act, tau_r_act = dyn.map_thrusters_to_tau(F_L, F_R)

        # True cross-track error
        idx = min(los.segment_idx, los.n_wps - 2)
        dp = np.array([x, y]) - los.waypoints[idx]
        y_e_true = float(los.seg_normal[idx] @ dp)

        t_arr[k] = t
        ye_arr[k] = y_e_true
        tau_u_arr[k] = tau_u_act
        tau_r_arr[k] = tau_r_act

        # RK4 integration
        k1 = ode_rhs(state, (tau_u_act, tau_r_act), t)
        k2 = ode_rhs(state + 0.5 * dt * k1, (tau_u_act, tau_r_act), t + 0.5 * dt)
        k3 = ode_rhs(state + 0.5 * dt * k2, (tau_u_act, tau_r_act), t + 0.5 * dt)
        k4 = ode_rhs(state + dt * k3, (tau_u_act, tau_r_act), t + dt)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        state[2] = wrap_angle(state[2])

    # Steady-state metrics (t > 30 s)
    ss_mask = t_arr > 30.0
    rms_ye = float(np.sqrt(np.mean(ye_arr[ss_mask] ** 2)))
    max_ye = float(np.max(np.abs(ye_arr[ss_mask])))
    rms_tau_u = float(np.sqrt(np.mean(tau_u_arr[ss_mask] ** 2)))
    rms_tau_r = float(np.sqrt(np.mean(tau_r_arr[ss_mask] ** 2)))
    energy = float(np.sum(np.abs(tau_u_arr) + np.abs(tau_r_arr)) * dt)

    restore_disturbances()

    return {
        "rms_ye": rms_ye,
        "max_ye": max_ye,
        "rms_tau_u": rms_tau_u,
        "rms_tau_r": rms_tau_r,
        "energy": energy,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_mc_boxplot(mc_results, save_path, ss_label=""):
    """Box plot of RMS y_e across MC trials for each controller."""
    ctrl_names = list(mc_results.keys())
    # Clip outliers >= 10 m for cleaner visualisation
    data = [np.clip(mc_results[c]["rms_ye"], None, 10.0) for c in ctrl_names]
    colors_list = [COLORS.get(c, "#333333") for c in ctrl_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, labels=ctrl_names,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylim(0, 10)
    ax.set_ylabel("Steady-State RMS $y_e$ [m]", fontsize=12)
    title = f"Monte Carlo Robustness — {N_TRIALS} Trials"
    if ss_label:
        title += f"\n{ss_label}"
    title += (f"\n(mass ±{int(MC_UNCERTAINTY['m_u']*100)}%, "
              f"damping ±{int(MC_UNCERTAINTY['a_u']*100)}%, "
              f"disturbance ±{int(DIST_VARIATION*100)}%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    eps_path = save_path.replace("plots/", "plots-eps/").replace(".png", ".eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mc_bar(mc_results, save_path, ss_label=""):
    """Bar chart with error bars (mean ± std) of RMS y_e."""
    ctrl_names = list(mc_results.keys())
    means = [np.mean(mc_results[c]["rms_ye"]) for c in ctrl_names]
    stds = [np.std(mc_results[c]["rms_ye"]) for c in ctrl_names]
    colors_list = [COLORS.get(c, "#333333") for c in ctrl_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ctrl_names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.5,
                  color=colors_list, alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        cv = 100 * s / m if m > 0 else 0
        ax.text(i, m + s + 0.05, f"{m:.2f}±{s:.2f}\nCV={cv:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ctrl_names, fontsize=11)
    ax.set_ylabel("Mean RMS $y_e$ (ss) [m]", fontsize=12)
    title = f"Monte Carlo Robustness — Mean ± Std ({N_TRIALS} Trials)"
    if ss_label:
        title += f"\n{ss_label}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    eps_path = save_path.replace("plots/", "plots-eps/").replace(".png", ".eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mc_convergence(mc_results, save_path, ss_label=""):
    """Running mean of RMS y_e vs trial number (convergence plot)."""
    ctrl_names = list(mc_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in ctrl_names:
        vals = mc_results[c]["rms_ye"]
        running_mean = np.cumsum(vals) / np.arange(1, len(vals) + 1)
        ax.plot(np.arange(1, len(vals) + 1), running_mean,
                color=COLORS.get(c, "#333333"),
                ls=LINE_STYLES.get(c, "-"),
                lw=LINE_WIDTHS.get(c, 1.5),
                label=c, alpha=0.9)

    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Running Mean RMS $y_e$ (ss) [m]", fontsize=12)
    title = f"Monte Carlo Convergence — {N_TRIALS} Trials"
    if ss_label:
        title += f"\n{ss_label}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper center", ncol=6, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    eps_path = save_path.replace("plots/", "plots-eps/").replace(".png", ".eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mc_scatter(mc_results, save_path, ss_label=""):
    """Scatter plot: RMS y_e vs total energy for each trial."""
    ctrl_names = list(mc_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for c in ctrl_names:
        rms = mc_results[c]["rms_ye"]
        energy = mc_results[c]["energy"]
        ax.scatter(energy, rms, color=COLORS.get(c, "#333333"),
                   label=c, alpha=0.6, s=40, edgecolor="black", linewidth=0.3)
        # Draw mean point
        ax.scatter(np.mean(energy), np.mean(rms),
                   color=COLORS.get(c, "#333333"),
                   marker="D", s=100, edgecolor="black", linewidth=1.5,
                   zorder=10)

    ax.set_xlabel("Total Control Energy", fontsize=12)
    ax.set_ylabel("Steady-State RMS $y_e$ [m]", fontsize=12)
    title = f"Tracking Accuracy vs Control Energy — {N_TRIALS} MC Trials"
    if ss_label:
        title += f"\n{ss_label}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper center", ncol=6, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    eps_path = save_path.replace("plots/", "plots-eps/").replace(".png", ".eps")
    os.makedirs(os.path.dirname(eps_path), exist_ok=True)
    plt.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(OUTPUT_DIR), "plots-eps"), exist_ok=True)

    path_str = ", ".join(f"{p} ({T_FINAL_MAP[p]:.0f}s)" for p in PATH_TYPES)
    sims_per_ss = N_TRIALS * len(CONTROLLERS) * len(PATH_TYPES)
    total_sims = sims_per_ss * len(SEA_STATES)

    print("=" * 70)
    print("  MONTE CARLO ROBUSTNESS STUDY")
    print(f"  Controllers: {len(CONTROLLERS)}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Paths: {path_str}")
    print(f"  Sea States: {', '.join(SEA_STATES.keys())}")
    print(f"  Uncertainty: mass ±{int(MC_UNCERTAINTY['m_u']*100)}%, "
          f"damping ±{int(MC_UNCERTAINTY['a_u']*100)}%")
    print(f"  Disturbance variation: ±{int(DIST_VARIATION*100)}%")
    print(f"  Simulations per sea state: {sims_per_ss}")
    print(f"  Total simulations: {total_sims}")
    print("=" * 70)

    ctrl_names = [c[0] for c in CONTROLLERS]
    all_lines = []  # accumulate text output for results file

    global_sim = 0
    t_start_global = time.time()

    for ss_name, ss_cfg in SEA_STATES.items():
        base_scale = ss_cfg["scale"]
        ss_label = ss_cfg["label"]

        print(f"\n{'─' * 70}")
        print(f"  Sea State: {ss_label}  (base scale = {base_scale})")
        print(f"{'─' * 70}")

        # Master RNG for generating per-trial seeds (same seed per SS for reproducibility)
        master_rng = np.random.RandomState(2024)

        # Pre-generate all trial parameters (same plant for all controllers)
        trial_params = []
        for trial in range(N_TRIALS):
            rng = np.random.RandomState(master_rng.randint(0, 2**31))

            # Perturbed plant parameters (uniform ±range around nominal)
            pp = {}
            for param, unc_range in MC_UNCERTAINTY.items():
                cfg_key = {
                    "m_u": "m_u", "m_v": "m_v", "I_r": "I_r",
                    "a_u": "a_u", "a_v": "a_v", "a_r": "a_r",
                }[param]
                nominal = CONFIG[cfg_key]
                pp[cfg_key] = nominal * (1.0 + unc_range * (2.0 * rng.rand() - 1.0))

            # Disturbance scale variation around this SS baseline
            dist_scale = base_scale * (1.0 + DIST_VARIATION * (2.0 * rng.rand() - 1.0))

            # Sensor noise seed (different per trial)
            noise_seed = rng.randint(0, 2**31)

            trial_params.append({
                "plant_params": pp,
                "dist_scale": dist_scale,
                "noise_seed": noise_seed,
            })

        # Run MC simulations across all paths
        per_path_results = {}
        t_start = time.time()
        sim_count = 0

        for path_type in PATH_TYPES:
            t_final = T_FINAL_MAP[path_type]
            waypoints = get_waypoints_for_path_type(path_type)
            per_path_results[path_type] = {
                c: {"rms_ye": [], "max_ye": [], "rms_tau_u": [],
                    "rms_tau_r": [], "energy": []}
                for c in ctrl_names
            }

            for trial_idx in range(N_TRIALS):
                tp = trial_params[trial_idx]

                for ctrl_name, ctrl_factory in CONTROLLERS:
                    sim_count += 1
                    global_sim += 1
                    ctrl = ctrl_factory()

                    sys.stdout.write(
                        f"\r  [{global_sim:4d}/{total_sims}] {ss_name.upper()} | "
                        f"Path: {path_type:<12s} Trial {trial_idx + 1:3d}/{N_TRIALS} | "
                        f"{ctrl_name:<12s} ..."
                    )
                    sys.stdout.flush()

                    result = run_mc_simulation(
                        ctrl, ctrl_name, waypoints,
                        plant_params=tp["plant_params"],
                        dist_scale=tp["dist_scale"],
                        noise_seed=tp["noise_seed"],
                        path_type=path_type,
                        t_final=t_final,
                    )

                    for key in per_path_results[path_type][ctrl_name]:
                        per_path_results[path_type][ctrl_name][key].append(result[key])

        elapsed = time.time() - t_start
        print(f"\n\n  {ss_name.upper()}: {sims_per_ss} simulations completed in {elapsed:.1f} s\n")

        # Convert lists to arrays
        for path_type in PATH_TYPES:
            for c in ctrl_names:
                for key in per_path_results[path_type][c]:
                    per_path_results[path_type][c][key] = np.array(
                        per_path_results[path_type][c][key])

        # Aggregate across paths
        mc_results = {c: {"rms_ye": np.zeros(N_TRIALS), "max_ye": np.zeros(N_TRIALS),
                          "rms_tau_u": np.zeros(N_TRIALS), "rms_tau_r": np.zeros(N_TRIALS),
                          "energy": np.zeros(N_TRIALS)}
                      for c in ctrl_names}
        for c in ctrl_names:
            for key in mc_results[c]:
                for path_type in PATH_TYPES:
                    mc_results[c][key] = mc_results[c][key] + \
                        per_path_results[path_type][c][key] / len(PATH_TYPES)

        # ── Print Summary for this SS ──
        lines = []
        lines.append("=" * 80)
        lines.append(f"  MONTE CARLO — {ss_label}")
        lines.append(f"  {N_TRIALS} Trials × {len(PATH_TYPES)} Paths = "
                     f"{sims_per_ss} Simulations")
        lines.append(f"  Paths: {path_str}")
        lines.append(f"  Base Disturbance Scale: {base_scale}")
        lines.append(f"  Uncertainty: mass ±{int(MC_UNCERTAINTY['m_u']*100)}%, "
                     f"damping ±{int(MC_UNCERTAINTY['a_u']*100)}%, "
                     f"disturbance ±{int(DIST_VARIATION*100)}%")
        lines.append("=" * 80)

        for path_type in PATH_TYPES:
            lines.append(f"\n  ── Path: {path_type} ──")
            lines.append(f"  {'Controller':<12s}  {'Mean RMS':>10s}  {'Std':>8s}  "
                         f"{'CV%':>6s}  {'Min':>8s}  {'Max':>8s}")
            lines.append(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}")
            ranked_path = sorted(ctrl_names,
                                 key=lambda c: np.mean(per_path_results[path_type][c]["rms_ye"]))
            for c in ranked_path:
                rms = per_path_results[path_type][c]["rms_ye"]
                mean_rms = np.mean(rms)
                std_rms = np.std(rms)
                cv = 100 * std_rms / mean_rms if mean_rms > 0 else 0
                lines.append(
                    f"  {c:<12s}  {mean_rms:10.4f}  {std_rms:8.4f}  "
                    f"{cv:5.1f}%  {np.min(rms):8.4f}  {np.max(rms):8.4f}"
                )

        # Aggregated summary
        lines.append(f"\n\n  ══ AGGREGATED (averaged across {len(PATH_TYPES)} paths) ══")
        lines.append(f"  {'Controller':<12s}  {'Mean RMS':>10s}  {'Std':>8s}  "
                     f"{'CV%':>6s}  {'Min':>8s}  {'Max':>8s}  "
                     f"{'Mean Energy':>12s}")
        lines.append(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*12}")

        ranked = sorted(ctrl_names, key=lambda c: np.mean(mc_results[c]["rms_ye"]))
        for c in ranked:
            rms = mc_results[c]["rms_ye"]
            energy = mc_results[c]["energy"]
            mean_rms = np.mean(rms)
            std_rms = np.std(rms)
            cv = 100 * std_rms / mean_rms if mean_rms > 0 else 0
            lines.append(
                f"  {c:<12s}  {mean_rms:10.4f}  {std_rms:8.4f}  "
                f"{cv:5.1f}%  {np.min(rms):8.4f}  {np.max(rms):8.4f}  "
                f"{np.mean(energy):12.0f}"
            )

        lines.append("")
        lines.append(f"  ── Rankings ({ss_name.upper()}, by mean RMS y_e) ──")
        for rank, c in enumerate(ranked, 1):
            mean_rms = np.mean(mc_results[c]["rms_ye"])
            std_rms = np.std(mc_results[c]["rms_ye"])
            lines.append(f"    #{rank}: {c:<12s}  {mean_rms:.4f} ± {std_rms:.4f} m")

        lines.append("")
        best = ranked[0]
        second = ranked[1]
        best_mean = np.mean(mc_results[best]["rms_ye"])
        second_mean = np.mean(mc_results[second]["rms_ye"])
        improvement = 100 * (second_mean - best_mean) / second_mean
        lines.append(f"  Key: {best} achieves {improvement:.1f}% lower mean RMS y_e "
                     f"than next-best {second}")

        table_str = "\n".join(lines)
        print(table_str)
        all_lines.extend(lines)
        all_lines.append("\n")

        # ── Generate Plots for this sea state ──
        print(f"\n  Generating {ss_name.upper()} Monte Carlo plots...")
        plot_mc_boxplot(mc_results,
                        os.path.join(OUTPUT_DIR, f"mc_boxplot_{ss_name}.png"),
                        ss_label=ss_label)
        plot_mc_bar(mc_results,
                    os.path.join(OUTPUT_DIR, f"mc_bar_meanstd_{ss_name}.png"),
                    ss_label=ss_label)
        plot_mc_convergence(mc_results,
                            os.path.join(OUTPUT_DIR, f"mc_convergence_{ss_name}.png"),
                            ss_label=ss_label)
        plot_mc_scatter(mc_results,
                        os.path.join(OUTPUT_DIR, f"mc_scatter_energy_{ss_name}.png"),
                        ss_label=ss_label)

    # ── Save combined results ──
    elapsed_total = time.time() - t_start_global
    all_lines.insert(0, f"  Total: {total_sims} simulations in {elapsed_total:.1f} s\n")
    all_lines.insert(0, "  MONTE CARLO ROBUSTNESS STUDY — ALL SEA STATES")
    all_lines.insert(0, "=" * 80)

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"\n  Results saved to {RESULTS_FILE}")

    print(f"\n{'=' * 70}")
    print(f"  Monte Carlo study complete! ({total_sims} sims, "
          f"{elapsed_total:.1f} s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
