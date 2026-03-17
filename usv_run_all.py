#!/usr/bin/env python3
"""
Run all 7 USV controllers across all path types and disturbance levels.

Produces:
  - Per-controller per-path summary table (printed + saved to results.txt)
  - Trajectory comparison plots (all controllers overlaid)
  - Cross-track error comparison plots
  - Control effort comparison plots
  - Individual controller plots per path

All plots are saved as PNG files in the output directory.

Usage:
    python usv_run_all.py

Dependencies: numpy, scipy, matplotlib, usv_common + all controller modules
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from usv_common import (
    CONFIG, wrap_angle,
    run_simulation, print_summary,
    get_waypoints_for_path_type,
)

# Import all controller classes
from usv_lqr_sim import LQRController
from usv_smc_sim import SMCController
from usv_adrc_sim import ADRCController
from usv_ntsmc_eso_sim import ANTSMCController
from usv_asmc_sim import ASMCController
from usv_fntsmc_sim import FNTSMCController


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")

PATH_TYPES = ["custom", "circular", "rectangular", "zigzag"]
T_FINAL_MAP = {
    "custom": 300.0,
    "circular": 350.0,
    "rectangular": 350.0,
    "zigzag": 300.0,
}

# Disturbance scaling levels  (base = CONFIG at scale 1.0)
# ──────────────────────────────────────────────────────────────────────
# Scale   V_wind   V_current  Beaufort  WMO Sea State         H_s
# ──────────────────────────────────────────────────────────────────────
# 0.10×   ~1 m/s   0.05 m/s   1        SS 1 (Calm/rippled)   0–0.1 m
# 0.35×   ~3.5 m/s 0.18 m/s   2–3      SS 2 (Smooth)         0.1–0.5 m
# 0.60×   ~6 m/s   0.30 m/s   3–4      SS 3 (Slight)         0.5–1.25 m
# ──────────────────────────────────────────────────────────────────────
# Note: Sea State 4+ omitted — small USVs are not recommended to
# operate above Sea State 3 (Hs > 1.25 m) per IMO guidelines.
DISTURBANCE_LEVELS = {
    "ss1": 0.10,
    "ss2": 0.35,
    "ss3": 0.60,
}

# Display labels for disturbance levels (used in plots and tables)
DIST_LABELS = {
    "ss1": "Sea State 1 – Calm (Beaufort 1, Hs ≈ 0–0.1 m)",
    "ss2": "Sea State 2 – Smooth (Beaufort 2–3, Hs ≈ 0.1–0.5 m)",
    "ss3": "Sea State 3 – Slight (Beaufort 3–4, Hs ≈ 0.5–1.25 m)",
}

# Short labels for plot titles
DIST_SHORT = {
    "ss1": "Sea State 1 – Calm",
    "ss2": "Sea State 2 – Smooth",
    "ss3": "Sea State 3 – Slight",
}

# Controller definitions: (name, factory_function)
CONTROLLERS = [
    ("LQR",      lambda: LQRController()),
    ("SMC",      lambda: SMCController()),
    ("ASMC",     lambda: ASMCController()),
    ("ADRC",     lambda: ADRCController()),
    ("FNTSMC",   lambda: FNTSMCController()),
    ("ANTSMC",    lambda: ANTSMCController()),
]

# Colors for each controller (consistent across all plots)
COLORS = {
    "LQR":      "#ff7f0e",   # orange
    "SMC":      "#2ca02c",   # green
    "ASMC":     "#d62728",   # red (ablation baseline)
    "ADRC":     "#9467bd",   # purple
    "FNTSMC":   "#8c564b",   # brown (Fan2021 literature)
    "ANTSMC":   "#17becf",   # teal (novel controller)
}

LINE_STYLES = {
    "LQR":      "--",
    "SMC":      "-.",
    "ASMC":     ":",
    "ADRC":     "-",
    "FNTSMC":   (0, (3, 1, 1, 1)),   # dash-dot-dot
    "ANTSMC":   "-",
}

LINE_WIDTHS = {
    "LQR":      2.0,
    "SMC":      2.0,
    "ASMC":     2.0,
    "ADRC":     2.0,
    "FNTSMC":   2.0,
    "ANTSMC":   2.0,
}


# =============================================================================
# Disturbance Scaling Helper
# =============================================================================

# Original (unscaled) disturbance values — saved once at import time
_ORIG_DIST = {
    "d_u_const":      CONFIG["d_u_const"],
    "d_v_const":      CONFIG["d_v_const"],
    "d_r_const":      CONFIG["d_r_const"],
    "d_u_wave_amp":   CONFIG["d_u_wave_amp"],
    "d_v_wave_amp":   CONFIG["d_v_wave_amp"],
    "d_r_wave_amp":   CONFIG["d_r_wave_amp"],
    "d_wave_freq":    CONFIG["d_wave_freq"],
    "V_current":      CONFIG["V_current"],
    "beta_current":   CONFIG["beta_current"],
    "F_wind_u":       CONFIG["F_wind_u"],
    "F_wind_v":       CONFIG["F_wind_v"],
    "F_wind_r":       CONFIG["F_wind_r"],
    "wind_gust_amp_u": CONFIG["wind_gust_amp_u"],
    "wind_gust_amp_v": CONFIG["wind_gust_amp_v"],
    "wind_gust_amp_r": CONFIG["wind_gust_amp_r"],
    "wind_gust_freq": CONFIG["wind_gust_freq"],
}


def set_disturbance_scale(scale):
    """Patch the global CONFIG dict so that disturbance functions use scaled values.

    The disturbance_u/disturbance_r functions in usv_common.py read directly
    from CONFIG, so we must modify CONFIG in-place for scaling to take effect.
    """
    CONFIG["d_u_const"]      = _ORIG_DIST["d_u_const"]      * scale
    CONFIG["d_v_const"]      = _ORIG_DIST["d_v_const"]      * scale
    CONFIG["d_r_const"]      = _ORIG_DIST["d_r_const"]      * scale
    CONFIG["d_u_wave_amp"]   = _ORIG_DIST["d_u_wave_amp"]   * scale
    CONFIG["d_v_wave_amp"]   = _ORIG_DIST["d_v_wave_amp"]   * scale
    CONFIG["d_r_wave_amp"]   = _ORIG_DIST["d_r_wave_amp"]   * scale
    CONFIG["V_current"]      = _ORIG_DIST["V_current"]      * scale
    CONFIG["F_wind_u"]       = _ORIG_DIST["F_wind_u"]       * scale
    CONFIG["F_wind_v"]       = _ORIG_DIST["F_wind_v"]       * scale
    CONFIG["F_wind_r"]       = _ORIG_DIST["F_wind_r"]       * scale
    CONFIG["wind_gust_amp_u"] = _ORIG_DIST["wind_gust_amp_u"] * scale
    CONFIG["wind_gust_amp_v"] = _ORIG_DIST["wind_gust_amp_v"] * scale
    CONFIG["wind_gust_amp_r"] = _ORIG_DIST["wind_gust_amp_r"] * scale


def restore_disturbances():
    """Restore original disturbance values in CONFIG."""
    for k, v in _ORIG_DIST.items():
        CONFIG[k] = v


def get_scaled_config(path_type):
    """Return a simulation config dict for the given path type.

    Disturbance scaling is handled separately via set_disturbance_scale()
    which patches the global CONFIG that disturbance functions read from.
    """
    return {
        "path_type": path_type,
        "t_final": T_FINAL_MAP[path_type],
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def _find_nearest_index(x_arr, y_arr, target_x, target_y):
    """Find the time-index closest to a target (x,y) point."""
    dist2 = (x_arr - target_x)**2 + (y_arr - target_y)**2
    return int(np.argmin(dist2))


def _trim_to_path(data, waypoints):
    """Trim a simulation data dict so it starts at the first waypoint and
    ends when the USV is closest to the last waypoint (after reaching
    at least 80 % of the path).  Returns (x, y) arrays of the trimmed
    trajectory, plus the trim indices (i_start, i_end).

    This ensures all controllers visually start and end at the same
    reference locations in the trajectory plot.
    """
    x = data["x"]
    y = data["y"]
    N = len(x)

    wp_start = waypoints[0]
    wp_end   = waypoints[-1]

    # --- Start index: nearest to first waypoint (usually index 0) ---
    i_start = _find_nearest_index(x, y, wp_start[0], wp_start[1])

    # --- End index: nearest to last waypoint, but only search the
    #     last 50 % of the trajectory so we don't accidentally pick
    #     the start point on closed paths (circular, rectangular). ---
    half = max(N // 2, i_start + 1)
    dist2_end = (x[half:] - wp_end[0])**2 + (y[half:] - wp_end[1])**2
    i_end = half + int(np.argmin(dist2_end))

    return x[i_start:i_end+1], y[i_start:i_end+1], i_start, i_end


def plot_trajectory_comparison(all_data, path_type, dist_level, save_dir):
    """Overlay all controller trajectories on one plot.

    All trajectories are trimmed so they visually start from the first
    waypoint and end at (or near) the last waypoint.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot reference path (from first controller's data)
    wps = all_data[0]["waypoints"]
    ax.plot(wps[:, 0], wps[:, 1], "ks--", ms=3, lw=1.5, label="Reference", zorder=10)

    # Mark start waypoint with annotation (no legend entry)
    ax.plot(wps[0, 0], wps[0, 1], "g^", ms=12, zorder=15)
    ax.annotate("Start", (wps[0, 0], wps[0, 1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=11, fontweight="bold", color="green", zorder=16)
    if np.linalg.norm(wps[0] - wps[-1]) > 1.0:
        # Open path — mark distinct end point (no legend entry)
        ax.plot(wps[-1, 0], wps[-1, 1], "rv", ms=12, zorder=15)

    for data in all_data:
        name = data["controller"]
        xt, yt, _, _ = _trim_to_path(data, wps)
        ax.plot(xt, yt,
                color=COLORS[name], ls=LINE_STYLES[name],
                lw=LINE_WIDTHS[name], label=name, alpha=0.85)

    # Styling
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]", fontsize=18)
    ax.set_ylabel("y [m]", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Simplified title per request: just "Sea state X"
    if dist_level == "none":
        title_text = "No Disturbance"
    elif dist_level == "ss1":
        title_text = "Sea State 1"
    elif dist_level == "ss2":
        title_text = "Sea State 2"
    elif dist_level == "ss3":
        title_text = "Sea State 3"
    else:
        title_text = dist_level.capitalize()
        
    ax.set_title(title_text, fontsize=20, fontweight="bold")
    ax.legend(fontsize=13, loc="upper center", ncol=4, framealpha=0.9)
    fig.tight_layout()

    fname = os.path.join(save_dir, f"traj_{path_type}_{dist_level}.png")
    fname_eps = os.path.join("plots-eps", f"traj_{path_type}_{dist_level}.eps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(fname_eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_crosstrack_comparison(all_data, path_type, dist_level, save_dir):
    """Overlay all controllers' cross-track error on one plot."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for data in all_data:
        name = data["controller"]
        ax.plot(data["t"], data["y_e"],
                color=COLORS[name], ls=LINE_STYLES[name],
                lw=LINE_WIDTHS[name], label=name, alpha=0.85)

    ax.axhline(0, color="gray", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Cross-Track Error y_e [m]", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    if dist_level == "none":
        title_text = "Cross-Tracking Error - No Disturbance"
    elif dist_level == "ss1":
        title_text = "Cross-Tracking Error - Sea State 1"
    elif dist_level == "ss2":
        title_text = "Cross-Tracking Error - Sea State 2"
    elif dist_level == "ss3":
        title_text = "Cross-Tracking Error - Sea State 3"
    else:
        title_text = f"Cross-Tracking Error - {dist_level.capitalize()}"
        
    ax.set_title(title_text, fontsize=20, fontweight="bold")
    ax.legend(fontsize=13, loc="upper center", ncol=6, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    fname = os.path.join(save_dir, f"error_{path_type}_{dist_level}.png")
    fname_eps = os.path.join("plots-eps", f"error_{path_type}_{dist_level}.eps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(fname_eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_control_comparison(all_data, path_type, dist_level, save_dir):
    """Overlay all controllers' control efforts on one plot (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for data in all_data:
        name = data["controller"]
        ax1.plot(data["t"], data["tau_u"],
                 color=COLORS[name], ls=LINE_STYLES[name],
                 lw=LINE_WIDTHS[name], label=name, alpha=0.8)
        ax2.plot(data["t"], data["tau_r"],
                 color=COLORS[name], ls=LINE_STYLES[name],
                 lw=LINE_WIDTHS[name], label=name, alpha=0.8)

    # Top subplot: tau_u
    ax1.axhline(0, color="gray", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("τ_u [N]", fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    dist_label = "No Dist" if dist_level == "none" else f"{dist_level.capitalize()} Dist"
    ax1.set_title(f"Control Inputs ({path_type}, {dist_label})",
                  fontsize=20, fontweight="bold")
    ax1.legend(fontsize=12, loc="upper center", ncol=6, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Bottom subplot: tau_r
    ax2.axhline(0, color="gray", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Time [s]", fontsize=18)
    ax2.set_ylabel("τ_r [N·m]", fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(fontsize=12, loc="upper center", ncol=6, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    fname = os.path.join(save_dir, f"control_{path_type}_{dist_level}.png")
    fname_eps = os.path.join("plots-eps", f"control_{path_type}_{dist_level}.eps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(fname_eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_heading_comparison(all_data, path_type, dist_level, save_dir):
    """4-subplot heading comparison: each subplot shows ψ vs ψ_d for one controller."""
    from scipy.ndimage import uniform_filter1d

    n_ctrl = len(all_data)
    fig, axes = plt.subplots(n_ctrl, 1, figsize=(12, 2.6 * n_ctrl), sharex=True)
    if n_ctrl == 1:
        axes = [axes]

    for i, data in enumerate(all_data):
        ax = axes[i]
        name = data["controller"]
        t = data["t"]
        # Smooth psi_d (LOS output) — 1-second moving average removes
        # wave-induced jitter while preserving waypoint transitions.
        dt = t[1] - t[0] if len(t) > 1 else 0.05
        win = max(int(round(1.0 / dt)), 1)
        psi_d_smooth = uniform_filter1d(np.degrees(data["psi_d"]), size=win)
        psi_smooth = uniform_filter1d(np.degrees(data["psi"]), size=win)
        ax.plot(t, psi_d_smooth, "k--", lw=LINE_WIDTHS[name], label=r"$\psi_d$")
        ax.plot(t, psi_smooth, color=COLORS[name], lw=LINE_WIDTHS[name], label=name)
        
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylabel("ψ [deg]", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=13, loc="upper center", ncol=2, framealpha=0.9)
        
    axes[-1].set_xlabel("Time [s]", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    fname = os.path.join(save_dir, f"heading_{path_type}_{dist_level}.png")
    fname_eps = os.path.join("plots-eps", f"heading_{path_type}_{dist_level}.eps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(fname_eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_course_comparison(all_data, path_type, dist_level, save_dir):
    """Overlay all controllers' course angle ψ on one plot with target course γ.

    The desired path tangent angle γ is recovered from γ = ψ − χ_e, where
    χ_e = wrap(ψ − γ) is stored by the simulation.  Because each controller
    follows a slightly different trajectory the segment-switching (and thus γ)
    may differ, so we plot γ from the best-tracking controller (first in the
    list whose RMS χ_e is smallest) as the single reference.
    """
    from scipy.ndimage import uniform_filter1d

    dl = DIST_SHORT.get(dist_level, dist_level.capitalize())

    # Recover γ per controller:  γ = ψ − χ_e
    gamma_per = []
    rms_chi = []
    for d in all_data:
        g = np.degrees(d["psi"]) - np.degrees(d["chi_e"])
        gamma_per.append(g)
        rms_chi.append(np.sqrt(np.mean(np.degrees(d["chi_e"])**2)))

    # Use γ from the controller with smallest RMS course error as reference
    best = int(np.argmin(rms_chi))
    t_ref = all_data[best]["t"]
    dt = t_ref[1] - t_ref[0] if len(t_ref) > 1 else 0.05
    win = max(int(round(1.0 / dt)), 1)
    gamma_ref = uniform_filter1d(gamma_per[best], size=win)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Target course
    ax.plot(t_ref, gamma_ref, "k--", linewidth=1.5,
            label="$\\chi_{d}$ (Target Course)")

    for i, d in enumerate(all_data):
        c = COLORS[d["controller"]]
        ax.plot(d["t"], np.degrees(d["psi"]), color=c,
                linewidth=LINE_WIDTHS[d["controller"]], label=d["controller"], alpha=0.9)

    ax.set_ylabel("Course Angle $\\chi$ [deg]", fontsize=18)
    ax.set_xlabel("Time [s]", fontsize=18)
    
    # Map dist_level to "Sea State X" for the title
    if dist_level.lower() == "ss1":
        title_str = "Course Tracking Sea State 1"
    elif dist_level.lower() == "ss2":
        title_str = "Course Tracking Sea State 2"
    elif dist_level.lower() == "ss3":
        title_str = "Course Tracking Sea State 3"
    else:
        title_str = f"Course Tracking {dl}"

    ax.set_title(title_str, fontsize=20, fontweight="bold", y=1.02)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=12, loc="upper center", ncol=4, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fname = os.path.join(save_dir, f"course_{path_type}_{dist_level}.png")
    fname_eps = os.path.join("plots-eps", f"course_{path_type}_{dist_level}.eps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(fname_eps, format="eps", bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_bar_charts(group_data, save_dir):
    """Bar chart comparing RMS y_e across all controllers, paths, and disturbance levels."""
    ctrl_names = [name for name, _ in CONTROLLERS]
    n_ctrl = len(ctrl_names)

    for dist_level in DISTURBANCE_LEVELS:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(PATH_TYPES))
        width = 0.12
        offsets = np.linspace(-(n_ctrl-1)/2 * width, (n_ctrl-1)/2 * width, n_ctrl)

        for i, cname in enumerate(ctrl_names):
            vals = []
            for ptype in PATH_TYPES:
                key = (cname, ptype, dist_level)
                vals.append(group_data.get(key, {}).get("rms_ye", 0.0))
            ax.bar(x + offsets[i], vals, width * 0.9,
                   label=cname, color=COLORS[cname], alpha=0.85)

        ax.set_xlabel("Path Type", fontsize=12)
        ax.set_ylabel("RMS Cross-Track Error [m]", fontsize=12)
        dl = DIST_SHORT.get(dist_level, dist_level.capitalize())
        ax.set_title(f"RMS y_e Comparison — {dl}",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in PATH_TYPES], fontsize=11)
        ax.legend(fontsize=10, loc="upper center", ncol=n_ctrl, framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(save_dir, f"bar_rms_ye_{dist_level}.png")
        fname_eps = os.path.join("plots-eps", f"bar_rms_ye_{dist_level}.eps")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        fig.savefig(fname_eps, format="eps", bbox_inches="tight")
        plt.close(fig)


def plot_overall_ranking(metrics, save_dir):
    """Overall ranking plot: average RMS y_e across all paths per disturbance level."""
    ctrl_names = [name for name, _ in CONTROLLERS]

    fig, axes = plt.subplots(1, len(DISTURBANCE_LEVELS), figsize=(18, 6))
    if len(DISTURBANCE_LEVELS) == 1:
        axes = [axes]

    for ax, (dist_level, _) in zip(axes, DISTURBANCE_LEVELS.items()):
        avg_rms = []
        for cname in ctrl_names:
            vals = []
            for ptype in PATH_TYPES:
                key = (cname, ptype, dist_level)
                vals.append(metrics.get(key, {}).get("rms_ye", 0.0))
            avg_rms.append(np.mean(vals))

        # Sort by performance (best = smallest bar at top)
        sorted_idx = np.argsort(avg_rms)
        sorted_names = [ctrl_names[i] for i in sorted_idx]
        sorted_vals = [avg_rms[i] for i in sorted_idx]
        sorted_colors = [COLORS[n] for n in sorted_names]

        bars = ax.barh(range(len(sorted_names)), sorted_vals,
                       color=sorted_colors, alpha=0.85, edgecolor="black", lw=0.5)

        # Add value labels at end of each bar
        x_max = max(sorted_vals) * 1.35
        for bar, val, name in zip(bars, sorted_vals, sorted_names):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=14)

        # Color-coded controller name labels on y-axis
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=15, fontweight="bold")
        for tick_label, name in zip(ax.get_yticklabels(), sorted_names):
            tick_label.set_color(COLORS[name])

        ax.set_xlabel("Avg. RMS $y_e$ [m]", fontsize=15)
        ax.set_xlim(0, x_max)
        dl = DIST_SHORT.get(dist_level, dist_level.capitalize())
        ax.set_title(dl, fontsize=16, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Overall Controller Ranking",
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()

    fname = os.path.join(save_dir, "ranking_overall.png")
    fname_eps = os.path.join("plots-eps", "ranking_overall.eps")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    fig.savefig(fname_eps, format="eps", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Results Table
# =============================================================================

def format_results_table(metrics):
    """Format a comprehensive results table."""
    ctrl_names = [name for name, _ in CONTROLLERS]
    lines = []

    lines.append("=" * 100)
    lines.append("  USV Controller Comparison — Full Results")
    lines.append("=" * 100)

    for dist_level in DISTURBANCE_LEVELS:
        dl = DIST_LABELS.get(dist_level, dist_level.upper())
        lines.append(f"\n{'━' * 100}")
        lines.append(f"  Disturbance: {dl}")
        lines.append(f"{'━' * 100}")
        lines.append(f"  {'Controller':<12s} {'Path':<14s} "
                      f"{'Max|y_e|':>10s} {'RMS y_e':>10s} {'RMS y_e(ss)':>12s} "
                      f"{'Max|χ_e|':>10s} "
                      f"{'RMS τ_u':>10s} {'RMS τ_r':>10s} {'Energy':>10s}")
        lines.append(f"  {'─' * 12} {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

        for cname in ctrl_names:
            for ptype in PATH_TYPES:
                key = (cname, ptype, dist_level)
                m = metrics.get(key, {})
                lines.append(
                    f"  {cname:<12s} {ptype:<14s} "
                    f"{m.get('max_ye', 0):10.3f} {m.get('rms_ye', 0):10.3f} "
                    f"{m.get('rms_ye_ss', 0):12.4f} "
                    f"{m.get('max_chi_e', 0):10.2f}° "
                    f"{m.get('rms_tau_u', 0):10.1f} {m.get('rms_tau_r', 0):10.1f} "
                    f"{m.get('total_energy', 0):10.0f}"
                )
            lines.append("")

        # Per-disturbance ranking
        lines.append(f"\n  ── Ranking (by avg steady-state RMS y_e, t > 30 s) ──")
        avg_rms = {}
        for cname in ctrl_names:
            vals = [metrics[(cname, p, dist_level)]["rms_ye_ss"] for p in PATH_TYPES
                    if (cname, p, dist_level) in metrics]
            avg_rms[cname] = np.mean(vals) if vals else 999.0
        ranked = sorted(avg_rms.items(), key=lambda x: x[1])
        for rank, (cname, val) in enumerate(ranked, 1):
            lines.append(f"    #{rank}: {cname:<12s}  avg RMS y_e (ss) = {val:.4f} m")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  MBZIRC USV — Run All Controllers")
    print(f"  Controllers: {len(CONTROLLERS)}")
    print(f"  Path types:  {len(PATH_TYPES)}")
    print(f"  Disturbance levels: {len(DISTURBANCE_LEVELS)}")
    total_runs = len(CONTROLLERS) * len(PATH_TYPES) * len(DISTURBANCE_LEVELS)
    print(f"  Total simulations: {total_runs}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    metrics = {}
    all_results = {}  # (path_type, dist_level) -> list of data dicts
    run_count = 0
    t_start = time.time()

    for dist_level, dist_scale in DISTURBANCE_LEVELS.items():
        # Patch global CONFIG so disturbance functions use scaled values
        set_disturbance_scale(dist_scale)

        for ptype in PATH_TYPES:
            data_list = []

            for ctrl_name, ctrl_factory in CONTROLLERS:
                run_count += 1
                ctrl = ctrl_factory()
                cfg = get_scaled_config(ptype)

                sys.stdout.write(
                    f"\r  [{run_count:3d}/{total_runs}] "
                    f"{ctrl_name:<12s} | {ptype:<14s} | dist={dist_level:<10s} ..."
                )
                sys.stdout.flush()

                data = run_simulation(
                    controller=ctrl,
                    config=cfg,
                    controller_name=ctrl_name,
                )
                data_list.append(data)

                # Compute metrics
                ye = data["y_e"]
                chi_e = data["chi_e"]
                tau_u = data["tau_u"]
                tau_r = data["tau_r"]
                t_arr = data["t"]
                dt = CONFIG["dt"]

                # Steady-state mask: discard first 30 s of transient
                ss_mask = t_arr > 30.0

                key = (ctrl_name, ptype, dist_level)
                metrics[key] = {
                    "max_ye": float(np.max(np.abs(ye))),
                    "rms_ye": float(np.sqrt(np.mean(ye**2))),
                    "rms_ye_ss": float(np.sqrt(np.mean(ye[ss_mask]**2))),
                    "max_ye_ss": float(np.max(np.abs(ye[ss_mask]))),
                    "max_chi_e": float(np.degrees(np.max(np.abs(chi_e)))),
                    "rms_tau_u": float(np.sqrt(np.mean(tau_u**2))),
                    "rms_tau_r": float(np.sqrt(np.mean(tau_r**2))),
                    # Total energy: integral of |tau_u * u| + |tau_r * r| over time
                    "total_energy": float(np.sum(
                        np.abs(tau_u * data["u"]) + np.abs(tau_r * data["r"])
                    ) * dt),
                    # IAE: Integral of Absolute Error (common benchmark metric)
                    "iae": float(np.sum(np.abs(ye)) * dt),
                    "iae_ss": float(np.sum(np.abs(ye[ss_mask])) * dt),
                }

            all_results[(ptype, dist_level)] = data_list

    # Restore original disturbance values
    restore_disturbances()

    elapsed = time.time() - t_start
    print(f"\n\n  All {total_runs} simulations completed in {elapsed:.1f} s\n")

    # ── Generate comparison plots ──
    print("  Generating comparison plots...")
    plot_count = 0
    for dist_level in DISTURBANCE_LEVELS:
        for ptype in PATH_TYPES:
            data_list = all_results[(ptype, dist_level)]
            plot_trajectory_comparison(data_list, ptype, dist_level, OUTPUT_DIR)
            plot_crosstrack_comparison(data_list, ptype, dist_level, OUTPUT_DIR)
            plot_control_comparison(data_list, ptype, dist_level, OUTPUT_DIR)
            plot_heading_comparison(data_list, ptype, dist_level, OUTPUT_DIR)
            plot_course_comparison(data_list, ptype, dist_level, OUTPUT_DIR)
            plot_count += 5

    # ── Summary plots ──
    plot_bar_charts(metrics, OUTPUT_DIR)
    plot_overall_ranking(metrics, OUTPUT_DIR)
    plot_count += len(DISTURBANCE_LEVELS) + 1

    print(f"  {plot_count} plots saved to {OUTPUT_DIR}/")

    # ── Results table ──
    table = format_results_table(metrics)
    print("\n" + table)

    with open(RESULTS_FILE, "w") as f:
        f.write(table)
    print(f"\n  Results saved to {RESULTS_FILE}")

    # ── Print plot file list ──
    print(f"\n  Plot files in {OUTPUT_DIR}/:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(".png"):
            fpath = os.path.join(OUTPUT_DIR, fname)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {fname:50s} ({size_kb:.0f} KB)")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
