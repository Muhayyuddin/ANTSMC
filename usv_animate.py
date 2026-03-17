#!/usr/bin/env python3
"""
USV Animated GIF Generator
============================

Step 1: Run all simulations and save raw data to  data/  as .npz files.
Step 2: Load the saved data and render animated GIFs to  gifs/.

Generates 4 types of animation per (path, sea-state) combination:
  - Trajectory:   all controllers tracing the path simultaneously
  - Cross-track:  y_e over time for all controllers
  - Heading:      ψ vs ψ_d over time
  - Control:      τ_u and τ_r over time

Total: 4 paths × 3 sea states × 4 animation types = 48 GIFs

Usage
-----
    python usv_animate.py              # run sims + generate GIFs
    python usv_animate.py --gif-only   # skip sims, use saved data
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

# ── Project imports ──
from usv_common import CONFIG, run_simulation, get_waypoints_for_path_type
from usv_run_all import (
    CONTROLLERS, PATH_TYPES, DISTURBANCE_LEVELS, DIST_SHORT,
    COLORS, LINE_STYLES, LINE_WIDTHS,
    set_disturbance_scale, restore_disturbances,
)

T_FINAL_MAP = {"custom": 300, "circular": 350, "rectangular": 350, "zigzag": 300}

DATA_DIR = "data"
GIF_DIR = "gifs"

# Animation settings
FPS = 12                # frames per second  (slower playback)
DURATION_S = 8          # target GIF duration [seconds]
TOTAL_FRAMES = FPS * DURATION_S  # 96 frames
TRAIL_FADE = True       # older trail points fade slightly


# =============================================================================
# Step 1 — Run simulations & save data
# =============================================================================

def save_simulation_data():
    """Run all 72 simulations and save the time-history arrays as .npz."""
    os.makedirs(DATA_DIR, exist_ok=True)

    total = len(CONTROLLERS) * len(PATH_TYPES) * len(DISTURBANCE_LEVELS)
    count = 0
    t0 = time.time()

    for dist_level, dist_scale in DISTURBANCE_LEVELS.items():
        set_disturbance_scale(dist_scale)
        for ptype in PATH_TYPES:
            for ctrl_name, ctrl_factory in CONTROLLERS:
                count += 1
                sys.stdout.write(
                    f"\r  [{count:3d}/{total}] {ctrl_name:<12s} | "
                    f"{ptype:<14s} | {dist_level}  "
                )
                sys.stdout.flush()

                ctrl = ctrl_factory()
                cfg = {"path_type": ptype, "t_final": T_FINAL_MAP[ptype]}
                data = run_simulation(ctrl, cfg, ctrl_name)

                fname = os.path.join(
                    DATA_DIR, f"{ctrl_name}_{ptype}_{dist_level}.npz"
                )
                np.savez_compressed(
                    fname,
                    t=data["t"], x=data["x"], y=data["y"],
                    psi=data["psi"], u=data["u"], v=data["v"], r=data["r"],
                    y_e=data["y_e"], chi_e=data["chi_e"],
                    psi_d=data["psi_d"], u_d=data["u_d"],
                    tau_u=data["tau_u"], tau_r=data["tau_r"],
                    F_L=data["F_L"], F_R=data["F_R"],
                    waypoints=data["waypoints"],
                    controller=ctrl_name,
                    path_type=ptype,
                )
        restore_disturbances()

    elapsed = time.time() - t0
    print(f"\n  All {total} simulations saved in {elapsed:.1f} s → {DATA_DIR}/")


def load_sim_data(ctrl_name, ptype, dist_level):
    """Load a saved .npz simulation file and return a dict."""
    fname = os.path.join(DATA_DIR, f"{ctrl_name}_{ptype}_{dist_level}.npz")
    d = np.load(fname, allow_pickle=True)
    return {k: d[k] for k in d.files}


# =============================================================================
# Trim helper (same logic as usv_run_all)
# =============================================================================

def _trim_to_path(x, y, waypoints):
    """Return (x_trim, y_trim, i_start, i_end) — same trim logic as static plots."""
    N = len(x)
    wp_start, wp_end = waypoints[0], waypoints[-1]

    dist2_start = (x - wp_start[0])**2 + (y - wp_start[1])**2
    i_start = int(np.argmin(dist2_start))

    half = max(N // 2, i_start + 1)
    dist2_end = (x[half:] - wp_end[0])**2 + (y[half:] - wp_end[1])**2
    i_end = half + int(np.argmin(dist2_end))

    return x[i_start:i_end+1], y[i_start:i_end+1], i_start, i_end


# =============================================================================
# USV marker helper
# =============================================================================

def _draw_usv_marker(ax, x, y, psi, color, size=1.0):
    """Draw a small triangle representing the USV at (x, y, psi)."""
    L = 4.0 * size  # marker half-length
    W = 2.0 * size  # marker half-width
    cp, sp = np.cos(psi), np.sin(psi)
    # Triangle: bow, port-stern, starboard-stern
    pts = np.array([
        [x + L * cp,          y + L * sp],           # bow
        [x - L * cp - W * sp, y - L * sp + W * cp],  # port
        [x - L * cp + W * sp, y - L * sp - W * cp],  # starboard
    ])
    tri = plt.Polygon(pts, closed=True, fc=color, ec="black", lw=0.5,
                       alpha=0.9, zorder=20)
    ax.add_patch(tri)
    return tri


# =============================================================================
# Step 2 — Animated GIF generators
# =============================================================================

def _frames_to_gif(frames, save_path, fps=FPS):
    """Save a list of PIL Image frames as an animated GIF."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    duration_ms = int(1000 / fps)
    # Build a shared palette from the first frame, then quantize all
    # frames against it — much faster than ADAPTIVE per-frame quantization.
    ref = frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = ref.getpalette()
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(palette)
    p_frames = [f.quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG)
                for f in frames]
    p_frames[0].save(
        save_path,
        save_all=True,
        append_images=p_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _fig_to_pil(fig, dpi=100):
    """Convert a matplotlib figure to a PIL Image (RGB)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))
    return Image.fromarray(buf)


def animate_trajectory(all_data, ptype, dist_level, save_dir):
    """Animated trajectory — all 6 controllers trace the path simultaneously."""
    wps = all_data[0]["waypoints"]
    dl = DIST_SHORT.get(dist_level, dist_level.capitalize())

    # Trim and prepare data for each controller
    ctrl_traces = []
    for d in all_data:
        name = str(d["controller"])
        xt, yt, i_s, i_e = _trim_to_path(d["x"], d["y"], wps)
        # Get psi for the trimmed range
        psi_t = d["psi"][i_s:i_e+1]
        ctrl_traces.append({
            "name": name, "x": xt, "y": yt, "psi": psi_t,
            "n": len(xt),
        })

    # Number of simulation steps → map to animation frames
    max_n = max(c["n"] for c in ctrl_traces)

    # Compute axis limits from all trajectories + waypoints
    all_x = np.concatenate([c["x"] for c in ctrl_traces] + [wps[:, 0]])
    all_y = np.concatenate([c["y"] for c in ctrl_traces] + [wps[:, 1]])
    x_pad = (all_x.max() - all_x.min()) * 0.08 + 5
    y_pad = (all_y.max() - all_y.min()) * 0.08 + 5
    xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
    ylim = (all_y.min() - y_pad, all_y.max() + y_pad)

    frames = []
    fig, ax = plt.subplots(figsize=(10, 7), dpi=80)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.10)

    for fi in range(TOTAL_FRAMES):
        ax.clear()

        # Fraction of animation complete
        frac = (fi + 1) / TOTAL_FRAMES

        # Reference path
        ax.plot(wps[:, 0], wps[:, 1], "ks--", ms=3, lw=1.5,
                label="Reference", zorder=5)
        ax.plot(wps[0, 0], wps[0, 1], "g^", ms=12, zorder=15)

        for c in ctrl_traces:
            name = c["name"]
            # How far along this controller's trace
            idx = min(int(frac * c["n"]), c["n"] - 1)
            if idx < 1:
                continue

            # Draw trail
            ax.plot(c["x"][:idx+1], c["y"][:idx+1],
                    color=COLORS[name], ls=LINE_STYLES[name],
                    lw=LINE_WIDTHS[name], label=name, alpha=0.85)

            # Draw USV marker at current position
            _draw_usv_marker(ax, c["x"][idx], c["y"][idx],
                             c["psi"][idx], COLORS[name], size=1.2)

        ax.set_xlabel("x [m]", fontsize=12)
        ax.set_ylabel("y [m]", fontsize=12)
        ax.set_title(f"Trajectory — {ptype.capitalize()} Path | {dl}",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="upper center", ncol=4, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Time indicator
        t_max = max_n * CONFIG["dt"]
        t_now = frac * t_max
        ax.text(0.98, 0.02, f"t = {t_now:.1f} s",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        frames.append(_fig_to_pil(fig))

    plt.close(fig)

    save_path = os.path.join(save_dir, f"traj_{ptype}_{dist_level}.gif")
    _frames_to_gif(frames, save_path)
    return save_path


def animate_crosstrack(all_data, ptype, dist_level, save_dir):
    """Animated cross-track error time series."""
    dl = DIST_SHORT.get(dist_level, dist_level.capitalize())
    ctrl_names = [str(d["controller"]) for d in all_data]

    # Pre-compute y-axis limits
    all_ye = np.concatenate([d["y_e"] for d in all_data])
    y_max = max(abs(all_ye.min()), abs(all_ye.max())) * 1.15
    t_max = max(d["t"][-1] for d in all_data)

    frames = []
    fig, ax = plt.subplots(figsize=(11, 5), dpi=80)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.13)

    for fi in range(TOTAL_FRAMES):
        ax.clear()
        frac = (fi + 1) / TOTAL_FRAMES

        for d in all_data:
            name = str(d["controller"])
            t_arr = d["t"]
            ye = d["y_e"]
            N = len(t_arr)
            idx = min(int(frac * N), N - 1)
            if idx < 1:
                continue
            ax.plot(t_arr[:idx+1], ye[:idx+1],
                    color=COLORS[name], ls=LINE_STYLES[name],
                    lw=LINE_WIDTHS[name], label=name, alpha=0.85)
            # Dot at current position
            ax.plot(t_arr[idx], ye[idx], "o", color=COLORS[name],
                    ms=5, zorder=10)

        ax.axhline(0, color="k", ls=":", lw=0.5)
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Cross-Track Error $y_e$ [m]", fontsize=12)
        ax.set_title(f"Cross-Track Error — {ptype.capitalize()} Path | {dl}",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(0, t_max * 1.02)
        ax.set_ylim(-y_max, y_max)
        ax.legend(fontsize=8, loc="upper center", ncol=6, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        t_now = frac * t_max
        ax.text(0.98, 0.02, f"t = {t_now:.1f} s",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        frames.append(_fig_to_pil(fig))

    plt.close(fig)

    save_path = os.path.join(save_dir, f"ye_{ptype}_{dist_level}.gif")
    _frames_to_gif(frames, save_path)
    return save_path


def animate_heading(all_data, ptype, dist_level, save_dir):
    """Animated heading tracking (ψ vs ψ_d) and course error."""
    dl = DIST_SHORT.get(dist_level, dist_level.capitalize())

    # Pre-compute limits
    all_psi = np.concatenate([np.degrees(d["psi"]) for d in all_data]
                              + [np.degrees(d["psi_d"]) for d in all_data])
    psi_min, psi_max = all_psi.min(), all_psi.max()
    psi_pad = (psi_max - psi_min) * 0.1 + 5
    all_chi = np.concatenate([np.degrees(d["chi_e"]) for d in all_data])
    chi_max = max(abs(all_chi.min()), abs(all_chi.max())) * 1.15
    t_max = max(d["t"][-1] for d in all_data)

    frames = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=80, sharex=True)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.10, hspace=0.28)

    for fi in range(TOTAL_FRAMES):
        ax1.clear()
        ax2.clear()
        frac = (fi + 1) / TOTAL_FRAMES

        for d in all_data:
            name = str(d["controller"])
            t_arr = d["t"]
            N = len(t_arr)
            idx = min(int(frac * N), N - 1)
            if idx < 1:
                continue
            # Each controller's own ψ_d (thin dotted, same colour)
            ax1.plot(t_arr[:idx+1], np.degrees(d["psi_d"][:idx+1]),
                     color=COLORS[name], ls=":", lw=0.7, alpha=0.45)
            ax1.plot(t_arr[:idx+1], np.degrees(d["psi"][:idx+1]),
                     color=COLORS[name], ls=LINE_STYLES[name],
                     lw=LINE_WIDTHS[name], label=name, alpha=0.8)
            ax2.plot(t_arr[:idx+1], np.degrees(d["chi_e"][:idx+1]),
                     color=COLORS[name], ls=LINE_STYLES[name],
                     lw=LINE_WIDTHS[name], label=name, alpha=0.8)

        # Dummy legend entry for ψ_d
        ax1.plot([], [], "k:", lw=0.7, alpha=0.45, label="$\\psi_d$")

        ax1.set_ylabel("Heading ψ [deg]", fontsize=12)
        ax1.set_title(f"Heading Tracking — {ptype.capitalize()} Path | {dl}",
                      fontsize=13, fontweight="bold")
        ax1.set_xlim(0, t_max * 1.02)
        ax1.set_ylim(psi_min - psi_pad, psi_max + psi_pad)
        ax1.legend(fontsize=7, loc="upper center", ncol=7, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        ax2.axhline(0, color="k", ls=":", lw=0.5)
        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("Course Error χ_e [deg]", fontsize=12)
        ax2.set_xlim(0, t_max * 1.02)
        ax2.set_ylim(-chi_max, chi_max)
        ax2.legend(fontsize=7, loc="upper center", ncol=6, framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        t_now = frac * t_max
        ax2.text(0.98, 0.02, f"t = {t_now:.1f} s",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        frames.append(_fig_to_pil(fig))

    plt.close(fig)

    save_path = os.path.join(save_dir, f"heading_{ptype}_{dist_level}.gif")
    _frames_to_gif(frames, save_path)
    return save_path


def animate_control(all_data, ptype, dist_level, save_dir):
    """Animated control effort (τ_u, τ_r) time series."""
    dl = DIST_SHORT.get(dist_level, dist_level.capitalize())

    # Pre-compute limits
    all_tau_u = np.concatenate([d["tau_u"] for d in all_data])
    all_tau_r = np.concatenate([d["tau_r"] for d in all_data])
    tau_u_lim = max(abs(all_tau_u.min()), abs(all_tau_u.max())) * 1.15
    tau_r_lim = max(abs(all_tau_r.min()), abs(all_tau_r.max())) * 1.15
    t_max = max(d["t"][-1] for d in all_data)

    frames = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=80, sharex=True)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.10, hspace=0.28)

    for fi in range(TOTAL_FRAMES):
        ax1.clear()
        ax2.clear()
        frac = (fi + 1) / TOTAL_FRAMES

        for d in all_data:
            name = str(d["controller"])
            t_arr = d["t"]
            N = len(t_arr)
            idx = min(int(frac * N), N - 1)
            if idx < 1:
                continue
            ax1.plot(t_arr[:idx+1], d["tau_u"][:idx+1],
                     color=COLORS[name], ls=LINE_STYLES[name],
                     lw=LINE_WIDTHS[name], label=name, alpha=0.8)
            ax2.plot(t_arr[:idx+1], d["tau_r"][:idx+1],
                     color=COLORS[name], ls=LINE_STYLES[name],
                     lw=LINE_WIDTHS[name], label=name, alpha=0.8)

        ax1.set_ylabel("$\\tau_u$ [N]", fontsize=12)
        ax1.set_title(f"Control Effort — {ptype.capitalize()} Path | {dl}",
                      fontsize=13, fontweight="bold")
        ax1.set_xlim(0, t_max * 1.02)
        ax1.set_ylim(-tau_u_lim, tau_u_lim)
        ax1.legend(fontsize=7, loc="upper center", ncol=6, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Time [s]", fontsize=12)
        ax2.set_ylabel("$\\tau_r$ [N·m]", fontsize=12)
        ax2.set_xlim(0, t_max * 1.02)
        ax2.set_ylim(-tau_r_lim, tau_r_lim)
        ax2.legend(fontsize=7, loc="upper center", ncol=6, framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        t_now = frac * t_max
        ax2.text(0.98, 0.02, f"t = {t_now:.1f} s",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        frames.append(_fig_to_pil(fig))

    plt.close(fig)

    save_path = os.path.join(save_dir, f"ctrl_{ptype}_{dist_level}.gif")
    _frames_to_gif(frames, save_path)
    return save_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="USV Animated GIF Generator")
    parser.add_argument("--gif-only", action="store_true",
                        help="Skip simulations, use saved data in data/")
    args = parser.parse_args()

    ctrl_names = [c[0] for c in CONTROLLERS]

    # ── Step 1: Run and save simulation data ──
    if not args.gif_only:
        print("=" * 60)
        print("  Step 1: Running simulations and saving data")
        print("=" * 60)
        save_simulation_data()
    else:
        print("  Skipping simulations (--gif-only), using saved data/")

    # ── Step 2: Generate animated GIFs ──
    print("\n" + "=" * 60)
    print("  Step 2: Generating animated GIFs")
    print(f"  {TOTAL_FRAMES} frames @ {FPS} fps = {DURATION_S} s per GIF")
    print("=" * 60)

    os.makedirs(GIF_DIR, exist_ok=True)

    gif_count = 0
    t0 = time.time()

    for dist_level in DISTURBANCE_LEVELS:
        for ptype in PATH_TYPES:
            # Load all 5 controllers for this (path, sea-state)
            all_data = []
            for cname in ctrl_names:
                d = load_sim_data(cname, ptype, dist_level)
                all_data.append(d)

            combo = f"{ptype}/{dist_level}"
            print(f"\n  ── {combo} ──")

            # Trajectory
            print(f"    Trajectory ...", end="", flush=True)
            f = animate_trajectory(all_data, ptype, dist_level, GIF_DIR)
            size_kb = os.path.getsize(f) / 1024
            print(f" ✓ ({size_kb:.0f} KB)")
            gif_count += 1

            # Cross-track
            print(f"    Cross-track ...", end="", flush=True)
            f = animate_crosstrack(all_data, ptype, dist_level, GIF_DIR)
            size_kb = os.path.getsize(f) / 1024
            print(f" ✓ ({size_kb:.0f} KB)")
            gif_count += 1

            # Heading
            print(f"    Heading ...", end="", flush=True)
            f = animate_heading(all_data, ptype, dist_level, GIF_DIR)
            size_kb = os.path.getsize(f) / 1024
            print(f" ✓ ({size_kb:.0f} KB)")
            gif_count += 1

            # Control
            print(f"    Control ...", end="", flush=True)
            f = animate_control(all_data, ptype, dist_level, GIF_DIR)
            size_kb = os.path.getsize(f) / 1024
            print(f" ✓ ({size_kb:.0f} KB)")
            gif_count += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  {gif_count} GIFs saved to {GIF_DIR}/ in {elapsed:.1f} s")
    print(f"{'=' * 60}")

    # List all GIFs with sizes
    print(f"\n  GIF files:")
    total_mb = 0
    for fname in sorted(os.listdir(GIF_DIR)):
        if fname.endswith(".gif"):
            fpath = os.path.join(GIF_DIR, fname)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            total_mb += size_mb
            print(f"    {fname:45s} ({size_mb:.1f} MB)")
    print(f"\n  Total: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
