#!/usr/bin/env python3
"""
ANTSMC block diagram  ---  clean, minimal, publication-ready.

Top-to-bottom signal flow with generous spacing.
Three novel mechanisms highlighted with red borders.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# ── colours ──
BG_GREEN  = "#C8E6C9"
BG_ORANGE = "#FFE0B2"
BG_BLUE   = "#BBDEFB"
BG_PURPLE = "#D1C4E9"
BG_PINK   = "#F8BBD0"
BG_LIME   = "#DCEDC8"
BG_YELLOW = "#FFF9C4"
BG_SALMON = "#FFCCBC"
BG_GREY   = "#CFD8DC"
DARK      = "#263238"
ARROW_CLR = "#37474F"
RED       = "#C62828"
TXT       = "#212121"
SUBTXT    = "#455A64"


# ── helpers ──
def box(ax, cx, cy, w, h, label, bg,
        sub=None, fs=16, fs2=13, novel=False, bold=False):
    ec, lw = (RED, 2.6) if novel else (DARK, 1.4)
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.12",
                       fc=bg, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    fw = "bold" if bold or novel else "normal"
    if sub:
        ax.text(cx, cy + 0.35, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=TXT, zorder=4)
        ax.text(cx, cy - 0.3, sub, ha="center", va="center",
                fontsize=fs2, color=SUBTXT, style="italic", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=TXT, zorder=4)


def arr(ax, x1, y1, x2, y2, label="", pos="above",
        col=None, lw=1.6, fs=14, rad=0):
    c = col or ARROW_CLR
    cs = "arc3,rad={}".format(rad)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw,
                                connectionstyle=cs,
                                shrinkA=5, shrinkB=5),
                zorder=2)
    if label:
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if pos == "above":  my += 0.28
        elif pos == "below": my -= 0.28
        elif pos == "right": mx += 0.5
        elif pos == "left":  mx -= 0.5
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=fs, color=TXT, zorder=7)


def lin(ax, pts, col=None, lw=1.5):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col or ARROW_CLR, lw=lw, zorder=2,
            solid_capstyle="round")


def sumn(ax, cx, cy, r=0.32):
    ax.add_patch(Circle((cx, cy), r, fc="#ECEFF1", ec=DARK,
                         lw=1.4, zorder=5))
    ax.text(cx, cy, "+", ha="center", va="center",
            fontsize=15, fontweight="bold", color=TXT, zorder=6)


# ════════════════════════════════════════════════════════════════════
def main():
    fig, ax = plt.subplots(figsize=(22, 26))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 26)
    ax.axis("off")

    # ── Y rows (top to bottom, generous gaps) ──
    Y_title = 25.0
    Y_los   = 23.2
    Y_err   = 20.8
    Y_surf  = 17.8
    Y_reach = 14.2
    Y_sum   = 11.2
    Y_alloc =  9.0
    Y_plant =  6.5

    # ── X columns ──
    XL = 5.5     # surge
    XR = 13.0    # yaw
    XM = 9.2     # centre

    # ────────────────────────────────────────────────────────────
    #  TITLE  +  LEGEND
    # ────────────────────────────────────────────────────────────
    ax.text(10, Y_title, "ANTSMC  Control  Architecture",
            ha="center", va="center", fontsize=24,
            fontweight="bold", color=TXT)

    ax.plot([14.5, 15.5], [Y_title - 0.55, Y_title - 0.55],
            color=RED, lw=2.6)
    ax.text(15.7, Y_title - 0.55, "Novel ANTSMC mechanisms",
            fontsize=13, color=RED, fontweight="bold", va="center")
    ax.plot([14.5, 15.5], [Y_title - 1.05, Y_title - 1.05],
            color=DARK, lw=1.4)
    ax.text(15.7, Y_title - 1.05, "Standard / shared elements",
            fontsize=13, color=DARK, va="center")

    # ────────────────────────────────────────────────────────────
    #  ROW  LOS GUIDANCE
    # ────────────────────────────────────────────────────────────
    box(ax, XM, Y_los, 7, 1.3, "LOS Guidance", BG_GREEN,
        sub=r"Waypoints  $\rightarrow$  "
            r"$\psi_d,\; u_d,\; y_e,\; \gamma$",
        fs=17, fs2=14)

    # ────────────────────────────────────────────────────────────
    #  ROW  ERROR COMPUTATION
    # ────────────────────────────────────────────────────────────
    box(ax, XM, Y_err, 7.5, 1.3, "Error Computation", BG_ORANGE,
        sub=r"$e_u = \hat{u} - u_d$      "
            r"$\chi_e = \mathrm{wrap}(\hat{\psi} - \gamma)$",
        fs=17, fs2=14)
    arr(ax, XM, Y_los - 0.7, XM, Y_err + 0.7)

    # ────────────────────────────────────────────────────────────
    #  DASHED BOX  —  ANTSMC controller region
    # ────────────────────────────────────────────────────────────
    cb_l, cb_b = 2.3, Y_reach - 2.4
    cb_w = 18.2
    cb_h = (Y_surf + 1.3) - cb_b
    dbox = FancyBboxPatch((cb_l, cb_b), cb_w, cb_h,
                          boxstyle="round,pad=0.3",
                          fc="none", ec=RED, lw=2.2,
                          ls=(0, (7, 3)), alpha=0.45, zorder=1)
    ax.add_patch(dbox)
    ax.text(cb_l + 0.4, cb_b + cb_h - 0.25,
            "ANTSMC  Controller", fontsize=14,
            fontweight="bold", color=RED, style="italic",
            va="top", zorder=7)

    # channel labels
    ax.text(XL, Y_surf + 1.05, "Surge channel",
            ha="center", fontsize=15, color=SUBTXT, style="italic")
    ax.text(XR, Y_surf + 1.05, "Yaw channel  (novel)",
            ha="center", fontsize=15, color=RED,
            fontweight="bold", style="italic")

    # ────────────────────────────────────────────────────────────
    #  ROW  SLIDING SURFACES
    # ────────────────────────────────────────────────────────────
    # surge surface (standard)
    box(ax, XL, Y_surf, 5.5, 1.4, "Surge Surface", BG_BLUE,
        sub=r"$s_u = c_u \; e_u$", fs=16, fs2=14)

    # yaw terminal surface (NOVEL)
    box(ax, XR, Y_surf, 7.0, 1.8, "Terminal Sliding Surface", BG_BLUE,
        sub=r"$s_r = c_1 \chi_e + c_2 r"
            r" + c_3 \varphi(y_e)"
            r" + k_i \!\!\int\! y_e \, dt$",
        novel=True, fs=16, fs2=13)

    # φ annotation below yaw surface
    ax.text(XR, Y_surf - 1.3,
            r"$\varphi(y_e)"
            r" = |y_e|^{0.6}\,\mathrm{sgn}(y_e)$"
            r"  for $|y_e| \leq 1$;"
            r"   linear for $|y_e| > 1$",
            ha="center", va="center", fontsize=13,
            color=RED, style="italic", zorder=7)

    # error → surfaces
    arr(ax, XM - 3, Y_err - 0.7, XL, Y_surf + 0.75,
        label=r"$e_u$", pos="left", fs=14)
    arr(ax, XM + 2.5, Y_err - 0.7, XR, Y_surf + 0.95,
        label=r"$\chi_e,\; y_e,\; r$", pos="right", fs=14)

    # ────────────────────────────────────────────────────────────
    #  ROW  REACHING LAWS  +  ADAPTIVE GAIN
    # ────────────────────────────────────────────────────────────
    # surge reaching (standard)
    box(ax, XL, Y_reach, 5.5, 1.4, "Linear Reaching Law", BG_PURPLE,
        sub=r"$\lambda_u s_u"
            r" + k_{s,u}\,\mathrm{sat}(s_u / \varepsilon_u)$",
        fs=16, fs2=13)

    # power-rate reaching (NOVEL)
    box(ax, XR, Y_reach, 7.0, 1.4,
        "Power-Rate Reaching Law", BG_PURPLE,
        sub=r"$\lambda_r |s_r|^p \mathrm{sgn}(s_r)"
            r" + k_{s,r}(t)\,\mathrm{sat}(s_r / \varepsilon_r)$",
        novel=True, fs=16, fs2=13)

    # surface → reaching
    arr(ax, XL, Y_surf - 0.75, XL, Y_reach + 0.75,
        label=r"$s_u$", fs=14)
    arr(ax, XR, Y_surf - 0.95, XR, Y_reach + 0.75,
        label=r"$s_r$", fs=14)

    # ── ADAPTIVE GAIN (right margin, between surface & reaching) ──
    AX = 19.3
    AY = 0.5 * (Y_surf + Y_reach)   # midpoint
    box(ax, AX, AY, 3.2, 2.6,
        "Adaptive\nSwitching\nGain", BG_PINK,
        novel=True, fs=14)

    # sub-equations below adaptive gain
    ax.text(AX, AY - 1.65,
            r"$\dot{k}_a = \mu|s_r|"
            r" - \sigma_l^{\mathrm{eff}} k_a$",
            ha="center", va="center", fontsize=12,
            color=RED, fontweight="bold", zorder=7)
    ax.text(AX, AY - 2.1,
            r"$\sigma_l^{\mathrm{eff}}"
            r" = \sigma_l(1 + 0.2\,\min(|y_e|,20))$",
            ha="center", va="center", fontsize=12,
            color=RED, style="italic", zorder=7)

    # s_r → adaptive gain  (right from surface)
    arr(ax, XR + 3.5, Y_surf - 0.2, AX - 1.6, AY + 0.8,
        label=r"$|s_r|$", pos="above", col=RED, lw=1.5, fs=13)

    # adaptive gain → power-rate  (k_sr output)
    arr(ax, AX - 1.6, AY - 0.8, XR + 3.5, Y_reach + 0.2,
        label=r"$k_{s,r}(t)$", pos="above", col=RED, lw=2.0, fs=13)

    # ────────────────────────────────────────────────────────────
    #  ROW  SUMMATION  (control law)
    # ────────────────────────────────────────────────────────────
    sumn(ax, XL, Y_sum)
    sumn(ax, XR, Y_sum)

    arr(ax, XL, Y_reach - 0.75, XL, Y_sum + 0.38)
    arr(ax, XR, Y_reach - 0.75, XR, Y_sum + 0.38)

    # drift cancellation
    ax.text(XL - 2.8, Y_sum, r"$\frac{a_u}{b_u}\,e_u$",
            ha="center", va="center", fontsize=15,
            color=SUBTXT, zorder=7)
    arr(ax, XL - 1.8, Y_sum, XL - 0.38, Y_sum, lw=1.2)

    ax.text(XR - 2.8, Y_sum, r"$\frac{a_r}{b_r}\,r$",
            ha="center", va="center", fontsize=15,
            color=SUBTXT, zorder=7)
    arr(ax, XR - 1.8, Y_sum, XR - 0.38, Y_sum, lw=1.2)

    # gain inversion labels
    ax.text(XL, Y_sum - 0.6, r"$\times\,(-1/b_u)$",
            ha="center", va="center", fontsize=13,
            color=SUBTXT, zorder=7)
    ax.text(XR, Y_sum - 0.6, r"$\times\,(-1/b_r)$",
            ha="center", va="center", fontsize=13,
            color=SUBTXT, zorder=7)

    # ────────────────────────────────────────────────────────────
    #  ROW  THRUST ALLOCATION
    # ────────────────────────────────────────────────────────────
    box(ax, XM, Y_alloc, 6.5, 1.2, "Thrust Allocation", BG_LIME,
        sub=r"$F_L,\; F_R$  with surge priority",
        fs=17, fs2=14)
    arr(ax, XL, Y_sum - 0.38, XM - 1.6, Y_alloc + 0.65,
        label=r"$\tau_u$", pos="left", fs=15)
    arr(ax, XR, Y_sum - 0.38, XM + 1.6, Y_alloc + 0.65,
        label=r"$\tau_r$", pos="right", fs=15)

    # ────────────────────────────────────────────────────────────
    #  ROW  USV PLANT
    # ────────────────────────────────────────────────────────────
    box(ax, XM, Y_plant, 6.5, 1.3,
        "USV Plant  (3-DOF)", BG_YELLOW,
        sub="surge  +  sway  +  yaw",
        bold=True, fs=18, fs2=15)
    arr(ax, XM, Y_alloc - 0.65, XM, Y_plant + 0.7)

    # disturbances
    box(ax, 16, Y_plant, 3.2, 1.0, "Disturbances", BG_SALMON,
        sub=r"$\delta_u,\;\delta_v,\;\delta_r$", fs=14, fs2=13)
    arr(ax, 14.4, Y_plant, XM + 3.3, Y_plant,
        col="#D84315", lw=1.6)

    # sensor noise
    box(ax, 3.2, Y_plant, 2.8, 1.0, "Sensor Noise", BG_GREY,
        sub=r"$n_x,\;n_\psi,\;n_u,\;n_r$", fs=14, fs2=12)

    # ────────────────────────────────────────────────────────────
    #  FEEDBACK  —  one clean L-path on the left
    # ────────────────────────────────────────────────────────────
    fb = 1.2   # x of vertical bus

    # plant left → bus → up
    lin(ax, [(XM - 3.3, Y_plant),
             (fb, Y_plant),
             (fb, Y_los)])

    # noise injection
    arr(ax, 3.2, Y_plant - 0.55, fb + 0.5, Y_plant - 0.05,
        col=ARROW_CLR, lw=0.9, label="+", pos="above", fs=12)

    # label on vertical bus
    ax.text(fb - 0.1, 0.5 * (Y_plant + Y_los),
            r"$\hat{x},\;\hat{y},\;\hat{\psi},"
            r"\;\hat{u},\;\hat{r}$",
            ha="center", va="center", fontsize=14, color=SUBTXT,
            rotation=90, zorder=7)

    # bus → LOS
    arr(ax, fb, Y_los, XM - 3.55, Y_los, lw=1.5)

    # bus → Error
    arr(ax, fb, Y_err, XM - 3.8, Y_err, lw=1.5)

    # bus → |y_e| for adaptive leakage
    leaky = AY - 2.4
    lin(ax, [(fb, leaky), (AX, leaky)], col=RED, lw=1.0)
    arr(ax, AX, leaky, AX, AY - 1.35,
        col=RED, lw=1.0, label=r"$|y_e|$", pos="left", fs=13)

    # ────────────────────────────────────────────────────────────
    #  SAVE
    # ────────────────────────────────────────────────────────────
    png = "/home/muhayy/LQR/plots/antsmc_block_diagram.png"
    pdf = "/home/muhayy/LQR/plots/antsmc_block_diagram.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved  " + png)
    print("Saved  " + pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
