#!/usr/bin/env bash
# =============================================================================
#  run_montecarlo.sh
#  ------------------
#  Run the Monte Carlo robustness study:
#    30 trials × 6 controllers × 2 paths × 3 sea states = 1,080 simulations
#
#  Outputs (per sea state SS1/SS2/SS3):
#    plots/mc_boxplot_ssX.png        — Box-plot of RMS cross-track error
#    plots/mc_bar_meanstd_ssX.png    — Bar chart of mean ± std
#    plots/mc_convergence_ssX.png    — Running-mean convergence
#    plots/mc_scatter_energy_ssX.png — Accuracy vs energy scatter
#    (+ EPS versions in plots-eps/)
#
#  Usage:
#    chmod +x run_montecarlo.sh
#    ./run_montecarlo.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Colours for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}${BOLD}============================================================${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}============================================================${NC}"
}

elapsed() {
    local t=$1
    printf '%dm %ds' $((t/60)) $((t%60))
}

banner "Monte Carlo Robustness Study (1,080 simulations)"
echo ""
echo "  Configuration:"
echo "    Controllers:   LQR, SMC, ASMC, ADRC, FNTSMC, ANTSMC"
echo "    Paths:         custom, rectangular"
echo "    Sea states:    SS1 (calm), SS2 (moderate), SS3 (slight)"
echo "    Trials:        30 per combination"
echo "    Variations:    ±20% mass, ±30% damping, ±25% disturbance"
echo "    Total sims:    1,080"
echo ""

STEP_START=$SECONDS
"$PYTHON" "${SCRIPT_DIR}/usv_monte_carlo.py"
ELAPSED=$((SECONDS - STEP_START))

banner "Monte Carlo Complete!"
echo -e "  Total time: ${BOLD}$(elapsed $ELAPSED)${NC}"
echo ""
echo "  Outputs:"
echo "    plots/mc_boxplot_ss1.png .. ss3.png"
echo "    plots/mc_bar_meanstd_ss1.png .. ss3.png"
echo "    plots/mc_convergence_ss1.png .. ss3.png"
echo "    plots/mc_scatter_energy_ss1.png .. ss3.png"
echo "    plots-eps/  (EPS versions)"
echo ""
echo "  Results summary printed above."
echo ""
