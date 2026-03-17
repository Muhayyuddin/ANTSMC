#!/usr/bin/env bash
# =============================================================================
#  run_analysis.sh
#  ---------------
#  Run the full USV deterministic analysis pipeline:
#    1. 72-simulation benchmark  (6 controllers × 4 paths × 3 sea states)
#    2. JONSWAP stochastic wave validation
#    3. Animated GIF generation  (48 GIFs)
#    4. LaTeX manuscript compilation
#
#  Usage:
#    chmod +x run_analysis.sh
#    ./run_analysis.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
DOC_DIR="${SCRIPT_DIR}/doc"

# Colours for terminal output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Colour

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

TOTAL_START=$SECONDS

# ── 1. Deterministic Benchmark (72 simulations) ─────────────────────────────
banner "Step 1/4 — Deterministic Benchmark (72 simulations)"
STEP_START=$SECONDS
"$PYTHON" "${SCRIPT_DIR}/usv_run_all.py"
echo -e "${GREEN}  ✓ Benchmark completed in $(elapsed $((SECONDS - STEP_START)))${NC}"

# ── 2. JONSWAP Stochastic Validation ────────────────────────────────────────
banner "Step 2/4 — JONSWAP Stochastic Wave Validation"
STEP_START=$SECONDS
"$PYTHON" "${SCRIPT_DIR}/usv_jonswap_validation.py"
echo -e "${GREEN}  ✓ JONSWAP validation completed in $(elapsed $((SECONDS - STEP_START)))${NC}"

# ── 3. Animated GIFs (48 GIFs) ──────────────────────────────────────────────
banner "Step 3/4 — Animated GIF Generation (48 GIFs)"
STEP_START=$SECONDS
"$PYTHON" "${SCRIPT_DIR}/usv_animate.py"
echo -e "${GREEN}  ✓ GIF generation completed in $(elapsed $((SECONDS - STEP_START)))${NC}"

# ── 4. Compile LaTeX Manuscript ──────────────────────────────────────────────
banner "Step 4/4 — Compiling LaTeX Manuscript"
STEP_START=$SECONDS
if command -v pdflatex &> /dev/null; then
    cd "$DOC_DIR"
    pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode manuscript.tex > /dev/null 2>&1 || true
    echo -e "${GREEN}  ✓ Manuscript compiled → doc/manuscript.pdf${NC}"
    cd "$SCRIPT_DIR"
else
    echo "  ⚠  pdflatex not found — skipping manuscript compilation"
fi
echo -e "  Completed in $(elapsed $((SECONDS - STEP_START)))"

# ── Summary ──────────────────────────────────────────────────────────────────
banner "All Done!"
echo -e "  Total time: ${BOLD}$(elapsed $((SECONDS - TOTAL_START)))${NC}"
echo ""
echo "  Outputs:"
echo "    plots/          — 65+ benchmark PNG plots"
echo "    plots-eps/      — EPS versions for LaTeX"
echo "    gifs/           — 48 animated GIFs"
echo "    doc/manuscript.pdf"
echo ""
