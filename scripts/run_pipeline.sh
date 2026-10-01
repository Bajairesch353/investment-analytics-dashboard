#!/bin/bash
# Runs the full notebook pipeline (01-12) in dependency order, then
# (optionally) triggers the newsletter. "Light" v1 automation per
# docs/roadmap_v1.md point 4 (section 2) — no retry/backoff, no scheduling.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOKS_DIR="$PROJECT_ROOT/notebooks"
JUPYTER="$PROJECT_ROOT/.venv/bin/jupyter"
STREAMLIT="$PROJECT_ROOT/.venv/bin/streamlit"
TIMEOUT=1800  # seconds/cell; yfinance + Ollama calls can be slow

# Execution order is plain numeric order (01-12), which already satisfies
# every dependency: 02 must precede 03/04/05/07-10/12 (writes
# current_holdings_trackable.csv, actual_vs_target_weights.csv); 04 precedes
# 05/12 (asset_universe_metrics/overview.csv); 07 precedes 10
# (backtest_cumulative_growth.csv) — all already true for 01<02<...<12.
# The one deliberate deviation: 11 (macro) is moved to the very end. Not
# because anything depends on it (nothing does), but because it's the
# slowest/flakiest notebook (Ollama + Fed/EZB scraping) and not yet
# content-approved (LLM stance mode-collapse, see roadmap_v1.md 2.2) — this
# way an early Ctrl+C still leaves all dashboard-relevant data from 01-10+12
# fresh. 06 and 11 both run best-effort: log failure and continue instead of
# aborting, since nothing downstream reads their output.
NOTEBOOKS=(
    "01_transactions_cashflows.ipynb|critical"
    "02_portfolio_holdings_tracker.ipynb|critical"
    "03_portfolio_risk_performance.ipynb|critical"
    "04_asset_universe_watchlist.ipynb|critical"
    "05_asset_selection_scoring.ipynb|critical"
    "06_cash_consumption_analytics.ipynb|best-effort"
    "07_portfolio_backtesting.ipynb|critical"
    "08_portfolio_optimization.ipynb|critical"
    "09_portfolio_stress_testing.ipynb|critical"
    "10_factor_sector_exposure.ipynb|critical"
    "12_fundamental_stock_scoring.ipynb|critical"
    "11_macro_policy_sentiment.ipynb|best-effort"
)

SKIP_MACRO=0
DEMO=0
SKIP_CONSUMPTION=0
SKIP_NEWSLETTER=0
LAUNCH_DASHBOARD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-macro) SKIP_MACRO=1 ;;
        --skip-consumption) SKIP_CONSUMPTION=1 ;;
        --skip-newsletter) SKIP_NEWSLETTER=1 ;;
        --launch-dashboard) LAUNCH_DASHBOARD=1 ;;
        --demo) DEMO=1 ;;
        -h|--help)
            echo "Usage: $0 [--demo] [--skip-macro] [--skip-consumption] [--skip-newsletter] [--launch-dashboard]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# --demo: run on the bundled fictitious sample data (data_sample/) instead of
# data/. Notebooks and dashboard read INVESTMENT_DATA_DIR (see src/paths.py);
# executed notebooks go to data_sample/executed_notebooks/ so your own
# notebook outputs stay untouched. Macro (Ollama) and newsletter are skipped.
EXECUTE_ARGS=(--inplace)
if [ "$DEMO" -eq 1 ]; then
    export INVESTMENT_DATA_DIR="$PROJECT_ROOT/data_sample"
    mkdir -p "$INVESTMENT_DATA_DIR/manual" "$INVESTMENT_DATA_DIR/processed" "$INVESTMENT_DATA_DIR/executed_notebooks"
    EXECUTE_ARGS=(--output-dir "$INVESTMENT_DATA_DIR/executed_notebooks")
    SKIP_MACRO=1
    SKIP_NEWSLETTER=1
    echo ">>> Demo mode: using sample data in $INVESTMENT_DATA_DIR"
fi

# Wait for network before starting: when launchd catches up a missed run
# right after the Mac wakes, Wi-Fi is often not connected yet, and every
# yfinance download silently returns nothing (28.09.2026: NB02 then died
# with an opaque KeyError). Any HTTP response from Yahoo counts as online.
NETWORK_WAIT_SECONDS=600
waited=0
until curl -s -o /dev/null --max-time 5 https://query1.finance.yahoo.com; do
    if [ "$waited" -ge "$NETWORK_WAIT_SECONDS" ]; then
        echo "!!! No network after ${NETWORK_WAIT_SECONDS}s — aborting pipeline." >&2
        exit 1
    fi
    [ "$waited" -eq 0 ] && echo ">>> Waiting for network (max ${NETWORK_WAIT_SECONDS}s)..."
    sleep 15
    waited=$((waited + 15))
done

RESULTS=()      # "name|status|seconds"
PIPELINE_FAILED=0
PIPELINE_START=$(date +%s)

cd "$NOTEBOOKS_DIR" || exit 1

for entry in "${NOTEBOOKS[@]}"; do
    nb_name="${entry%%|*}"
    criticality="${entry##*|}"

    if [ "$nb_name" = "06_cash_consumption_analytics.ipynb" ] && [ "$SKIP_CONSUMPTION" -eq 1 ]; then
        RESULTS+=("$nb_name|SKIPPED|0")
        continue
    fi
    if [ "$nb_name" = "11_macro_policy_sentiment.ipynb" ] && [ "$SKIP_MACRO" -eq 1 ]; then
        RESULTS+=("$nb_name|SKIPPED|0")
        continue
    fi

    echo ">>> Running $nb_name ($criticality)..."
    nb_start=$(date +%s)
    "$JUPYTER" nbconvert --to notebook --execute "${EXECUTE_ARGS[@]}" \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        "$nb_name"
    status=$?
    nb_end=$(date +%s)
    elapsed=$((nb_end - nb_start))

    if [ $status -eq 0 ]; then
        RESULTS+=("$nb_name|OK|$elapsed")
    else
        RESULTS+=("$nb_name|FAIL|$elapsed")
        if [ "$criticality" = "critical" ]; then
            echo "!!! $nb_name failed (critical) — aborting pipeline." >&2
            PIPELINE_FAILED=1
            break
        else
            echo "!!! $nb_name failed (best-effort) — continuing." >&2
        fi
    fi
done

NEWSLETTER_STATUS="SKIPPED"
newsletter_elapsed=0
if [ "$PIPELINE_FAILED" -eq 0 ] && [ "$SKIP_NEWSLETTER" -eq 0 ]; then
    echo ">>> Triggering newsletter..."
    nl_start=$(date +%s)
    "$SCRIPT_DIR/run_newsletter.sh"
    nl_status=$?
    nl_end=$(date +%s)
    newsletter_elapsed=$((nl_end - nl_start))
    if [ $nl_status -eq 0 ]; then
        NEWSLETTER_STATUS="OK"
    else
        NEWSLETTER_STATUS="FAIL"
        echo "!!! Newsletter step failed — not fatal, notebooks already succeeded." >&2
    fi
elif [ "$SKIP_NEWSLETTER" -eq 1 ]; then
    NEWSLETTER_STATUS="SKIPPED (--skip-newsletter)"
elif [ "$PIPELINE_FAILED" -eq 1 ]; then
    NEWSLETTER_STATUS="SKIPPED (pipeline failed)"
fi

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "=== Pipeline Summary ==="
printf '%-42s %-10s %6s\n' "Notebook" "Status" "Time(s)"
for r in "${RESULTS[@]}"; do
    name="${r%%|*}"
    rest="${r#*|}"
    st="${rest%%|*}"
    sec="${rest##*|}"
    printf '%-42s %-10s %6s\n' "$name" "$st" "$sec"
done
printf '%-42s %-10s %6s\n' "newsletter" "$NEWSLETTER_STATUS" "$newsletter_elapsed"
echo "------------------------------------------------------------"
echo "Total elapsed: ${TOTAL_ELAPSED}s"

if [ "$LAUNCH_DASHBOARD" -eq 1 ]; then
    echo ""
    echo ">>> Launching dashboard (Ctrl+C to stop)..."
    "$STREAMLIT" run "$PROJECT_ROOT/app/dashboard.py" --server.address localhost
fi

if [ "$PIPELINE_FAILED" -eq 1 ]; then
    exit 1
fi
exit 0
