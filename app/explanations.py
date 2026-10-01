"""Short "how is this calculated" texts for the dashboard.

Built from the constants in src/, so they always match the definitions the
notebooks actually use -- change a weight or cutoff there and the text here
follows. Parameters that only exist in a notebook are referred to by their
constant name instead of a number. No numbers are hard-coded in this module.
"""

from src import fundamentals, portfolio, selection
from src.macro_sentiment import STANCE_SCORE_MAP
from src.risk import RISK_FREE_RATE


def _pct(value):
    return f"{value * 100:.0f}%"


def _lines(*lines):
    """One line per item in a Streamlit caption (markdown hard line breaks)."""
    return "  \n".join(lines)


# --- Asset selection (Notebook 05) -------------------------------------------

def selection_score_explanation():
    return _lines(
        f"**Selection Score** = {_pct(selection.QUANT_SCORE_WEIGHT)} Quant Score "
        f"+ {_pct(selection.ROLE_SCORE_WEIGHT)} Role Score",
        f"**Quant Score** = {_pct(selection.SHARPE_SCORE_WEIGHT)} Sharpe "
        f"+ {_pct(selection.DRAWDOWN_SCORE_WEIGHT)} Max Drawdown "
        f"+ {_pct(selection.VOLATILITY_SCORE_WEIGHT)} Volatility — each a percentile rank "
        f"across the universe (0 = worst, 1 = best); Sharpe with risk-free rate {RISK_FREE_RATE:.2%}",
        f"**Role Score** (0–1) = strategic importance of the asset's role, set per asset in "
        f"`asset_universe_manual.csv` (default {selection.DEFAULT_ROLE_SCORE:.2f})",
        "Relative to the assets in your universe, not to the market (Notebook 05).",
    )


def decision_explanation():
    return _lines(
        f"*Watch – limited data*: fewer than {selection.MIN_RETURN_OBSERVATIONS} daily returns",
        f"Holdings: *Keep – strategic role* if Role Score ≥ {selection.PROTECTED_ROLE_SCORE_CUTOFF:.2f} · "
        f"*Review* if weight ≥ {_pct(selection.MATERIAL_WEIGHT)} and Selection Score "
        f"≤ {selection.REVIEW_SCORE_CUTOFF:.2f} · otherwise *Keep*",
        f"Watchlist: *Candidate* if Selection Score ≥ {selection.CANDIDATE_SCORE_CUTOFF:.2f} · "
        f"otherwise *Watch*",
    )


def fundamental_score_explanation():
    return _lines(
        f"**Fundamental Score** = {_pct(fundamentals.VALUATION_SCORE_WEIGHT)} Valuation "
        f"+ {_pct(fundamentals.QUALITY_SCORE_WEIGHT)} Quality "
        f"+ {_pct(fundamentals.GROWTH_SCORE_WEIGHT)} Growth "
        f"+ {_pct(fundamentals.HEALTH_SCORE_WEIGHT)} Health",
        f"Valuation: P/E, P/B (cheaper = better) · Quality: ROE, operating margin · "
        f"Growth: {fundamentals.GROWTH_YEARS}-year CAGR of revenue and net income · "
        f"Health: debt/equity, current ratio",
        "Each metric is a percentile rank among the stocks; missing metrics are skipped and "
        "the weights re-normalised",
        f"**Rating:** Strong ≥ {fundamentals.RATING_STRONG_CUTOFF:.2f} · "
        f"Weak < {fundamentals.RATING_WEAK_CUTOFF:.2f} · Watch – limited data if fewer than "
        f"{fundamentals.MIN_REQUIRED_FIELDS} of 8 metrics are available (Notebook 12)",
    )


# --- Portfolio (Notebooks 01/02) ---------------------------------------------

def drift_explanation():
    t = portfolio.DEFAULT_DRIFT_THRESHOLD_PP
    return _lines(
        "Drift = current − target weight (percentage points)",
        f"Action: *Add / buy more* below −{t:g} pp · *Reduce / no new buys* above +{t:g} pp · "
        f"otherwise within target range",
    )


def dca_explanation():
    return _lines(
        f"**Likely DCA** = at least {portfolio.DCA_MIN_ORDERS} orders, "
        f"{portfolio.DCA_MIN_ACTIVE_MONTHS} active months and {portfolio.DCA_MIN_TOTAL_MONTHS} "
        f"months since the first buy",
        f"and Regularity (active months / months since first buy) "
        f"≥ {_pct(portfolio.DCA_MIN_REGULARITY_RATE)}",
    )


CASHFLOWS = _lines(
    "Per month (Notebook 01):",
    "**Net Cash Flow** = sum of all cash movements on the account",
    "**Consumption** = card and direct-debit spending",
    "**Cash Inflow** = incoming transfers and deposits",
    "**Investment Volume** = security purchases",
)



# --- Risk & performance (Notebooks 03/07/10) ---------------------------------

RETURN_PA = "Geometric annualized return of the daily returns (252 trading days)."
VOLATILITY_PA = "Standard deviation of daily returns × √252."
MAX_DRAWDOWN = "Largest fall from a previous high to a subsequent low."


def sharpe_help():
    return f"(Return p.a. − risk-free rate {RISK_FREE_RATE:.2%}) / Volatility p.a."


BETA = "Cov(portfolio, benchmark) / Var(benchmark) on weekly returns."
R_SQUARED = "Share of the portfolio's weekly return variance explained by the benchmark (correlation²)."
HHI = "Σ weight² — 1/N for N equal positions, 1 for a single position."
EFFECTIVE_N = "1 / HHI: number of equal-sized positions with the same concentration."
TOP_N = "Combined weight of the largest positions."
RISK_CONTRIBUTION = (
    "Risk contribution = weight × marginal contribution to portfolio volatility, "
    "normalised to 100% (covariance of daily returns, Notebook 03)."
)


def strategy_metrics_explanation():
    return _lines(
        f"**Return p.a.** geometric · **Sharpe** = (Return − {RISK_FREE_RATE:.2%}) / Volatility",
        f"**Sortino** = (Return − {RISK_FREE_RATE:.2%}) / downside deviation (shortfall below 0 over all days)",
        "**Calmar** = Return p.a. / |Max Drawdown| · **Recovery** = calendar days from the "
        "maximum-drawdown trough back to the old high",
        "*Net* lines deduct `TRANSACTION_COST_BPS` (Notebook 07) per unit of turnover",
    )


REBALANCING_FREQUENCY = _lines(
    "**Turnover** per rebalance = ½ Σ |drifted weight − target weight|",
    "Default frequency: `REBALANCE_FREQUENCY` in Notebook 07 — annual by default: lowest "
    "turnover, and selling to rebalance triggers taxes that are not modelled here",
)


# --- Macro (Notebook 11) -----------------------------------------------------

def stance_explanation(window_days=None):
    def scores(direction):
        return " · ".join(
            f"{intensity} {score:+.2f}"
            for (d, intensity), score in STANCE_SCORE_MAP.items()
            if d == direction
        )

    window = f"{window_days}-day" if window_days else "rolling"
    return _lines(
        "**Stance** per speech from the LLM's category:",
        f"hawkish: {scores('hawkish')}",
        f"dovish: {scores('dovish')}",
        "balanced: 0 · not about monetary policy: excluded",
        f"The line is the {window} average per central bank.",
    )
