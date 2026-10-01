"""Asset selection scoring and decision-bucket logic.

Extracted from notebooks/05_asset_selection_scoring.ipynb.
"""

# Generic fallback role scores. The authoritative per-asset value is the
# `role_score` column in data/manual/asset_universe_manual.csv (gitignored,
# like the rest of the real universe) — asset-specific roles would otherwise
# fingerprint the real holdings/watchlist in published source code. This map
# only fills gaps for assets without an explicit role_score; notebook 05
# warns when that happens, so every new asset gets a deliberate score.
ROLE_SCORE_MAP = {
    "Core equity": 1.00,
    "Equity diversification": 0.90,
    "Thematic growth": 0.85,
    "Crisis hedge": 0.80,
    "Size factor": 0.70,
    "Regional tilt": 0.70,
    "Single stock": 0.50,
    "Alternative asset": 0.50,
}

DEFAULT_ROLE_SCORE = 0.50

# Quant sub-score weights (must sum to 1.0). Raw return is deliberately not
# a separate component: Sharpe already is (return - rf) / volatility, so
# scoring return on its own as well would double-count it and tilt the
# ranking towards high-return/high-risk assets.
SHARPE_SCORE_WEIGHT = 0.50
DRAWDOWN_SCORE_WEIGHT = 0.30
VOLATILITY_SCORE_WEIGHT = 0.20

# Final blend of quant score vs strategic role score.
QUANT_SCORE_WEIGHT = 0.60
ROLE_SCORE_WEIGHT = 0.40

MIN_RETURN_OBSERVATIONS = 252

PROTECTED_ROLE_SCORE_CUTOFF = 0.80
REVIEW_SCORE_CUTOFF = 0.40
CANDIDATE_SCORE_CUTOFF = 0.60
MATERIAL_WEIGHT = 0.02


def percentile_score(series, higher_is_better=True):
    score = series.rank(pct=True)

    if not higher_is_better:
        score = 1 - score

    return score


def role_score(role):
    return ROLE_SCORE_MAP.get(role, DEFAULT_ROLE_SCORE)


def compute_quant_score(sharpe_score, drawdown_score, volatility_score):
    return (
        SHARPE_SCORE_WEIGHT * sharpe_score
        + DRAWDOWN_SCORE_WEIGHT * drawdown_score
        + VOLATILITY_SCORE_WEIGHT * volatility_score
    )


def compute_selection_score(quant_score, role_score_value):
    return (
        QUANT_SCORE_WEIGHT * quant_score
        + ROLE_SCORE_WEIGHT * role_score_value
    )


def assign_decision(row):
    """Decision bucket for one asset selection row.

    `row` must support dict-like access to: data_quality_ok,
    is_current_holding, is_protected_role, actual_weight,
    selection_score, source.
    """
    if not row["data_quality_ok"]:
        return "Watch - limited data"

    if (
        row["is_current_holding"]
        and row["is_protected_role"]
    ):
        return "Keep - strategic role"

    if (
        row["is_current_holding"]
        and row["actual_weight"] >= MATERIAL_WEIGHT
        and row["selection_score"] <= REVIEW_SCORE_CUTOFF
    ):
        return "Review"

    if row["is_current_holding"]:
        return "Keep"

    if (
        row["source"] == "Watchlist"
        and row["selection_score"] >= CANDIDATE_SCORE_CUTOFF
    ):
        return "Candidate"

    if row["source"] == "Watchlist":
        return "Watch"

    return "Watch"
