"""Fundamental scoring for individual stock holdings and watchlist candidates.

Extracted from notebooks/12_fundamental_stock_scoring.ipynb. Percentile-based,
mirrors the structure of src/selection.py -- scores are only comparable
within the STOCK-class universe they were computed on, not against
funds/crypto or across a universe of a different size.

Missing values: a dimension averages the sub-scores that are available, and
the final score re-weights over the available dimensions. A stock that fails
the data-quality threshold or ends up without a score is rated
"Watch - limited data", never silently "Weak".
"""

import numpy as np
import pandas as pd

# Dimension weights for the final blend (must sum to 1.0). A deliberate
# judgement, not calibrated (a percentile score within a self-chosen universe
# has no ground truth to calibrate against): quality
# weighted highest, in line with the quality-factor literature; valuation,
# growth and balance-sheet health roughly equal.
VALUATION_SCORE_WEIGHT = 0.25
QUALITY_SCORE_WEIGHT = 0.30
GROWTH_SCORE_WEIGHT = 0.25
HEALTH_SCORE_WEIGHT = 0.20

# Of the 8 raw yfinance fields used below, how many must be present for a
# ticker's score to be trusted (mirrors MIN_RETURN_OBSERVATIONS in
# src/selection.py). Below this, percentile ranks are computed on too few
# comparable metrics to mean anything.
MIN_REQUIRED_FIELDS = 6

# Scores are percentiles within the universe, so the cutoffs are relative:
# roughly the upper and lower third.
RATING_STRONG_CUTOFF = 0.65
RATING_WEAK_CUTOFF = 0.35

# Growth = compound annual growth over up to this many fiscal years.
GROWTH_YEARS = 3


def cagr(latest, base, years):
    """Compound annual growth rate; NaN if undefined (missing values, a
    non-positive base or end value, e.g. a loss year, or years < 1)."""
    if years is None or years < 1 or pd.isna(latest) or pd.isna(base) or base <= 0 or latest <= 0:
        return np.nan
    return (latest / base) ** (1 / years) - 1


def growth_from_annual(values, max_years=GROWTH_YEARS):
    """CAGR over the longest span up to `max_years` fiscal years.

    `values`: Series of annual figures indexed by fiscal-year end date (any
    order, NaN allowed). Returns (growth, years used); (NaN, None) if fewer
    than two years are available."""
    values = values.dropna().sort_index(ascending=False)
    if len(values) < 2:
        return np.nan, None
    years = min(max_years, len(values) - 1)
    return cagr(values.iloc[0], values.iloc[years], years), years


def _nanmean(*values):
    """Mean of the available (non-NaN) values; works for scalars and Series."""
    if any(isinstance(v, pd.Series) for v in values):
        return pd.concat(values, axis=1).mean(axis=1, skipna=True)
    finite = [v for v in values if not pd.isna(v)]
    return float(np.mean(finite)) if finite else np.nan


def compute_valuation_score(pe_score, pb_score):
    return _nanmean(pe_score, pb_score)


def compute_quality_score(roe_score, operating_margin_score):
    return _nanmean(roe_score, operating_margin_score)


def compute_growth_score(revenue_growth_score, earnings_growth_score):
    return _nanmean(revenue_growth_score, earnings_growth_score)


def compute_health_score(debt_to_equity_score, current_ratio_score):
    return _nanmean(debt_to_equity_score, current_ratio_score)


def compute_fundamental_score(valuation_score, quality_score, growth_score, health_score):
    """Weighted blend, re-weighted over the dimensions that are available."""
    scores = [valuation_score, quality_score, growth_score, health_score]
    weights = [VALUATION_SCORE_WEIGHT, QUALITY_SCORE_WEIGHT, GROWTH_SCORE_WEIGHT, HEALTH_SCORE_WEIGHT]

    if any(isinstance(v, pd.Series) for v in scores):
        frame = pd.concat(scores, axis=1)
        w = pd.Series(weights, index=frame.columns)
        available = frame.notna()
        weight_sum = available.mul(w, axis=1).sum(axis=1)
        weighted = frame.fillna(0).mul(w, axis=1).sum(axis=1)
        return (weighted / weight_sum).where(weight_sum > 0)

    pairs = [(s, w) for s, w in zip(scores, weights) if not pd.isna(s)]
    if not pairs:
        return np.nan
    return sum(s * w for s, w in pairs) / sum(w for _, w in pairs)


def assign_fundamental_rating(row):
    """Rating bucket for one fundamental-scoring row.

    `row` must support dict-like access to: data_quality_ok,
    fundamental_score. Mirrors assign_decision() in src/selection.py --
    trusts the score only when enough raw fields were available.
    """
    score = row["fundamental_score"]

    if not row["data_quality_ok"] or pd.isna(score):
        return "Watch - limited data"

    if score >= RATING_STRONG_CUTOFF:
        return "Strong fundamentals"

    if score >= RATING_WEAK_CUTOFF:
        return "Average fundamentals"

    return "Weak fundamentals"
