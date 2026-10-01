import numpy as np
import pandas as pd
import pytest

from src.fundamentals import (
    GROWTH_SCORE_WEIGHT,
    HEALTH_SCORE_WEIGHT,
    QUALITY_SCORE_WEIGHT,
    RATING_STRONG_CUTOFF,
    RATING_WEAK_CUTOFF,
    VALUATION_SCORE_WEIGHT,
    assign_fundamental_rating,
    cagr,
    growth_from_annual,
    compute_fundamental_score,
    compute_growth_score,
    compute_health_score,
    compute_quality_score,
    compute_valuation_score,
)


def test_compute_valuation_score_averages_sub_scores():
    assert compute_valuation_score(pe_score=1.0, pb_score=0.0) == pytest.approx(0.5)


def test_compute_quality_score_averages_sub_scores():
    assert compute_quality_score(
        roe_score=0.8, operating_margin_score=0.4
    ) == pytest.approx(0.6)


def test_compute_growth_score_averages_sub_scores():
    assert compute_growth_score(
        revenue_growth_score=1.0, earnings_growth_score=0.5
    ) == pytest.approx(0.75)


def test_compute_health_score_averages_sub_scores():
    assert compute_health_score(
        debt_to_equity_score=0.2, current_ratio_score=0.6
    ) == pytest.approx(0.4)


def test_compute_fundamental_score_all_ones_yields_one():
    assert compute_fundamental_score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_compute_fundamental_score_all_zeros_yields_zero():
    assert compute_fundamental_score(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_compute_fundamental_score_matches_manual_weighting():
    result = compute_fundamental_score(
        valuation_score=0.8,
        quality_score=0.6,
        growth_score=0.4,
        health_score=0.2,
    )
    expected = (
        VALUATION_SCORE_WEIGHT * 0.8
        + QUALITY_SCORE_WEIGHT * 0.6
        + GROWTH_SCORE_WEIGHT * 0.4
        + HEALTH_SCORE_WEIGHT * 0.2
    )
    assert result == pytest.approx(expected)


def _row(**overrides):
    base = {
        "data_quality_ok": True,
        "fundamental_score": 0.5,
    }
    base.update(overrides)
    return base


def test_assign_fundamental_rating_low_data_quality_is_watch():
    row = _row(data_quality_ok=False)
    assert assign_fundamental_rating(row) == "Watch - limited data"


def test_assign_fundamental_rating_strong_cutoff():
    row = _row(fundamental_score=RATING_STRONG_CUTOFF)
    assert assign_fundamental_rating(row) == "Strong fundamentals"


def test_assign_fundamental_rating_average_band():
    row = _row(fundamental_score=RATING_WEAK_CUTOFF)
    assert assign_fundamental_rating(row) == "Average fundamentals"


def test_assign_fundamental_rating_below_weak_cutoff_is_weak():
    row = _row(fundamental_score=RATING_WEAK_CUTOFF - 0.01)
    assert assign_fundamental_rating(row) == "Weak fundamentals"


def test_assign_fundamental_rating_missing_score_is_watch_not_weak():
    row = _row(fundamental_score=np.nan)
    assert assign_fundamental_rating(row) == "Watch - limited data"


def test_dimension_score_uses_available_sub_score():
    assert compute_valuation_score(np.nan, 0.8) == pytest.approx(0.8)
    assert np.isnan(compute_valuation_score(np.nan, np.nan))
    series = compute_growth_score(pd.Series([0.2, np.nan]), pd.Series([0.4, 0.6]))
    assert list(series) == pytest.approx([0.3, 0.6])


def test_fundamental_score_reweights_over_available_dimensions():
    result = compute_fundamental_score(1.0, 0.0, np.nan, 1.0)
    expected = (VALUATION_SCORE_WEIGHT + HEALTH_SCORE_WEIGHT) / (
        VALUATION_SCORE_WEIGHT + QUALITY_SCORE_WEIGHT + HEALTH_SCORE_WEIGHT
    )
    assert result == pytest.approx(expected)
    series = compute_fundamental_score(
        pd.Series([1.0, np.nan]), pd.Series([0.0, np.nan]),
        pd.Series([np.nan, np.nan]), pd.Series([1.0, np.nan]),
    )
    assert series.iloc[0] == pytest.approx(expected)
    assert np.isnan(series.iloc[1])


def test_cagr_and_undefined_cases():
    assert cagr(121.0, 100.0, 2) == pytest.approx(0.10)
    assert np.isnan(cagr(100.0, -50.0, 3))   # loss in the base year
    assert np.isnan(cagr(-10.0, 100.0, 3))   # loss in the latest year
    assert np.isnan(cagr(np.nan, 100.0, 3))


def test_growth_from_annual_uses_longest_span_up_to_max_years():
    idx = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31", "2021-12-31"])
    values = pd.Series([100.0, 110.0, 121.0, 133.1, 90.0], index=idx)
    growth, years = growth_from_annual(values, max_years=3)
    assert years == 3
    assert growth == pytest.approx(0.10)  # 2022 -> 2025

    short = pd.Series([100.0, 120.0], index=pd.to_datetime(["2024-12-31", "2025-12-31"]))
    assert growth_from_annual(short) == (pytest.approx(0.20), 1)
    assert np.isnan(growth_from_annual(short.iloc[:1])[0])
