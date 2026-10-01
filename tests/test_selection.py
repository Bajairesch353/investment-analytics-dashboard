import pandas as pd
import pytest

from src.selection import (
    CANDIDATE_SCORE_CUTOFF,
    MATERIAL_WEIGHT,
    PROTECTED_ROLE_SCORE_CUTOFF,
    REVIEW_SCORE_CUTOFF,
    assign_decision,
    compute_quant_score,
    compute_selection_score,
    percentile_score,
    role_score,
)


def test_percentile_score_higher_is_better_ranks_ascending():
    series = pd.Series([10, 20, 30])
    result = percentile_score(series, higher_is_better=True)
    assert result.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_percentile_score_lower_is_better_inverts_ranking():
    series = pd.Series([10, 20, 30])
    result = percentile_score(series, higher_is_better=False)
    assert result.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])


def test_role_score_known_role():
    assert role_score("Core equity") == 1.00


def test_crisis_hedge_counts_as_protected_role():
    assert role_score("Crisis hedge") >= PROTECTED_ROLE_SCORE_CUTOFF


def test_role_score_unknown_role_falls_back_to_default():
    assert role_score("Some new role") == 0.50


def test_compute_quant_score_weights_sum_to_expected_blend():
    # equal 1.0 scores on every sub-metric should yield 1.0
    assert compute_quant_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    # equal 0.0 scores should yield 0.0
    assert compute_quant_score(0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_compute_quant_score_matches_manual_weighting():
    result = compute_quant_score(
        sharpe_score=0.6,
        drawdown_score=0.4,
        volatility_score=0.2,
    )
    expected = 0.50 * 0.6 + 0.30 * 0.4 + 0.20 * 0.2
    assert result == pytest.approx(expected)


def test_compute_selection_score_blends_quant_and_role():
    result = compute_selection_score(quant_score=1.0, role_score_value=0.0)
    assert result == pytest.approx(0.60)

    result = compute_selection_score(quant_score=0.0, role_score_value=1.0)
    assert result == pytest.approx(0.40)


def _row(**overrides):
    base = {
        "data_quality_ok": True,
        "is_current_holding": False,
        "is_protected_role": False,
        "actual_weight": 0.0,
        "selection_score": 0.5,
        "source": "Current holding",
    }
    base.update(overrides)
    return base


def test_assign_decision_low_data_quality_is_watch_limited_data():
    row = _row(data_quality_ok=False)
    assert assign_decision(row) == "Watch - limited data"


def test_assign_decision_protected_role_current_holding_is_keep_strategic():
    row = _row(is_current_holding=True, is_protected_role=True)
    assert assign_decision(row) == "Keep - strategic role"


def test_assign_decision_material_underperformer_is_review():
    row = _row(
        is_current_holding=True,
        actual_weight=MATERIAL_WEIGHT,
        selection_score=REVIEW_SCORE_CUTOFF,
    )
    assert assign_decision(row) == "Review"


def test_assign_decision_current_holding_default_is_keep():
    row = _row(is_current_holding=True, actual_weight=0.5, selection_score=0.9)
    assert assign_decision(row) == "Keep"


def test_assign_decision_watchlist_high_score_is_candidate():
    row = _row(source="Watchlist", selection_score=CANDIDATE_SCORE_CUTOFF)
    assert assign_decision(row) == "Candidate"


def test_assign_decision_watchlist_low_score_is_watch():
    row = _row(source="Watchlist", selection_score=CANDIDATE_SCORE_CUTOFF - 0.01)
    assert assign_decision(row) == "Watch"


def test_assign_decision_non_holding_non_watchlist_is_watch():
    row = _row(source="Something else")
    assert assign_decision(row) == "Watch"


def test_protected_role_cutoff_value_unchanged():
    assert PROTECTED_ROLE_SCORE_CUTOFF == 0.80
