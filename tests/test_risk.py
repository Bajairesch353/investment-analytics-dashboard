import math

import numpy as np
import pandas as pd
import pytest

from src.risk import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_risk_contribution,
    drawdown_series,
    effective_number_of_positions,
    herfindahl_index,
    max_drawdown,
    recovery_time,
    sharpe_ratio,
    sortino_ratio,
    top_n_weight,
)


def test_annualized_return_flat_series_is_zero():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert annualized_return(returns, periods_per_year=3) == pytest.approx(0.0)


def test_annualized_return_compounds_correctly():
    # 1% daily return for 252 days should annualize to (1.01**252) - 1
    returns = pd.Series([0.01] * 252)
    expected = 1.01 ** 252 - 1
    assert annualized_return(returns, periods_per_year=252) == pytest.approx(expected)


def test_annualized_return_empty_series_is_nan():
    assert math.isnan(annualized_return(pd.Series([], dtype=float)))


def test_annualized_return_drops_nan_before_compounding():
    returns = pd.Series([0.01, np.nan, 0.01])
    returns_clean = pd.Series([0.01, 0.01])
    assert annualized_return(returns, periods_per_year=2) == pytest.approx(
        annualized_return(returns_clean, periods_per_year=2)
    )


def test_annualized_volatility_scales_by_sqrt_periods():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    expected = returns.std() * np.sqrt(252)
    assert annualized_volatility(returns, periods_per_year=252) == pytest.approx(expected)


def test_sharpe_ratio_zero_vol_is_nan():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert math.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_matches_manual_computation():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    result = sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
    expected = (
        annualized_return(returns, 252) - 0.02
    ) / annualized_volatility(returns, 252)
    assert result == pytest.approx(expected)


def test_drawdown_series_tracks_peak_to_trough():
    returns = pd.Series([0.1, -0.2, 0.05])
    dd = drawdown_series(returns)
    # wealth path: 1.1, 0.88, 0.924 -> running max 1.1, 1.1, 1.1
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.88 / 1.1 - 1)
    assert dd.iloc[2] == pytest.approx(0.924 / 1.1 - 1)


def test_max_drawdown_is_minimum_of_drawdown_series():
    returns = pd.Series([0.1, -0.3, 0.05, -0.1])
    assert max_drawdown(returns) == pytest.approx(drawdown_series(returns).min())


def test_max_drawdown_empty_series_is_nan():
    assert math.isnan(max_drawdown(pd.Series([], dtype=float)))


def test_calmar_ratio_zero_drawdown_is_nan():
    returns = pd.Series([0.0, 0.0])
    assert math.isnan(calmar_ratio(returns))


def test_calmar_ratio_matches_manual_computation():
    returns = pd.Series([0.02, -0.05, 0.03, -0.01, 0.04])
    result = calmar_ratio(returns, periods_per_year=252)
    expected = annualized_return(returns, 252) / abs(max_drawdown(returns))
    assert result == pytest.approx(expected)


def test_herfindahl_index_equal_weights():
    weights = [0.25, 0.25, 0.25, 0.25]
    assert herfindahl_index(weights) == pytest.approx(0.25)


def test_herfindahl_index_concentrated_portfolio():
    weights = [1.0]
    assert herfindahl_index(weights) == pytest.approx(1.0)


def test_effective_number_of_positions_matches_inverse_hhi():
    weights = [0.5, 0.3, 0.2]
    expected = 1 / herfindahl_index(weights)
    assert effective_number_of_positions(weights) == pytest.approx(expected)


def test_top_n_weight_sorts_regardless_of_input_order():
    weights = [0.1, 0.5, 0.05, 0.25, 0.1]
    assert top_n_weight(weights, 2) == pytest.approx(0.75)
    assert top_n_weight(weights, len(weights)) == pytest.approx(sum(weights))


def test_compute_risk_contribution_sums_to_total_variance_share():
    weights = np.array([0.6, 0.4])
    cov_matrix = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    portfolio_vol, risk_contribution, risk_contribution_pct = compute_risk_contribution(
        weights, cov_matrix
    )

    expected_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    assert portfolio_vol == pytest.approx(expected_vol)
    assert risk_contribution.sum() == pytest.approx(portfolio_vol)
    assert risk_contribution_pct.sum() == pytest.approx(1.0)


def test_compute_risk_contribution_equal_weight_single_asset():
    weights = np.array([1.0])
    cov_matrix = np.array([[0.16]])

    portfolio_vol, risk_contribution, risk_contribution_pct = compute_risk_contribution(
        weights, cov_matrix
    )

    assert portfolio_vol == pytest.approx(0.4)
    assert risk_contribution[0] == pytest.approx(0.4)
    assert risk_contribution_pct[0] == pytest.approx(1.0)


def test_sortino_ratio_uses_target_downside_deviation_over_all_periods():
    returns = pd.Series([0.02, -0.01, 0.03, -0.03, 0.01])
    result = sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
    # shortfalls below 0: 0, -0.01, 0, -0.03, 0 -> mean square over all 5 periods
    downside_deviation = math.sqrt((0.01**2 + 0.03**2) / 5) * math.sqrt(252)
    expected = (annualized_return(returns, 252) - 0.02) / downside_deviation
    assert result == pytest.approx(expected)


def test_sortino_ratio_respects_mar():
    returns = pd.Series([0.02, 0.005, 0.03])
    # nothing below 0, but 0.005 is below a 1% target
    downside_deviation = math.sqrt(0.005**2 / 3) * math.sqrt(252)
    expected = annualized_return(returns, 252) / downside_deviation
    assert sortino_ratio(returns, mar=0.01) == pytest.approx(expected)


def test_sortino_ratio_no_downside_is_nan():
    returns = pd.Series([0.01, 0.02, 0.0])
    assert math.isnan(sortino_ratio(returns))


def test_recovery_time_counts_calendar_days_from_trough_to_old_high():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-10"])
    # wealth: 1.1, 0.88 (trough), 0.968, 1.1616 (new high)
    returns = pd.Series([0.1, -0.2, 0.1, 0.2], index=index)
    assert recovery_time(returns) == 8


def test_recovery_time_without_drawdown_is_zero():
    index = pd.date_range("2024-01-01", periods=4)
    returns = pd.Series([0.01, 0.02, 0.0, 0.01], index=index)
    assert recovery_time(returns) == 0


def test_recovery_time_unrecovered_drawdown_is_nan():
    index = pd.date_range("2024-01-01", periods=3)
    returns = pd.Series([0.1, -0.2, 0.05], index=index)
    assert math.isnan(recovery_time(returns))
