import pandas as pd
import pytest

from src.backtest import (
    period_start_dates,
    simulate_dca,
    simulate_rebalanced_portfolio,
    xirr,
)


def test_period_start_dates_skips_first_partial_period_by_default():
    index = pd.bdate_range("2020-01-03", "2020-03-31")
    assert period_start_dates(index, "MS") == [pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02")]


def test_period_start_dates_include_first_adds_start_date():
    index = pd.bdate_range("2020-01-03", "2020-03-31")
    dates = period_start_dates(index, "MS", include_first=True)
    assert dates[0] == pd.Timestamp("2020-01-03")
    assert len(dates) == 3


def test_simulate_rebalanced_portfolio_drift_and_turnover():
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    returns = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 0.0]}, index=index)
    target = pd.Series({"a": 0.5, "b": 0.5})

    port, turnover = simulate_rebalanced_portfolio(returns, target, [index[0]])

    # day 1: 50/50 -> a doubles -> 2/3 vs 1/3, rebalanced back at day end
    assert port.iloc[0] == pytest.approx(0.5)
    assert turnover.loc[index[0]] == pytest.approx(0.5 * (abs(2 / 3 - 0.5) + abs(1 / 3 - 0.5)))


def test_simulate_rebalanced_portfolio_without_rebalancing_lets_weights_drift():
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    returns = pd.DataFrame({"a": [1.0, 0.1], "b": [0.0, 0.0]}, index=index)
    port, turnover = simulate_rebalanced_portfolio(returns, pd.Series({"a": 0.5, "b": 0.5}), [])
    # day 2 weights are 2/3 a, 1/3 b
    assert port.iloc[1] == pytest.approx(2 / 3 * 0.1)
    assert turnover.empty


def test_simulate_dca_contributions_and_growth():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"a": [0.0, 0.1, 0.0]}, index=index)
    result = simulate_dca(returns, pd.Series({"a": 1.0}), [index[0], index[2]], monthly_amount=100)

    assert result["total_invested"].iloc[-1] == pytest.approx(200)
    # 100 grows 10% on day 2, second 100 added on day 3
    assert result["portfolio_value"].iloc[-1] == pytest.approx(210)


def test_xirr_single_year_matches_simple_return():
    dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]  # 365 days
    assert xirr(dates, [-100, 110]) == pytest.approx(0.10, abs=1e-8)


def test_xirr_negative_return():
    dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]
    assert xirr(dates, [-100, 80]) == pytest.approx(-0.20, abs=1e-8)


def test_xirr_without_sign_change_raises():
    dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]
    with pytest.raises(ValueError):
        xirr(dates, [-100, -50])


def test_xirr_outside_bracket_raises():
    dates = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]
    with pytest.raises(ValueError):
        xirr(dates, [-1, 100])  # +9900% > default upper bound
