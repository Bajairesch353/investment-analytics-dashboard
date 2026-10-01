import math

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    annualized_arithmetic_mu,
    build_constraints,
    compute_turnover,
    portfolio_return,
    portfolio_sharpe,
    portfolio_volatility,
    shrink_mu,
    solve_max_return,
    solve_max_sharpe,
    solve_min_variance,
    solve_min_variance_for_target_return,
    solve_risk_parity,
)

# Two uncorrelated assets: A low risk/low return, B high risk/high return.
MU = np.array([0.05, 0.15])
COV = np.array([[0.04, 0.0], [0.0, 0.16]])  # vols 20% / 40%
LONG_ONLY = [(0.0, 1.0), (0.0, 1.0)]
FULLY_INVESTED = build_constraints({}, {})


def test_annualized_arithmetic_mu_is_mean_times_periods():
    returns = pd.DataFrame({"a": [0.01, -0.01, 0.02]})
    assert annualized_arithmetic_mu(returns, 252)["a"] == pytest.approx(np.mean([0.01, -0.01, 0.02]) * 252)


def test_portfolio_return_and_volatility():
    w = np.array([0.5, 0.5])
    assert portfolio_return(w, MU) == pytest.approx(0.10)
    assert portfolio_volatility(w, COV) == pytest.approx(math.sqrt(0.25 * 0.04 + 0.25 * 0.16))


def test_portfolio_sharpe_zero_vol_is_nan():
    assert math.isnan(portfolio_sharpe(np.array([1.0, 0.0]), MU, np.zeros((2, 2))))


def test_min_variance_matches_inverse_variance_weights():
    # uncorrelated assets: w_i proportional to 1 / sigma_i^2 -> 0.8 / 0.2
    w = solve_min_variance(COV, LONG_ONLY, FULLY_INVESTED)
    assert w == pytest.approx([0.8, 0.2], abs=1e-4)


def test_max_sharpe_matches_tangency_portfolio():
    # uncorrelated: w_i proportional to (mu_i - rf) / sigma_i^2
    rf = 0.01
    raw = (MU - rf) / np.diag(COV)
    w = solve_max_sharpe(MU, COV, LONG_ONLY, FULLY_INVESTED, risk_free_rate=rf)
    assert w == pytest.approx(raw / raw.sum(), abs=1e-3)


def test_asset_class_cap_is_respected():
    constraints = build_constraints({"risky": np.array([False, True])}, {"risky": 0.1})
    w = solve_max_return(MU, LONG_ONLY, constraints)
    assert w[1] == pytest.approx(0.1, abs=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_solve_max_return_with_negative_mu_gives_min_return():
    w = solve_max_return(-MU, LONG_ONLY, FULLY_INVESTED)
    assert portfolio_return(w, MU) == pytest.approx(MU.min(), abs=1e-6)


def test_risk_parity_equalizes_risk_contributions():
    # uncorrelated: equal risk contribution means w_i proportional to 1 / sigma_i -> 2/3, 1/3
    w = solve_risk_parity(COV, LONG_ONLY, FULLY_INVESTED)
    assert w == pytest.approx([2 / 3, 1 / 3], abs=1e-3)


def test_min_variance_for_target_return_hits_target():
    w = solve_min_variance_for_target_return(MU, COV, LONG_ONLY, FULLY_INVESTED, 0.12)
    assert portfolio_return(w, MU) == pytest.approx(0.12, abs=1e-6)


def test_min_variance_for_unreachable_target_is_none():
    assert solve_min_variance_for_target_return(MU, COV, LONG_ONLY, FULLY_INVESTED, 0.5) is None


def test_shrink_mu_moves_toward_grand_mean():
    mu = pd.Series([0.0, 0.2])
    assert list(shrink_mu(mu, 0.5)) == pytest.approx([0.05, 0.15])
    assert list(shrink_mu(mu, 1.0)) == pytest.approx([0.1, 0.1])


def test_compute_turnover_is_one_way():
    a = pd.Series({"x": 1.0})
    b = pd.Series({"y": 1.0})
    assert compute_turnover(a, b) == pytest.approx(1.0)
