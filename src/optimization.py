"""Mean-variance / risk-parity portfolio optimization helpers.

Extracted from notebooks/08_portfolio_optimization.ipynb. All solvers use
scipy's SLSQP with long-only bounds and linear constraints (fully invested
plus optional asset-class caps); weights are plain numpy arrays aligned with
the order of `mu` / `cov`.

Expected returns `mu` are annualized ARITHMETIC means: mean-variance theory
needs E[w'r] = w'mu, which only holds for arithmetic expectations — a
geometric return (CAGR) doesn't aggregate linearly across weights.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk import RISK_FREE_RATE, compute_risk_contribution


def annualized_arithmetic_mu(returns, periods_per_year=252):
    """Annualized arithmetic mean return per column (input for mean-variance)."""
    return pd.DataFrame(returns).mean() * periods_per_year


def portfolio_return(weights, mu):
    return float(np.dot(weights, mu))


def portfolio_volatility(weights, cov):
    weights = np.asarray(weights, dtype=float)
    return float(np.sqrt(weights.T @ np.asarray(cov, dtype=float) @ weights))


def portfolio_sharpe(weights, mu, cov, risk_free_rate=RISK_FREE_RATE):
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return np.nan
    return (portfolio_return(weights, mu) - risk_free_rate) / vol


def build_constraints(class_masks, class_aggregate_caps):
    """Fully-invested constraint plus one aggregate cap per asset class.

    class_masks: {asset_class: boolean array over assets}
    class_aggregate_caps: {asset_class: max total weight}; classes without a
    cap are unconstrained.
    """
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for asset_class, mask in class_masks.items():
        cap = class_aggregate_caps.get(asset_class)
        if cap is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, mask=mask, cap=cap: cap - w[mask].sum()
            })
    return constraints


def _solve(objective, n, bounds, constraints, x0=None, label="Optimization"):
    if x0 is None:
        x0 = np.repeat(1 / n, n)
    result = minimize(
        objective, x0,
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    if not result.success:
        raise RuntimeError(f"{label} failed: {result.message}")
    return result.x


def solve_min_variance(cov, bounds, constraints):
    cov = np.asarray(cov, dtype=float)
    return _solve(
        lambda w: portfolio_volatility(w, cov),
        cov.shape[0], bounds, constraints, label="Min-Variance optimization"
    )


def solve_max_sharpe(mu, cov, bounds, constraints, risk_free_rate=RISK_FREE_RATE):
    return _solve(
        lambda w: -portfolio_sharpe(w, mu, cov, risk_free_rate),
        len(mu), bounds, constraints, label="Max-Sharpe optimization"
    )


def solve_max_return(mu, bounds, constraints):
    """Highest expected return reachable under the bounds/caps — the upper
    end of the constrained frontier (usually far below mu.max()). Pass -mu
    for the lowest reachable return."""
    return _solve(
        lambda w: -portfolio_return(w, mu),
        len(mu), bounds, constraints, label="Max-return optimization"
    )


def solve_risk_parity(cov, bounds, constraints):
    """Weights whose risk contributions are as equal as the bounds allow.

    With tight position caps exact risk parity is often infeasible; the
    result is then the closest feasible portfolio, not true risk parity.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    target = np.repeat(1 / n, n)

    def risk_parity_objective(w):
        _, _, risk_contribution_pct = compute_risk_contribution(w, cov)
        return float(np.sum((risk_contribution_pct - target) ** 2))

    return _solve(
        risk_parity_objective, n, bounds, constraints, label="Risk-Parity optimization"
    )


def solve_min_variance_for_target_return(mu, cov, bounds, base_constraints, target_return, x0=None):
    """Min-variance weights for a given expected return, or None if infeasible."""
    constraints = base_constraints + [{
        "type": "eq",
        "fun": lambda w: portfolio_return(w, mu) - target_return
    }]
    try:
        return _solve(
            lambda w: portfolio_volatility(w, cov),
            len(mu), bounds, constraints, x0=x0
        )
    except RuntimeError:
        return None


def shrink_mu(mu, shrinkage):
    """Shrink expected returns toward their cross-sectional mean (0 = none, 1 = all equal)."""
    grand_mean = mu.mean()
    return (1 - shrinkage) * mu + shrinkage * grand_mean


def compute_turnover(from_weights, to_weights):
    """One-way turnover 0.5 * sum|w_to - w_from| between two weight Series."""
    aligned = pd.concat([from_weights, to_weights], axis=1).fillna(0.0)
    return 0.5 * (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum()
