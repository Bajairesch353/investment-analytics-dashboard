"""Portfolio and asset risk/performance metric calculations.

Extracted from notebooks/03_portfolio_risk_performance.ipynb and
notebooks/04_asset_universe_watchlist.ipynb, which both defined
near-identical versions of annualized_return / annualized_volatility /
max_drawdown inline. Consolidated here as a single implementation
(the more defensive dropna-guarded form from notebook 04, which is
numerically equivalent to notebook 03's version since pandas .prod()
already skips NaNs by default).
"""

import numpy as np
import pandas as pd

# Risk-free proxy for all Sharpe/Sortino calculations (notebooks 03, 04, 07,
# 08): ECB deposit facility rate (2.50% as of the 10 Sep 2026 ECB decision),
# not a US Treasury yield. The portfolio is reported entirely in EUR, so the
# risk-free reference should be a short-term EUR rate, not USD-denominated
# and not a long (10Y) maturity that itself carries duration risk. Single
# source of truth — update here when the ECB rate changes.
RISK_FREE_RATE = 0.025


def annualized_return(returns, periods_per_year=252):
    returns = pd.Series(returns).dropna()

    if returns.shape[0] == 0:
        return np.nan

    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]

    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualized_volatility(returns, periods_per_year=252):
    returns = pd.Series(returns).dropna()

    if returns.shape[0] == 0:
        return np.nan

    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)

    if ann_vol == 0 or pd.isna(ann_vol):
        return np.nan

    return (ann_return - risk_free_rate) / ann_vol


def drawdown_series(returns):
    returns = pd.Series(returns)
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1


def max_drawdown(returns):
    returns = pd.Series(returns).dropna()

    if returns.shape[0] == 0:
        return np.nan

    return drawdown_series(returns).min()


def calmar_ratio(returns, periods_per_year=252):
    ann_return = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))

    if mdd == 0 or pd.isna(mdd):
        return np.nan

    return ann_return / mdd


def herfindahl_index(weights):
    return float(np.sum(np.asarray(weights, dtype=float) ** 2))


def effective_number_of_positions(weights):
    hhi = herfindahl_index(weights)

    if hhi == 0:
        return np.nan

    return 1 / hhi


def top_n_weight(weights, n):
    sorted_weights = np.sort(np.asarray(weights, dtype=float))[::-1]
    return float(sorted_weights[:n].sum())


def compute_risk_contribution(weights, cov_matrix):
    """Marginal / total risk contribution per position.

    weights: 1D array-like aligned with cov_matrix's rows/columns.
    cov_matrix: 2D array-like (annualized covariance matrix).

    Returns (portfolio_vol, risk_contribution, risk_contribution_pct).
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    portfolio_vol = np.sqrt(w.T @ cov @ w)

    marginal_risk_contribution = (cov @ w) / portfolio_vol
    risk_contribution = w * marginal_risk_contribution
    risk_contribution_pct = risk_contribution / portfolio_vol

    return portfolio_vol, risk_contribution, risk_contribution_pct


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252, mar=0.0):
    """(Annualized return - risk-free rate) / annualized downside deviation.

    Downside deviation is the target downside deviation (Sortino & Price):
    sqrt(mean(min(r - mar, 0)^2)) over ALL periods, so days above the target
    count as zero shortfall. Not the std of only the negative days, which is
    centred on their own mean and ignores how often losses occur.
    """
    returns = pd.Series(returns).dropna()
    ann_return = annualized_return(returns, periods_per_year)

    shortfall = np.minimum(returns - mar, 0.0)
    downside_deviation = np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods_per_year)

    if downside_deviation == 0 or pd.isna(downside_deviation):
        return np.nan

    return (ann_return - risk_free_rate) / downside_deviation


def recovery_time(returns):
    """Calendar days from the maximum-drawdown trough until the old high is regained.

    Returns 0 if there is no drawdown at all and NaN if the trough has not
    been recovered by the end of the series. Needs a DatetimeIndex.
    """
    returns = pd.Series(returns).dropna()

    if returns.shape[0] == 0:
        return np.nan

    dd = drawdown_series(returns)

    if dd.min() >= -1e-9:
        return 0

    trough_date = dd.idxmin()

    after_trough = dd.loc[dd.index > trough_date]
    recovered = after_trough[after_trough >= -1e-9]

    if recovered.empty:
        return np.nan

    recovery_date = recovered.index[0]
    return (recovery_date - trough_date).days