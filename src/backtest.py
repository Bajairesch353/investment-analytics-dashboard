"""Backtest simulation helpers: periodic rebalancing, DCA and XIRR.

Extracted from notebooks/07_portfolio_backtesting.ipynb. Returns are daily
simple returns in a DataFrame (dates x tickers); weights are Series indexed
by ticker.
"""

import pandas as pd


def period_start_dates(index, frequency, include_first=False):
    """First trading day in `index` of each period (e.g. "MS", "QS", "YS").

    Periods are anchored after index.min(), so the first (partial) period is
    skipped unless include_first=True — right for rebalancing (the portfolio
    starts at target weights anyway), wrong for contributions (the first
    month's contribution would be missing).
    """
    index = pd.DatetimeIndex(index)
    candidates = pd.date_range(index.min(), index.max(), freq=frequency)
    positions = index.searchsorted(candidates)
    dates = [index[p] for p in positions if p < len(index)]

    if include_first and (not dates or dates[0] != index[0]):
        dates = [index[0]] + dates

    return list(dict.fromkeys(dates))


def simulate_rebalanced_portfolio(asset_returns, target_weights, rebalance_dates):
    """Daily returns of a portfolio that drifts with the market and is reset
    to target weights at the end of each rebalance date.

    Returns (portfolio_returns, turnover) where turnover is the one-way
    turnover 0.5 * sum|drifted - target| per rebalance date.
    """
    tickers = asset_returns.columns
    target_weights = target_weights.reindex(tickers).fillna(0)
    weights = target_weights.copy()
    rebalance_dates = set(pd.to_datetime(rebalance_dates))
    portfolio_returns = []
    turnover_records = {}

    for date, day_returns in asset_returns.iterrows():
        portfolio_returns.append((weights * day_returns).sum())
        weights = weights * (1 + day_returns)
        weights = weights / weights.sum()
        if date in rebalance_dates:
            turnover_records[date] = 0.5 * (weights - target_weights).abs().sum()
            weights = target_weights.copy()

    return (
        pd.Series(portfolio_returns, index=asset_returns.index),
        pd.Series(turnover_records, dtype=float)
    )


def simulate_dca(asset_returns, weights, contribution_dates, monthly_amount):
    """Value path of regular contributions split by `weights` (no rebalancing).

    Each contribution is added at the end of its date, after that day's
    returns. Returns a DataFrame with portfolio_value and total_invested.
    """
    weights = weights.reindex(asset_returns.columns).fillna(0)
    holdings_value = pd.Series(0.0, index=asset_returns.columns)
    total_invested = 0.0
    contribution_dates = set(pd.to_datetime(contribution_dates))
    history = []

    for date, day_returns in asset_returns.iterrows():
        holdings_value = holdings_value * (1 + day_returns)

        if date in contribution_dates:
            holdings_value = holdings_value + weights * monthly_amount
            total_invested += monthly_amount

        history.append({
            "date": date,
            "portfolio_value": holdings_value.sum(),
            "total_invested": total_invested
        })

    return pd.DataFrame(history).set_index("date")


def xirr(cashflow_dates, cashflows, lo=-0.99, hi=10.0, iterations=200):
    """Money-weighted annual return (internal rate of return on actual dates).

    Solved by bisection on [lo, hi]. Raises ValueError if the cashflows don't
    change sign or the rate lies outside the bracket, instead of silently
    returning a bracket edge.
    """
    if not (any(cf < 0 for cf in cashflows) and any(cf > 0 for cf in cashflows)):
        raise ValueError("XIRR needs both negative (paid in) and positive (paid out) cashflows.")

    t0 = min(cashflow_dates)

    def npv(rate):
        return sum(
            cf / (1 + rate) ** ((d - t0).days / 365)
            for cf, d in zip(cashflows, cashflow_dates)
        )

    if npv(lo) * npv(hi) > 0:
        raise ValueError(f"XIRR not bracketed in [{lo:.0%}, {hi:.0%}] — no sign change in NPV.")

    for _ in range(iterations):
        mid = (lo + hi) / 2
        if npv(mid) * npv(lo) > 0:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
