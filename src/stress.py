"""Stress-test helpers: bucket-based shock scenarios, historical and FX shocks.

Extracted from notebooks/09_portfolio_stress_testing.ipynb. Scenarios are
defined per stress bucket (a coarse risk category like "global_equity" or
"gold"), not per ticker, so the tracked code doesn't reveal the actual
holdings and new positions are shocked automatically. The ticker -> bucket
mapping lives in the gitignored data/manual/asset_universe_manual.csv
(column `stress_bucket`); missing entries fall back to asset class + region.
"""

import warnings

import pandas as pd

# Valid bucket names — a generic menu of risk categories, not the actual
# portfolio. Numbers per bucket (currency split incl. USD share, fund
# composition) live in the gitignored data/manual/bucket_lookthrough_manual.csv,
# see src/exposure.py.
STRESS_BUCKETS = (
    "global_equity",
    "em_equity",
    "europe_equity",
    "small_cap",
    "tech_thematic",
    "us_single_stock",
    "eu_single_stock",
    "gold",
    "crypto",
)


def fallback_stress_bucket(asset_class, region=None):
    """Best-effort bucket when a holding has no explicit stress_bucket."""
    if asset_class == "CRYPTO":
        return "crypto"
    if asset_class == "STOCK":
        return "eu_single_stock" if region == "Europe" else "us_single_stock"
    return "global_equity"


def resolve_stress_buckets(holdings, universe=None):
    """Series yf_ticker -> stress bucket for all holdings.

    Uses universe["stress_bucket"] where set and valid, otherwise the
    asset-class/region fallback — with a warning, because a fallback guess
    (e.g. a gold fund treated as global equity) can badly misstate a scenario.
    """
    explicit = {}
    regions = {}
    if universe is not None and "yf_ticker" in universe.columns:
        universe = universe.set_index("yf_ticker")
        if "stress_bucket" in universe.columns:
            explicit = universe["stress_bucket"].dropna().to_dict()
        if "region" in universe.columns:
            regions = universe["region"].dropna().to_dict()

    unknown = {t: b for t, b in explicit.items() if b not in STRESS_BUCKETS}
    if unknown:
        raise ValueError(f"Unknown stress_bucket values: {unknown}. Valid: {sorted(STRESS_BUCKETS)}")

    buckets = {}
    fallbacks = []
    for _, row in holdings.iterrows():
        ticker = row["yf_ticker"]
        if ticker in explicit:
            buckets[ticker] = explicit[ticker]
        else:
            buckets[ticker] = fallback_stress_bucket(row.get("asset_class"), regions.get(ticker))
            fallbacks.append(f"{ticker} -> {buckets[ticker]}")

    if fallbacks:
        warnings.warn(
            "No stress_bucket in asset_universe_manual.csv for: "
            + ", ".join(fallbacks)
            + " (fallback by asset class/region — set it explicitly)."
        )

    return pd.Series(buckets, name="stress_bucket")


def expand_bucket_scenario(bucket_shocks, ticker_buckets):
    """{bucket: shock} -> {ticker: shock}; buckets not in the scenario get 0."""
    return {
        ticker: bucket_shocks.get(bucket, 0.0)
        for ticker, bucket in ticker_buckets.items()
    }


def historical_shocks(asset_returns, start, end):
    """Compounded return per ticker over [start, end] from daily returns."""
    window = asset_returns.loc[start:end]
    if window.empty:
        raise ValueError(f"No return data between {start} and {end}.")
    return ((1 + window).prod() - 1).to_dict()


def fx_translation_shocks(eur_move, ticker_buckets, usd_shares):
    """Pure currency-translation effect of a EUR move (e.g. +0.15 = EUR +15%)
    on each holding, all else equal: USD-denominated value in EUR changes by
    1 / (1 + eur_move) - 1, scaled by the bucket's USD share
    ({bucket: share}, from src.exposure.usd_shares)."""
    missing = sorted({b for _, b in ticker_buckets.items() if b not in usd_shares})
    if missing:
        warnings.warn(f"No USD share for buckets {missing} — treated as 0 in the FX scenario.")
    usd_value_change = 1 / (1 + eur_move) - 1
    return {
        ticker: usd_value_change * usd_shares.get(bucket, 0.0)
        for ticker, bucket in ticker_buckets.items()
    }


def combine_worst_of(*scenarios):
    """Per ticker the worst (lowest) shock across several {ticker: shock} dicts."""
    tickers = set().union(*scenarios)
    return {
        ticker: min(scenario.get(ticker, 0.0) for scenario in scenarios)
        for ticker in tickers
    }


def summarize_scenario(holdings, shock_pct_by_ticker):
    """Apply {ticker: shock} to holdings' market values.

    Tickers missing from the scenario are left unshocked (0%). Returns
    (summary dict, per-position DataFrame).
    """
    shocks = holdings["yf_ticker"].map(shock_pct_by_ticker).fillna(0.0)

    result = holdings.copy()
    result["shock_pct"] = shocks
    result["shocked_value"] = holdings["market_value"] * (1 + shocks)
    result["loss_eur"] = result["shocked_value"] - result["market_value"]
    result["loss_pct_of_position"] = result["loss_eur"] / result["market_value"]
    result["new_weight"] = result["shocked_value"] / result["shocked_value"].sum()

    total_before = result["market_value"].sum()
    total_after = result["shocked_value"].sum()

    summary = {
        "total_before_eur": total_before,
        "total_after_eur": total_after,
        "total_loss_eur": total_after - total_before,
        "total_loss_pct": (total_after - total_before) / total_before,
    }
    return summary, result
