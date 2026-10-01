"""Look-through exposure (country / sector / currency / single stocks) and
market beta for notebook 10; USD shares for the FX scenario in notebook 09.

Fund compositions are kept per stress bucket (src/stress.py) in the gitignored
data/manual/bucket_lookthrough_manual.csv — one row per (bucket, dimension,
category) with weight, source and as-of date. Fund-level numbers would
identify the actual funds held, so this tracked module contains only a
generic example (seeded when the file is missing). The currency rows of that
file are also the single source for the USD shares used in notebook 09.

CSV dimensions: country, sector, currency (each sums to 1 per bucket) and
top_holding (largest constituents, sum <= 1; `category` = ticker, optional
`label` = company name for charts). Single-stock buckets need no
rows: country/currency come from SINGLE_STOCK_BUCKETS, the sector (and an
optional country override) from asset_universe_manual.csv.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.stress import STRESS_BUCKETS

DIMENSIONS = ("country", "sector", "currency")
LOOKTHROUGH_COLUMNS = ["bucket", "dimension", "category", "weight", "label", "source", "as_of", "note"]

# Buckets that are not equity -- everything else counts as equity exposure.
NON_EQUITY_BUCKETS = ("gold", "crypto")

SINGLE_STOCK_BUCKETS = {
    "us_single_stock": {"country": "USA", "currency": "USD"},
    "eu_single_stock": {"country": "Other Europe", "currency": "EUR"},
}


def example_bucket_lookthrough():
    """Generic, rounded example (not real fund data) for every fund bucket —
    seeds the manual CSV on a fresh checkout. Replace the values with the
    factsheet of the fund you hold in each bucket; buckets you don't use can
    stay or be deleted."""
    rows = [
        ("global_equity", "country", "USA", 0.70), ("global_equity", "country", "Other Developed", 0.30),
        ("global_equity", "sector", "Technology", 0.25), ("global_equity", "sector", "Financials", 0.15),
        ("global_equity", "sector", "Other Sectors", 0.60),
        ("global_equity", "currency", "USD", 0.70), ("global_equity", "currency", "EUR", 0.10),
        ("global_equity", "currency", "Other Developed FX", 0.20),
        ("global_equity", "top_holding", "AAPL", 0.05, "Apple"),
        ("global_equity", "top_holding", "MSFT", 0.04, "Microsoft"),
        ("global_equity", "top_holding", "NVDA", 0.05, "NVIDIA"),
        ("global_equity", "top_holding", "AMZN", 0.025, "Amazon"),
        ("global_equity", "top_holding", "GOOGL", 0.02, "Alphabet"),
        ("em_equity", "country", "Taiwan", 0.25), ("em_equity", "country", "China", 0.25),
        ("em_equity", "country", "Other Emerging", 0.50),
        ("em_equity", "sector", "Technology", 0.30), ("em_equity", "sector", "Other Sectors", 0.70),
        ("em_equity", "currency", "TWD", 0.25), ("em_equity", "currency", "CNY/HKD", 0.25),
        ("em_equity", "currency", "Other Emerging FX", 0.50),
        ("europe_equity", "country", "UK", 0.20), ("europe_equity", "country", "France", 0.15),
        ("europe_equity", "country", "Other Europe", 0.65),
        ("europe_equity", "sector", "Financials", 0.20), ("europe_equity", "sector", "Other Sectors", 0.80),
        ("europe_equity", "currency", "EUR", 0.50), ("europe_equity", "currency", "GBP", 0.20),
        ("europe_equity", "currency", "Other Europe FX", 0.30),
        ("small_cap", "country", "USA", 0.60), ("small_cap", "country", "Other Developed", 0.40),
        ("small_cap", "sector", "Industrials", 0.20), ("small_cap", "sector", "Other Sectors", 0.80),
        ("small_cap", "currency", "USD", 0.60), ("small_cap", "currency", "Other Developed FX", 0.40),
        ("tech_thematic", "country", "USA", 0.80), ("tech_thematic", "country", "Other Developed", 0.20),
        ("tech_thematic", "sector", "Technology", 0.80), ("tech_thematic", "sector", "Other Sectors", 0.20),
        ("tech_thematic", "currency", "USD", 0.80), ("tech_thematic", "currency", "Other Developed FX", 0.20),
        ("tech_thematic", "top_holding", "MSFT", 0.05, "Microsoft"),
        ("tech_thematic", "top_holding", "AAPL", 0.05, "Apple"),
        ("gold", "country", "N/A (Commodity)", 1.0), ("gold", "sector", "Commodities", 1.0),
        ("gold", "currency", "USD", 1.0),
        ("crypto", "country", "N/A (Crypto)", 1.0), ("crypto", "sector", "Crypto", 1.0),
        ("crypto", "currency", "USD", 1.0),
    ]
    df = pd.DataFrame(
        [row if len(row) == 5 else (*row, "") for row in rows],
        columns=["bucket", "dimension", "category", "weight", "label"],
    )
    df["source"] = "Example (rounded, illustrative) - replace with your fund's factsheet"
    df["as_of"] = ""
    df["note"] = ""
    return df[LOOKTHROUGH_COLUMNS]


def validate_bucket_lookthrough(df):
    """Raise on unknown buckets/dimensions or weights that don't add up."""
    missing_cols = {"bucket", "dimension", "category", "weight"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Look-through file lacks columns {sorted(missing_cols)}")
    unknown = set(df["bucket"]) - set(STRESS_BUCKETS)
    if unknown:
        raise ValueError(f"Unknown buckets {sorted(unknown)}. Valid: {sorted(STRESS_BUCKETS)}")
    bad_dims = set(df["dimension"]) - set(DIMENSIONS) - {"top_holding"}
    if bad_dims:
        raise ValueError(f"Unknown dimensions {sorted(bad_dims)}")

    sums = df.groupby(["bucket", "dimension"])["weight"].sum()
    for (bucket, dimension), total in sums.items():
        if dimension == "top_holding":
            if total > 1.0 + 1e-6:
                raise ValueError(f"{bucket} top holdings sum to {total:.4f} (> 1)")
        elif abs(total - 1.0) > 1e-6:
            raise ValueError(f"{bucket} {dimension} weights sum to {total:.4f}, not 1")


def load_bucket_lookthrough(path):
    """Read and validate the manual look-through CSV; seed the generic
    example if it doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        example_bucket_lookthrough().to_csv(path, index=False)
        warnings.warn(f"Seeded example look-through data at {path} — replace with real factsheet data.")
    df = pd.read_csv(path)
    validate_bucket_lookthrough(df)
    return df


def lookthrough_dicts(df):
    """(lookthrough, top_holdings) as nested dicts:
    lookthrough[bucket][dimension][category] = weight,
    top_holdings[bucket][stock] = weight."""
    lookthrough, top_holdings = {}, {}
    for row in df.itertuples(index=False):
        if row.dimension == "top_holding":
            top_holdings.setdefault(row.bucket, {})[row.category] = row.weight
        else:
            lookthrough.setdefault(row.bucket, {}).setdefault(row.dimension, {})[row.category] = row.weight
    return lookthrough, top_holdings


def top_holding_labels(df):
    """{ticker: company name} from the optional `label` column (top holdings)."""
    if "label" not in df.columns:
        return {}
    rows = df[(df["dimension"] == "top_holding") & df["label"].notna() & (df["label"] != "")]
    return dict(zip(rows["category"], rows["label"]))


def usd_shares(lookthrough):
    """{bucket: USD share} from the currency look-through, plus the
    single-stock buckets. Buckets without currency rows are left out."""
    shares = {
        bucket: dims["currency"].get("USD", 0.0)
        for bucket, dims in lookthrough.items()
        if "currency" in dims
    }
    for bucket, defaults in SINGLE_STOCK_BUCKETS.items():
        shares[bucket] = 1.0 if defaults["currency"] == "USD" else 0.0
    return shares


def lookthrough_weights(bucket, dimension, lookthrough, stock_info=None):
    """{category: weight} for one holding, or None if nothing is known."""
    if bucket in SINGLE_STOCK_BUCKETS:
        value = (stock_info or {}).get(dimension)
        if isinstance(value, str) and value:
            return {value: 1.0}
        if dimension == "sector":
            return None
        return {SINGLE_STOCK_BUCKETS[bucket][dimension]: 1.0}
    return lookthrough.get(bucket, {}).get(dimension)


def compute_lookthrough_exposure(holdings, ticker_buckets, dimension, lookthrough, stock_info=None):
    """Portfolio share per category (sums to 1). Holdings without usable
    look-through data land in 'Unclassified' — with a warning."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    stock_info = stock_info or {}

    rows, unclassified = [], []
    for _, row in holdings.iterrows():
        ticker, value = row["yf_ticker"], row["market_value"]
        weights = lookthrough_weights(
            ticker_buckets.get(ticker), dimension, lookthrough, stock_info.get(ticker)
        )
        if weights is None:
            unclassified.append(ticker)
            weights = {"Unclassified": 1.0}
        rows.extend({"category": c, "value": value * w} for c, w in weights.items())

    if unclassified:
        warnings.warn(
            f"No {dimension} look-through for: {', '.join(unclassified)} "
            "(check stress_bucket / sector in asset_universe_manual.csv and "
            "bucket_lookthrough_manual.csv)."
        )

    exposure = pd.DataFrame(rows).groupby("category")["value"].sum()
    return (exposure / holdings["market_value"].sum()).sort_values(ascending=False)


def compute_lookthrough_single_stock(holdings, ticker_buckets, top_holdings):
    """Direct vs. fund-implied (top holdings only) weight per stock."""
    total_value = holdings["market_value"].sum()
    direct, indirect = {}, {}

    for _, row in holdings.iterrows():
        ticker, value = row["yf_ticker"], row["market_value"]
        bucket = ticker_buckets.get(ticker)
        if bucket in SINGLE_STOCK_BUCKETS:
            direct[ticker] = direct.get(ticker, 0.0) + value
        for stock, w in top_holdings.get(bucket, {}).items():
            indirect[stock] = indirect.get(stock, 0.0) + value * w

    all_stocks = sorted(set(direct) | set(indirect))
    result = pd.DataFrame({
        "direct_weight": pd.Series({s: direct.get(s, 0.0) / total_value for s in all_stocks}),
        "indirect_weight": pd.Series({s: indirect.get(s, 0.0) / total_value for s in all_stocks}),
    })
    result["total_weight"] = result["direct_weight"] + result["indirect_weight"]
    result["hidden_multiple"] = result["total_weight"] / result["direct_weight"].replace(0, np.nan)
    return result.sort_values("total_weight", ascending=False)


def exposure_summary(holdings, ticker_buckets, single_stock):
    """Headline numbers for the exposure tab (all as portfolio shares):
    equity share incl. funds, directly held single stocks, and the same
    stocks including what the funds hold of them."""
    total = holdings["market_value"].sum()
    equity = holdings.loc[
        ~holdings["yf_ticker"].map(ticker_buckets).isin(NON_EQUITY_BUCKETS), "market_value"
    ].sum()
    held = single_stock[single_stock["direct_weight"] > 0]
    return {
        "equity_share": equity / total,
        "direct_single_stocks": held["direct_weight"].sum(),
        "direct_single_stocks_incl_funds": held["total_weight"].sum(),
        "n_direct_single_stocks": len(held),
    }


def market_beta(portfolio_level, benchmark_level, freq="W-FRI"):
    """Beta, correlation and R² of portfolio vs. benchmark returns.

    Weekly by default: daily returns understate beta when prices close at
    different times (Xetra benchmark vs. US stocks / crypto)."""
    levels = pd.DataFrame({"portfolio": portfolio_level, "benchmark": benchmark_level}).dropna()
    # actual data window (resampling would label the last week with a future Friday)
    start, end = levels.index.min(), levels.index.max()
    if freq is not None:
        levels = levels.resample(freq).last()
    returns = levels.pct_change().dropna()

    beta = returns.cov().loc["portfolio", "benchmark"] / returns["benchmark"].var()
    correlation = returns["portfolio"].corr(returns["benchmark"])
    return {
        "beta": beta,
        "correlation": correlation,
        "r_squared": correlation ** 2,
        "observations": len(returns),
        "start": start,
        "end": end,
    }
