import numpy as np
import pandas as pd
import pytest

from src.exposure import (
    SINGLE_STOCK_BUCKETS,
    compute_lookthrough_exposure,
    compute_lookthrough_single_stock,
    example_bucket_lookthrough,
    exposure_summary,
    load_bucket_lookthrough,
    lookthrough_dicts,
    lookthrough_weights,
    top_holding_labels,
    market_beta,
    usd_shares,
    validate_bucket_lookthrough,
)
from src.stress import STRESS_BUCKETS

LOOKTHROUGH, TOP_HOLDINGS = lookthrough_dicts(example_bucket_lookthrough())

HOLDINGS = pd.DataFrame({
    "yf_ticker": ["WORLD", "GOLD", "STOCK_US"],
    "market_value": [600.0, 200.0, 200.0],
})
BUCKETS = {"WORLD": "global_equity", "GOLD": "gold", "STOCK_US": "us_single_stock"}
STOCK_INFO = {"STOCK_US": {"sector": "Technology"}}


def test_example_lookthrough_is_valid():
    validate_bucket_lookthrough(example_bucket_lookthrough())


def test_example_covers_every_fund_bucket():
    fund_buckets = set(STRESS_BUCKETS) - set(SINGLE_STOCK_BUCKETS)
    assert set(LOOKTHROUGH) == fund_buckets
    for bucket in fund_buckets:
        assert set(LOOKTHROUGH[bucket]) == {"country", "sector", "currency"}, bucket


def test_validate_rejects_bad_sums_and_unknown_buckets():
    df = example_bucket_lookthrough()
    broken = df.copy()
    broken.loc[broken["category"] == "USA", "weight"] = 0.5
    with pytest.raises(ValueError, match="global_equity country"):
        validate_bucket_lookthrough(broken)

    unknown = df.copy()
    unknown.loc[0, "bucket"] = "my_secret_fund"
    with pytest.raises(ValueError, match="Unknown buckets"):
        validate_bucket_lookthrough(unknown)


def test_load_seeds_example_when_missing(tmp_path):
    path = tmp_path / "bucket_lookthrough_manual.csv"
    with pytest.warns(UserWarning, match="Seeded example"):
        df = load_bucket_lookthrough(path)
    assert path.exists()
    assert df["source"].str.startswith("Example").all()


def test_usd_shares_from_currency_rows_and_single_stock_buckets():
    shares = usd_shares(LOOKTHROUGH)
    assert shares["global_equity"] == pytest.approx(0.70)
    assert shares["em_equity"] == 0.0
    assert shares["gold"] == 1.0
    assert shares["us_single_stock"] == 1.0
    assert shares["eu_single_stock"] == 0.0


def test_single_stock_weights_use_sector_and_country_override():
    assert lookthrough_weights("us_single_stock", "sector", LOOKTHROUGH, {"sector": "Health Care"}) == {"Health Care": 1.0}
    assert lookthrough_weights("us_single_stock", "country", LOOKTHROUGH) == {"USA": 1.0}
    assert lookthrough_weights("eu_single_stock", "country", LOOKTHROUGH, {"country": "France"}) == {"France": 1.0}
    assert lookthrough_weights("eu_single_stock", "currency", LOOKTHROUGH) == {"EUR": 1.0}
    assert lookthrough_weights("us_single_stock", "sector", LOOKTHROUGH, {"sector": np.nan}) is None
    assert lookthrough_weights("small_cap", "country", {}) is None  # no rows for it


def test_currency_exposure_sums_to_one_and_splits_funds():
    exposure = compute_lookthrough_exposure(HOLDINGS, BUCKETS, "currency", LOOKTHROUGH, STOCK_INFO)
    assert exposure.sum() == pytest.approx(1.0)
    assert exposure["USD"] == pytest.approx(0.6 * 0.70 + 0.2 + 0.2)


def test_missing_lookthrough_is_unclassified_with_warning():
    buckets = {**BUCKETS, "STOCK_US": None}
    with pytest.warns(UserWarning, match="STOCK_US"):
        exposure = compute_lookthrough_exposure(HOLDINGS, buckets, "sector", LOOKTHROUGH)
    assert exposure["Unclassified"] == pytest.approx(0.2)


def test_single_stock_direct_plus_indirect():
    holdings = pd.DataFrame({"yf_ticker": ["WORLD", "AAPL"], "market_value": [900.0, 100.0]})
    result = compute_lookthrough_single_stock(
        holdings, {"WORLD": "global_equity", "AAPL": "us_single_stock"}, TOP_HOLDINGS
    )
    aapl_in_world = TOP_HOLDINGS["global_equity"]["AAPL"]
    assert result.loc["AAPL", "direct_weight"] == pytest.approx(0.1)
    assert result.loc["AAPL", "indirect_weight"] == pytest.approx(0.9 * aapl_in_world)
    assert result.loc["AAPL", "hidden_multiple"] == pytest.approx(1 + 9 * aapl_in_world)
    # stocks held only via funds have no multiple
    assert np.isnan(result.loc["MSFT", "hidden_multiple"])
    assert result.loc["MSFT", "direct_weight"] == 0.0


def test_market_beta_recovers_known_beta():
    idx = pd.bdate_range("2024-01-01", periods=600)
    rng = np.random.default_rng(0)
    bench_ret = rng.normal(0, 0.01, len(idx))
    bench = pd.Series(np.cumprod(1 + bench_ret), index=idx)
    port = pd.Series(np.cumprod(1 + 1.5 * bench_ret), index=idx)

    daily = market_beta(port, bench, freq=None)
    assert daily["beta"] == pytest.approx(1.5, rel=1e-6)
    assert daily["r_squared"] == pytest.approx(1.0)
    assert daily["observations"] == len(idx) - 1

    weekly = market_beta(port, bench)
    assert weekly["beta"] == pytest.approx(1.5, rel=0.02)
    assert weekly["observations"] < daily["observations"]


def test_market_beta_window_is_actual_data_range():
    idx = pd.bdate_range("2024-01-01", "2024-03-27")  # ends on a Wednesday
    level = pd.Series(np.linspace(1, 2, len(idx)), index=idx)
    result = market_beta(level * 1.1, level)
    assert result["start"] == idx[0]
    assert result["end"] == idx[-1]


def test_top_holding_labels_from_example():
    labels = top_holding_labels(example_bucket_lookthrough())
    assert labels["AAPL"] == "Apple"
    assert top_holding_labels(example_bucket_lookthrough().drop(columns="label")) == {}


def test_exposure_summary_equity_share_and_direct_stocks():
    holdings = pd.DataFrame({
        "yf_ticker": ["WORLD", "GOLD", "AAPL"],
        "market_value": [700.0, 200.0, 100.0],
    })
    buckets = {"WORLD": "global_equity", "GOLD": "gold", "AAPL": "us_single_stock"}
    single = compute_lookthrough_single_stock(holdings, buckets, TOP_HOLDINGS)
    summary = exposure_summary(holdings, buckets, single)
    assert summary["equity_share"] == pytest.approx(0.8)
    assert summary["direct_single_stocks"] == pytest.approx(0.1)
    aapl_in_world = TOP_HOLDINGS["global_equity"]["AAPL"]
    assert summary["direct_single_stocks_incl_funds"] == pytest.approx(0.1 + 0.7 * aapl_in_world)
    assert summary["n_direct_single_stocks"] == 1

