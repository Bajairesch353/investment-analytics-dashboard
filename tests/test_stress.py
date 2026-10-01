import pandas as pd
import pytest

from src.stress import (
    combine_worst_of,
    expand_bucket_scenario,
    fallback_stress_bucket,
    fx_translation_shocks,
    historical_shocks,
    resolve_stress_buckets,
    summarize_scenario,
)

HOLDINGS = pd.DataFrame({
    "yf_ticker": ["WORLD", "GOLD", "STOCK_US"],
    "asset_class": ["FUND", "FUND", "STOCK"],
    "market_value": [600.0, 200.0, 200.0],
})


def test_fallback_bucket_by_asset_class_and_region():
    assert fallback_stress_bucket("CRYPTO") == "crypto"
    assert fallback_stress_bucket("STOCK", "US") == "us_single_stock"
    assert fallback_stress_bucket("STOCK", "Europe") == "eu_single_stock"
    assert fallback_stress_bucket("FUND") == "global_equity"


def test_resolve_stress_buckets_prefers_explicit_and_warns_on_fallback():
    universe = pd.DataFrame({
        "yf_ticker": ["WORLD", "GOLD"],
        "stress_bucket": ["global_equity", "gold"],
    })
    with pytest.warns(UserWarning, match="STOCK_US"):
        buckets = resolve_stress_buckets(HOLDINGS, universe)
    assert buckets.to_dict() == {"WORLD": "global_equity", "GOLD": "gold", "STOCK_US": "us_single_stock"}


def test_resolve_stress_buckets_rejects_unknown_bucket():
    universe = pd.DataFrame({"yf_ticker": ["WORLD"], "stress_bucket": ["typo_bucket"]})
    with pytest.raises(ValueError):
        resolve_stress_buckets(HOLDINGS, universe)


def test_expand_bucket_scenario_defaults_to_zero():
    buckets = pd.Series({"WORLD": "global_equity", "GOLD": "gold"})
    assert expand_bucket_scenario({"global_equity": -0.3}, buckets) == {"WORLD": -0.3, "GOLD": 0.0}


def test_historical_shocks_compound_daily_returns():
    index = pd.to_datetime(["2022-01-03", "2022-01-04", "2023-01-02"])
    returns = pd.DataFrame({"a": [0.1, -0.1, 0.5]}, index=index)
    assert historical_shocks(returns, "2022-01-01", "2022-12-31")["a"] == pytest.approx(1.1 * 0.9 - 1)


def test_historical_shocks_empty_window_raises():
    returns = pd.DataFrame({"a": [0.1]}, index=pd.to_datetime(["2022-01-03"]))
    with pytest.raises(ValueError):
        historical_shocks(returns, "2020-01-01", "2020-12-31")


def test_fx_translation_scales_by_usd_share():
    buckets = pd.Series({"US": "us_single_stock", "WORLD": "global_equity", "EU": "europe_equity"})
    usd_shares = {"us_single_stock": 1.0, "global_equity": 0.7, "europe_equity": 0.0}
    shocks = fx_translation_shocks(0.15, buckets, usd_shares)
    usd_move = 1 / 1.15 - 1  # about -13.0%, not -15%
    assert shocks["US"] == pytest.approx(usd_move)
    assert shocks["WORLD"] == pytest.approx(usd_move * 0.7)
    assert shocks["EU"] == 0.0


def test_fx_translation_warns_on_missing_usd_share():
    buckets = pd.Series({"GOLD": "gold"})
    with pytest.warns(UserWarning, match="gold"):
        shocks = fx_translation_shocks(0.15, buckets, {})
    assert shocks["GOLD"] == 0.0


def test_combine_worst_of_takes_minimum_per_ticker():
    assert combine_worst_of({"a": -0.3, "b": 0.05}, {"b": -0.08}) == {"a": -0.3, "b": -0.08}


def test_summarize_scenario_totals_and_weights():
    summary, detail = summarize_scenario(HOLDINGS, {"WORLD": -0.5})
    assert summary["total_loss_eur"] == pytest.approx(-300)
    assert summary["total_loss_pct"] == pytest.approx(-0.3)
    assert detail["new_weight"].sum() == pytest.approx(1.0)
    assert detail.set_index("yf_ticker").loc["GOLD", "shock_pct"] == 0.0
