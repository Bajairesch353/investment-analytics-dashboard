import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    ECONOMIC_TYPE_MAP,
    allocate_buy_only_suggestion,
    classify_economic_type,
    classify_rebalance_action,
    compute_avg_cost_simple,
    compute_unrealized_pnl,
    is_likely_dca_asset,
)


def test_classify_economic_type_known_types():
    assert classify_economic_type("BUY") == "Investment"
    assert classify_economic_type("DIVIDEND") == "Passive Income"
    assert classify_economic_type("CARD_TRANSACTION") == "Consumption"
    assert classify_economic_type("TRANSFER_OUTBOUND") == "Cash Outflow"


def test_classify_economic_type_unknown_type_is_none():
    assert classify_economic_type("SOME_NEW_TYPE") is None


def test_economic_type_map_has_no_duplicate_semantics_drift():
    # sanity check that the map still covers investment / consumption sides
    assert ECONOMIC_TYPE_MAP["SELL"] == "Investment"
    assert ECONOMIC_TYPE_MAP["TRANSFER_DIRECT_DEBIT_INBOUND"] == "Consumption"


def test_is_likely_dca_asset_true_for_regular_investor():
    assert is_likely_dca_asset(
        n_orders=12,
        active_months=12,
        total_months=12,
        regularity_rate=1.0,
    )


def test_is_likely_dca_asset_false_for_one_off_purchase():
    assert not is_likely_dca_asset(
        n_orders=1,
        active_months=1,
        total_months=1,
        regularity_rate=1.0,
    )


def test_is_likely_dca_asset_false_when_irregular():
    assert not is_likely_dca_asset(
        n_orders=6,
        active_months=6,
        total_months=12,
        regularity_rate=0.3,
    )


def test_is_likely_dca_asset_vectorized_over_series():
    result = is_likely_dca_asset(
        n_orders=pd.Series([12, 1]),
        active_months=pd.Series([12, 1]),
        total_months=pd.Series([12, 1]),
        regularity_rate=pd.Series([1.0, 1.0]),
    )
    assert result.tolist() == [True, False]


def test_compute_avg_cost_simple_normal_case():
    result = compute_avg_cost_simple(
        total_bought=np.array([1000.0]),
        shares=np.array([10.0]),
    )
    assert result[0] == pytest.approx(100.0)


def test_compute_avg_cost_simple_near_zero_shares_is_nan():
    result = compute_avg_cost_simple(
        total_bought=np.array([1000.0]),
        shares=np.array([0.0]),
    )
    assert np.isnan(result[0])


def test_compute_unrealized_pnl():
    pnl, pnl_pct = compute_unrealized_pnl(market_value=1200.0, cost_basis=1000.0)
    assert pnl == pytest.approx(200.0)
    assert pnl_pct == pytest.approx(0.2)


def test_compute_unrealized_pnl_loss():
    pnl, pnl_pct = compute_unrealized_pnl(market_value=800.0, cost_basis=1000.0)
    assert pnl == pytest.approx(-200.0)
    assert pnl_pct == pytest.approx(-0.2)


def test_classify_rebalance_action_thresholds():
    assert classify_rebalance_action(3.0, threshold=2.0) == "Reduce / no new buys"
    assert classify_rebalance_action(-3.0, threshold=2.0) == "Add / buy more"
    assert classify_rebalance_action(1.0, threshold=2.0) == "Within target range"


def test_classify_rebalance_action_at_exact_threshold_is_within_range():
    # np.select used ">"/"<" (strict), not ">="/"<=" -- boundary stays "within range"
    assert classify_rebalance_action(2.0, threshold=2.0) == "Within target range"
    assert classify_rebalance_action(-2.0, threshold=2.0) == "Within target range"


def test_allocate_buy_only_suggestion_proportional_split():
    underweight = pd.Series([100.0, 300.0])
    share, suggested = allocate_buy_only_suggestion(underweight, cash_to_invest=200.0)

    assert share.tolist() == pytest.approx([0.25, 0.75])
    assert suggested.tolist() == pytest.approx([50.0, 150.0])
    assert suggested.sum() == pytest.approx(200.0)


def test_allocate_buy_only_suggestion_caps_at_own_underweight_amount():
    # cash_to_invest (1000) far exceeds total underweight (400) -- without
    # capping this would suggest buying past each position's target weight.
    underweight = pd.Series([100.0, 300.0])
    share, suggested = allocate_buy_only_suggestion(underweight, cash_to_invest=1000.0)

    assert suggested.tolist() == pytest.approx([100.0, 300.0])
    assert suggested.sum() == pytest.approx(400.0)


def test_allocate_buy_only_suggestion_no_cap_when_cash_is_scarce():
    # sanity check: capping must not kick in when cash is still the binding
    # constraint (same case as the proportional-split test above).
    underweight = pd.Series([100.0, 300.0])
    _, suggested = allocate_buy_only_suggestion(underweight, cash_to_invest=200.0)

    assert (suggested <= underweight).all()
