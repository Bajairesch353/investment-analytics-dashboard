"""Portfolio construction / rebalancing helpers.

Extracted from notebooks/01_transactions_cashflows.ipynb and
notebooks/02_portfolio_holdings_tracker.ipynb.
"""

import numpy as np

# Maps raw Trade Republic transaction "type" values to a coarser economic
# grouping used for cashflow analysis. Extracted verbatim from
# notebooks/01_transactions_cashflows.ipynb.
ECONOMIC_TYPE_MAP = {
    # Investments
    "BUY": "Investment",
    "SELL": "Investment",

    # Passive
    "DIVIDEND": "Passive Income",
    "INTEREST_PAYMENT": "Passive Income",

    # Consumption
    "CARD_TRANSACTION": "Consumption",
    "CARD_TRANSACTION_INTERNATIONAL": "Consumption",
    "TRANSFER_DIRECT_DEBIT_INBOUND": "Consumption",

    # Cash inflow
    "TRANSFER_INBOUND": "Cash Inflow",
    "CUSTOMER_INBOUND": "Cash Inflow",
    "CUSTOMER_INPAYMENT": "Cash Inflow",

    # Cash outflow
    "TRANSFER_OUTBOUND": "Cash Outflow",
    "TRANSFER_INSTANT_OUTBOUND": "Cash Outflow",
    "CUSTOMER_OUTBOUND_REQUEST": "Cash Outflow",

    # Rewards
    "BENEFITS_SAVEBACK": "Rewards",

    # Tax
    "EARNINGS": "Tax",
    "TAX_OPTIMIZATION": "Tax",

    # Corporate
    "SPLIT": "Corporate Action",
    "MERGER": "Corporate Action",
    "EXCHANGE": "Corporate Action"
}


def classify_economic_type(transaction_type):
    return ECONOMIC_TYPE_MAP.get(transaction_type)


# Thresholds mirror notebooks/01_transactions_cashflows.ipynb's DCA
# heuristic: an asset "looks like" a savings plan if it was bought often
# enough, in enough distinct months, over a long enough span, at a
# regular-enough cadence.
DCA_MIN_ORDERS = 6
DCA_MIN_ACTIVE_MONTHS = 6
DCA_MIN_TOTAL_MONTHS = 12
DCA_MIN_REGULARITY_RATE = 0.45


def is_likely_dca_asset(
    n_orders,
    active_months,
    total_months,
    regularity_rate,
    min_orders=DCA_MIN_ORDERS,
    min_active_months=DCA_MIN_ACTIVE_MONTHS,
    min_total_months=DCA_MIN_TOTAL_MONTHS,
    min_regularity=DCA_MIN_REGULARITY_RATE
):
    return (
        (n_orders >= min_orders)
        & (active_months >= min_active_months)
        & (total_months >= min_total_months)
        & (regularity_rate >= min_regularity)
    )


def compute_avg_cost_simple(total_bought, shares, epsilon=1e-8):
    return np.where(
        shares > epsilon,
        total_bought / shares,
        np.nan
    )


def compute_unrealized_pnl(market_value, cost_basis):
    pnl = market_value - cost_basis
    pnl_pct = pnl / cost_basis
    return pnl, pnl_pct


# Default drift threshold (percentage points) used to flag a position for
# rebalancing, from notebooks/02_portfolio_holdings_tracker.ipynb.
DEFAULT_DRIFT_THRESHOLD_PP = 2.0


def classify_rebalance_action(weight_drift_pp, threshold=DEFAULT_DRIFT_THRESHOLD_PP):
    if weight_drift_pp > threshold:
        return "Reduce / no new buys"

    if weight_drift_pp < -threshold:
        return "Add / buy more"

    return "Within target range"


def allocate_buy_only_suggestion(underweight_amount, cash_to_invest):
    """Proportionally allocate new cash across underweight positions.

    Each position is capped at its own underweight_amount -- otherwise more
    cash than needed to close the gaps would keep pushing suggestions past
    the target weight without bound. Any cash left over after capping is
    simply not allocated (the caller can compare against cash_to_invest to
    report it).

    underweight_amount: array-like/Series of positive underweight amounts.
    Returns (underweight_share, suggested_buy_amount) matching the input shape.
    """
    total_underweight_amount = underweight_amount.sum()

    underweight_share = underweight_amount / total_underweight_amount
    uncapped_buy_amount = underweight_share * cash_to_invest
    suggested_buy_amount = np.minimum(uncapped_buy_amount, underweight_amount)

    return underweight_share, suggested_buy_amount
