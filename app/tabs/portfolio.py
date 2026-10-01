import plotly.express as px
import streamlit as st

from app import explanations as ex
from app.formatting import clean_table, format_asset_class, format_eur
from src.portfolio import allocate_buy_only_suggestion


def render(portfolio_dashboard, actual_vs_target, dca_summary):
    st.header("Portfolio")

    if not portfolio_dashboard.empty:
        st.subheader("Asset Allocation")

        allocation = (
            portfolio_dashboard
            .assign(asset_class_label=portfolio_dashboard["asset_class"].apply(format_asset_class))
            .groupby("asset_class_label", as_index=False)["market_value"]
            .sum()
            .sort_values("market_value", ascending=False)
        )

        fig = px.pie(
            allocation,
            names="asset_class_label",
            values="market_value",
            hole=0.45,
            title="Asset Allocation"
        )

        fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=60, b=10)
        )

        st.plotly_chart(fig, width='stretch')

        st.subheader("Current Holdings")

        portfolio_display = portfolio_dashboard.copy()

        if "asset_class" in portfolio_display.columns:
            portfolio_display["asset_class"] = portfolio_display["asset_class"].apply(format_asset_class)

        holdings_table = clean_table(
            portfolio_display.sort_values("market_value", ascending=False),
            columns=[
                "name",
                "asset_class",
                "shares",
                "current_price_eur",
                "market_value",
                "actual_weight",
                "unrealized_pnl_simple",
                "unrealized_pnl_pct_simple"
            ],
            rename_map={
                "name": "Asset",
                "asset_class": "Class",
                "shares": "Shares",
                "current_price_eur": "Price",
                "market_value": "Market Value",
                "actual_weight": "Weight",
                "unrealized_pnl_simple": "Unrealized P&L",
                "unrealized_pnl_pct_simple": "Unrealized P&L %"
            },
            euro_cols=[
                "current_price_eur",
                "market_value",
                "unrealized_pnl_simple"
            ],
            pct_fraction_cols=[
                "actual_weight",
                "unrealized_pnl_pct_simple"
            ],
            round_cols=["shares"]
        )

        st.dataframe(
            holdings_table,
            width='stretch',
            hide_index=True
        )

    st.divider()

    if not actual_vs_target.empty:
        st.subheader("Actual vs Target Weights")
        st.caption(ex.drift_explanation())

        drift_chart = actual_vs_target.copy()
        drift_chart["Drift"] = drift_chart["weight_drift_pp"]

        fig = px.bar(
            drift_chart.sort_values("Drift", ascending=True),
            x="Drift",
            y="name",
            orientation="h",
            text="Drift",
            title="Weight Drift vs Target"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=560,
            xaxis_title="Drift (%)",
            yaxis_title="",
            margin=dict(l=10, r=40, t=60, b=20)
        )

        st.plotly_chart(fig, width='stretch')

        drift_table = clean_table(
            actual_vs_target.sort_values("weight_drift_pp", ascending=False),
            columns=[
                "name",
                "actual_weight",
                "target_weight",
                "weight_drift_pp",
                "rebalance_action"
            ],
            rename_map={
                "name": "Asset",
                "actual_weight": "Current Weight",
                "target_weight": "Target Weight",
                "weight_drift_pp": "Drift",
                "rebalance_action": "Action"
            },
            pct_fraction_cols=["actual_weight", "target_weight"],
            pp_cols=["weight_drift_pp"]
        )

        st.dataframe(
            drift_table,
            width='stretch',
            hide_index=True
        )

        underweight = actual_vs_target[
            actual_vs_target["rebalance_action"] == "Add / buy more"
        ].copy()

        if not underweight.empty:
            st.subheader("Rebalancing Suggestions")

            cash_to_invest = st.number_input(
                "Cash to invest (€)",
                min_value=0.0,
                value=300.0,
                step=50.0
            )

            underweight["underweight_amount"] = -underweight["value_drift"]
            underweight["underweight_share"], underweight["suggested_buy_amount"] = allocate_buy_only_suggestion(
                underweight["underweight_amount"],
                cash_to_invest
            )
            underweight = underweight.sort_values("suggested_buy_amount", ascending=False)

            unallocated_cash = cash_to_invest - underweight["suggested_buy_amount"].sum()
            if unallocated_cash > 0.01:
                st.caption(
                    f"{format_eur(unallocated_cash)} not allocated to any position — "
                    "the underweight positions are already fully covered at their target weight."
                )

            rebalancing_table = clean_table(
                underweight,
                columns=[
                    "name",
                    "market_value",
                    "actual_weight",
                    "target_weight",
                    "rebalance_action",
                    "suggested_buy_amount"
                ],
                rename_map={
                    "name": "Asset",
                    "market_value": "Market Value",
                    "actual_weight": "Current Weight",
                    "target_weight": "Target Weight",
                    "rebalance_action": "Action",
                    "suggested_buy_amount": "Amount"
                },
                euro_cols=["market_value", "suggested_buy_amount"],
                pct_fraction_cols=["actual_weight", "target_weight"]
            )

            st.dataframe(
                rebalancing_table,
                width='stretch',
                hide_index=True
            )

    st.divider()

    with st.expander("Savings Plan / DCA Details"):
        st.caption(ex.dca_explanation())
        if not dca_summary.empty:
            dca_table = clean_table(
                dca_summary.sort_values("total_invested", ascending=False),
                columns=[
                    "name",
                    "total_invested",
                    "n_orders",
                    "active_months",
                    "regularity_rate",
                    "likely_dca_asset"
                ],
                rename_map={
                    "name": "Asset",
                    "total_invested": "Total Invested",
                    "n_orders": "Orders",
                    "active_months": "Active Months",
                    "regularity_rate": "Regularity",
                    "likely_dca_asset": "Likely DCA"
                },
                euro_cols=["total_invested"],
                pct_fraction_cols=["regularity_rate"]
            )

            st.dataframe(
                dca_table,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No DCA data available.")

