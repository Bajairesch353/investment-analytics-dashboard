import pandas as pd
import plotly.express as px
import streamlit as st

from app import explanations as ex
from app.formatting import clean_table, format_eur


def render(
    monthly_transactions,
    consumption_monthly,
    consumption_category_monthly,
    consumption_merchant_summary,
    consumption_large_transactions,
    consumption_final_unclear
):
    st.header("Cashflows")
    st.caption(ex.CASHFLOWS)

    if not monthly_transactions.empty:
        monthly = monthly_transactions.copy()

        if "month" in monthly.columns:
            monthly["month"] = monthly["month"].astype(str)

        max_months = min(24, len(monthly))

        n_months = st.slider(
            "Months shown",
            min_value=1,
            max_value=max_months,
            value=min(12, max_months),
            key="cash_months_shown"
        )

        monthly_recent = monthly.tail(n_months).copy()

        chart_cols = [
            col for col in [
                "net_cash_flow",
                "consumption",
                "cash_inflow",
                "buy_volume"
            ]
            if col in monthly_recent.columns
        ]

        if "month" in monthly_recent.columns and chart_cols:
            chart_df = monthly_recent[["month"] + chart_cols].copy()

            chart_long = chart_df.melt(
                id_vars="month",
                value_vars=chart_cols,
                var_name="Category",
                value_name="Amount"
            )

            label_map = {
                "net_cash_flow": "Net Cash Flow",
                "consumption": "Consumption",
                "cash_inflow": "Cash Inflow",
                "buy_volume": "Investment Volume (Buys)"
            }

            chart_long["Category"] = chart_long["Category"].map(label_map)

            fig = px.line(
                chart_long,
                x="month",
                y="Amount",
                color="Category",
                markers=True,
                title="Monthly Cashflows"
            )

            fig.update_layout(
                height=500,
                xaxis_title="Month",
                yaxis_title="Amount (€)",
                margin=dict(l=10, r=10, t=60, b=20)
            )

            st.plotly_chart(
                fig,
                width='stretch'
            )

        cash_table = clean_table(
            monthly_recent,
            columns=[
                "month",
                "net_cash_flow",
                "consumption",
                "cash_inflow",
                "buy_volume",
                "passive_income"
            ],
            rename_map={
                "month": "Month",
                "net_cash_flow": "Net Cash Flow",
                "consumption": "Consumption",
                "cash_inflow": "Cash Inflow",
                "buy_volume": "Investment Volume (Buys)",
                "passive_income": "Passive Income"
            },
            euro_cols=[
                "net_cash_flow",
                "consumption",
                "cash_inflow",
                "buy_volume",
                "passive_income"
            ]
        )

        st.dataframe(
            cash_table,
            width='stretch',
            hide_index=True
        )

    else:
        st.info("No monthly cashflow data available.")

    st.divider()
    st.header("Consumption Analytics")

    if not consumption_monthly.empty:
        consumption_monthly_display = consumption_monthly.copy()

        if "month" in consumption_monthly_display.columns:
            consumption_monthly_display["month"] = (
                consumption_monthly_display["month"].astype(str)
            )

        max_consumption_months = min(
            24,
            len(consumption_monthly_display)
        )

        consumption_months_shown = st.slider(
            "Consumption months shown",
            min_value=1,
            max_value=max_consumption_months,
            value=min(6, max_consumption_months),
            key="consumption_months_shown"
        )

        consumption_recent = consumption_monthly_display.tail(
            consumption_months_shown
        ).copy()

        col1, col2, col3, col4 = st.columns(4)

        total_recent_consumption = consumption_recent["total_spending"].sum()
        avg_monthly_consumption = consumption_recent["total_spending"].mean()
        avg_transaction = consumption_recent["avg_transaction"].mean()
        total_transactions = consumption_recent["n_transactions"].sum()

        col1.metric("Total Consumption", format_eur(total_recent_consumption))
        col2.metric("Avg. Monthly Spending", format_eur(avg_monthly_consumption))
        col3.metric("Avg. Transaction", format_eur(avg_transaction))
        col4.metric("Transactions", f"{int(total_transactions)}")

        fig = px.line(
            consumption_recent,
            x="month",
            y="total_spending",
            markers=True,
            title="Monthly Consumption Spending"
        )

        fig.update_layout(
            height=450,
            xaxis_title="Month",
            yaxis_title="Spending (€)",
            margin=dict(l=10, r=10, t=60, b=20)
        )

        st.plotly_chart(fig, width='stretch')

        consumption_monthly_table = clean_table(
            consumption_recent,
            columns=[
                "month",
                "total_spending",
                "n_transactions",
                "avg_transaction",
                "median_transaction",
                "n_merchants",
                "n_categories"
            ],
            rename_map={
                "month": "Month",
                "total_spending": "Total Spending",
                "n_transactions": "Transactions",
                "avg_transaction": "Avg. Transaction",
                "median_transaction": "Median Transaction",
                "n_merchants": "Merchants",
                "n_categories": "Categories"
            },
            euro_cols=[
                "total_spending",
                "avg_transaction",
                "median_transaction"
            ]
        )

        st.dataframe(
            consumption_monthly_table,
            width='stretch',
            hide_index=True
        )

        st.subheader("Category Mix")

        if not consumption_category_monthly.empty:
            category_monthly = consumption_category_monthly.copy()
            category_monthly["month"] = category_monthly["month"].astype(str)

            available_months = sorted(
                category_monthly["month"].dropna().unique().tolist()
            )

            if available_months:
                category_mix_mode = st.radio(
                    "Category mix period",
                    options=[
                        "Single month",
                        "Month range"
                    ],
                    horizontal=True,
                    key="category_mix_period"
                )

                if category_mix_mode == "Single month":
                    selected_month = st.selectbox(
                        "Select month",
                        options=available_months,
                        index=len(available_months) - 1,
                        key="category_mix_single_month"
                    )

                    selected_months_for_mix = [selected_month]
                    category_mix_title = selected_month

                else:
                    col_start, col_end = st.columns(2)

                    with col_start:
                        start_month = st.selectbox(
                            "Start month",
                            options=available_months,
                            index=max(0, len(available_months) - consumption_months_shown),
                            key="category_mix_start_month"
                        )

                    with col_end:
                        end_month = st.selectbox(
                            "End month",
                            options=available_months,
                            index=len(available_months) - 1,
                            key="category_mix_end_month"
                        )

                    start_idx = available_months.index(start_month)
                    end_idx = available_months.index(end_month)

                    if start_idx > end_idx:
                        st.warning("Start month is after end month. Please adjust the range.")
                        selected_months_for_mix = []
                        category_mix_title = "Invalid range"
                    else:
                        selected_months_for_mix = available_months[
                            start_idx:end_idx + 1
                        ]
                        category_mix_title = f"{start_month} to {end_month}"

                if selected_months_for_mix:
                    category_mix = category_monthly[
                        category_monthly["month"].isin(selected_months_for_mix)
                    ].copy()

                    category_mix_summary = (
                        category_mix
                        .groupby("spending_category", dropna=False)
                        .agg(
                            total_spending=("monthly_spending", "sum"),
                            n_transactions=("n_transactions", "sum")
                        )
                        .reset_index()
                        .sort_values("total_spending", ascending=False)
                    )

                    top_n = 8

                    if len(category_mix_summary) > top_n:
                        top_mix = category_mix_summary.head(top_n).copy()

                        other_mix = pd.DataFrame(
                            [
                                {
                                    "spending_category": "Other categories",
                                    "total_spending": category_mix_summary.iloc[top_n:]["total_spending"].sum(),
                                    "n_transactions": category_mix_summary.iloc[top_n:]["n_transactions"].sum()
                                }
                            ]
                        )

                        category_mix_summary = pd.concat(
                            [
                                top_mix,
                                other_mix
                            ],
                            ignore_index=True
                        )

                    fig = px.pie(
                        category_mix_summary,
                        names="spending_category",
                        values="total_spending",
                        title=f"Category Mix: {category_mix_title}"
                    )

                    fig.update_traces(
                        textposition="inside",
                        textinfo="percent+label"
                    )

                    fig.update_layout(
                        height=500,
                        margin=dict(l=10, r=10, t=60, b=20)
                    )

                    st.plotly_chart(
                        fig,
                        width='stretch'
                    )

                    category_mix_table = clean_table(
                        category_mix_summary,
                        columns=[
                            "spending_category",
                            "total_spending",
                            "n_transactions"
                        ],
                        rename_map={
                            "spending_category": "Category",
                            "total_spending": "Total Spending",
                            "n_transactions": "Transactions"
                        },
                        euro_cols=["total_spending"]
                    )

                    st.dataframe(
                        category_mix_table,
                        width='stretch',
                        hide_index=True
                    )

            else:
                st.info("No category months available.")

        else:
            st.info("No category monthly data available.")

        st.subheader("Monthly Spending by Category")

        if not consumption_category_monthly.empty:
            category_trend = consumption_category_monthly.copy()
            category_trend["month"] = category_trend["month"].astype(str)

            recent_months = consumption_recent["month"].astype(str).tolist()

            category_trend_recent = category_trend[
                category_trend["month"].isin(recent_months)
            ].copy()

            if not category_trend_recent.empty:
                top_categories = (
                    category_trend_recent
                    .groupby("spending_category")["monthly_spending"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(6)
                    .index
                    .tolist()
                )

                category_trend_recent = category_trend_recent[
                    category_trend_recent["spending_category"].isin(top_categories)
                ]

                fig = px.line(
                    category_trend_recent,
                    x="month",
                    y="monthly_spending",
                    color="spending_category",
                    markers=True,
                    title="Monthly Spending by Top Categories"
                )

                fig.update_layout(
                    height=500,
                    xaxis_title="Month",
                    yaxis_title="Spending (€)",
                    margin=dict(l=10, r=10, t=60, b=20)
                )

                st.plotly_chart(fig, width='stretch')

            else:
                st.info("No category trend data for selected months.")

        st.subheader("Top Merchants")

        if not consumption_merchant_summary.empty:
            top_merchants = consumption_merchant_summary.sort_values(
                "total_spending",
                ascending=False
            ).head(15)

            merchant_table = clean_table(
                top_merchants,
                columns=[
                    "merchant_final",
                    "spending_category",
                    "fixed_variable",
                    "total_spending",
                    "n_transactions",
                    "avg_transaction"
                ],
                rename_map={
                    "merchant_final": "Merchant",
                    "spending_category": "Category",
                    "fixed_variable": "Fixed / Variable",
                    "total_spending": "Total Spending",
                    "n_transactions": "Transactions",
                    "avg_transaction": "Avg. Transaction"
                },
                euro_cols=[
                    "total_spending",
                    "avg_transaction"
                ]
            )

            st.dataframe(
                merchant_table,
                width='stretch',
                hide_index=True
            )

        with st.expander("Large Transactions"):
            if not consumption_large_transactions.empty:
                large_table = clean_table(
                    consumption_large_transactions,
                    columns=[
                        "date",
                        "merchant_final",
                        "spending_category",
                        "fixed_variable",
                        "spending_amount",
                        "description"
                    ],
                    rename_map={
                        "date": "Date",
                        "merchant_final": "Merchant",
                        "spending_category": "Category",
                        "fixed_variable": "Fixed / Variable",
                        "spending_amount": "Amount",
                        "description": "Description"
                    },
                    euro_cols=["spending_amount"]
                )

                st.dataframe(
                    large_table,
                    width='stretch',
                    hide_index=True
                )

            else:
                st.info("No large transactions available.")

        with st.expander("Remaining Unclear Merchants"):
            if not consumption_final_unclear.empty:
                unclear_table = clean_table(
                    consumption_final_unclear,
                    columns=[
                        "merchant_clean",
                        "total_spending",
                        "n_transactions",
                        "example_descriptions"
                    ],
                    rename_map={
                        "merchant_clean": "Merchant Clean",
                        "total_spending": "Total Spending",
                        "n_transactions": "Transactions",
                        "example_descriptions": "Examples"
                    },
                    euro_cols=["total_spending"]
                )

                st.dataframe(
                    unclear_table,
                    width='stretch',
                    hide_index=True
                )

            else:
                st.success("No relevant unclear merchants left.")

    else:
        st.info("No consumption analytics data available.")


