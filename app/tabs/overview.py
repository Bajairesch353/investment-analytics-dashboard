import plotly.express as px
import streamlit as st

from app.formatting import color_change, format_eur, format_pct_fraction


def render(portfolio_kpis, asset_universe_overview, portfolio_dashboard, monthly_transactions):
    st.header("Portfolio Overview")

    if not portfolio_kpis.empty:
        kpis = portfolio_kpis.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Portfolio Value",
            format_eur(kpis["total_portfolio_value_eur"])
        )

        col2.metric(
            "Funds",
            format_pct_fraction(kpis["fund_weight"])
        )

        col3.metric(
            "Stocks",
            format_pct_fraction(kpis["stock_weight"])
        )

        col4.metric(
            "Crypto",
            format_pct_fraction(kpis["crypto_weight"])
        )

    st.divider()

    if not monthly_transactions.empty:
        st.subheader("Net Cash Flow")
        st.caption("Last completed months")

        max_months = len(monthly_transactions)
        n_months = st.slider(
            "Months shown",
            min_value=1,
            max_value=max_months,
            value=min(12, max_months),
            label_visibility="collapsed"
        )

        recent_cash_flow = monthly_transactions.tail(n_months)

        fig_cash_flow = px.bar(
            recent_cash_flow,
            x="month",
            y="net_cash_flow",
            labels={"month": "Month", "net_cash_flow": "Net Cash Flow"}
        )
        fig_cash_flow.update_traces(
            hovertemplate="%{x}<br>%{y:,.2f} €<extra></extra>"
        )
        fig_cash_flow.update_layout(
            height=300,
            xaxis_title="",
            yaxis_title="€",
            margin=dict(l=10, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_cash_flow, width='stretch')

    st.divider()

    st.subheader("Asset Universe / Watchlist")

    if not asset_universe_overview.empty:
        watchlist_table = asset_universe_overview[
            [
                "name",
                "current_price_eur",
                "one_day_change"
            ]
        ].copy()

        watchlist_table = watchlist_table.rename(
            columns={
                "name": "Asset",
                "current_price_eur": "Price",
                "one_day_change": "1D Change"
            }
        )

        watchlist_table["1D Change"] = watchlist_table["1D Change"] * 100

        styled_watchlist = (
            watchlist_table
            .style
            .format({
                "Price": "{:,.2f} €",
                "1D Change": "{:+.2f}%"
            })
            .map(
                color_change,
                subset=["1D Change"]
            )
        )

        row_height = 35
        header_height = 38
        table_height = header_height + row_height * len(watchlist_table)

        st.dataframe(
            styled_watchlist,
            width='stretch',
            hide_index=True,
            height=table_height
        )
        
    else:
        st.info("No asset universe overview available yet.")

    st.divider()

    st.subheader("Largest Holdings")

    if not portfolio_dashboard.empty:
        largest_holdings = (
            portfolio_dashboard
            .sort_values("actual_weight", ascending=False)
            .head(10)
            .copy()
        )

        largest_holdings["Weight (%)"] = largest_holdings["actual_weight"] * 100

        fig = px.bar(
            largest_holdings.sort_values("Weight (%)", ascending=True),
            x="Weight (%)",
            y="name",
            orientation="h",
            text="Weight (%)",
            title="Largest Holdings by Portfolio Weight"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=520,
            xaxis_title="Portfolio Weight (%)",
            yaxis_title="",
            margin=dict(l=10, r=40, t=60, b=20)
        )

        st.plotly_chart(fig, width='stretch')

