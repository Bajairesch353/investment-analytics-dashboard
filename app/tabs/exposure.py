import pandas as pd
import plotly.express as px
import streamlit as st

from app.formatting import clean_table


def render(
    factor_single_stock,
    factor_country_exposure,
    factor_sector_exposure,
    factor_currency_exposure,
    factor_exposure_summary
):
    st.header("Look-Through Exposure")
    st.caption(
        "Fund compositions are a factsheet snapshot (sources and as-of dates in "
        "Notebook 10, Section 3), not live holdings data. Via funds only the "
        "largest constituents are counted."
    )

    if not factor_single_stock.empty:
        st.subheader("Single-Stock Exposure (Direct + via Funds)")

        stocks = factor_single_stock.reset_index().rename(columns={"index": "Ticker"})
        if "name" not in stocks.columns:
            stocks["name"] = stocks["Ticker"]

        for col in ["direct_weight", "indirect_weight", "total_weight"]:
            stocks[col] = stocks[col] * 100

        if not factor_exposure_summary.empty:
            summary = factor_exposure_summary["value"]
            n_direct = int(summary["n_direct_single_stocks"])
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Equity Share (incl. Funds)",
                f"{summary['equity_share'] * 100:.1f}%",
                help="Everything except the gold and crypto buckets.",
            )
            col2.metric(
                "Single Stocks Held Directly",
                f"{summary['direct_single_stocks'] * 100:.1f}%",
                help=f"{n_direct} directly held stocks, direct positions only.",
            )
            col3.metric(
                "Same Stocks incl. via Funds",
                f"{summary['direct_single_stocks_incl_funds'] * 100:.1f}%",
                f"+{(summary['direct_single_stocks_incl_funds'] - summary['direct_single_stocks']) * 100:.1f}pp",
                help="The directly held stocks plus what your funds hold of them "
                "(largest fund constituents only, so a lower bound).",
            )

        chart_data = stocks.melt(
            id_vars="name",
            value_vars=["direct_weight", "indirect_weight"],
            var_name="Source",
            value_name="Weight (%)"
        )
        chart_data["Source"] = chart_data["Source"].map({
            "direct_weight": "Held directly",
            "indirect_weight": "Held via funds"
        })

        fig = px.bar(
            chart_data,
            x="Weight (%)",
            y="name",
            color="Source",
            orientation="h",
            barmode="stack",
            title="Portfolio Weight per Stock: Direct vs. via Funds"
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=stocks.sort_values("total_weight")["name"].tolist()
        )
        fig.update_layout(
            height=max(420, 24 * len(stocks)),
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=20)
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Single-Stock Detail"):
            detail = stocks.copy()
            detail["hidden_multiple"] = detail["hidden_multiple"].apply(
                lambda v: "—" if pd.isna(v) else f"{v:.1f}x"
            )

            st.dataframe(
                clean_table(
                    detail,
                    columns=["name", "Ticker", "direct_weight", "indirect_weight",
                             "total_weight", "hidden_multiple"],
                    rename_map={
                        "name": "Stock",
                        "direct_weight": "Direct",
                        "indirect_weight": "Via Funds",
                        "total_weight": "Total",
                        "hidden_multiple": "Multiple"
                    },
                    pct_point_cols=["direct_weight", "indirect_weight", "total_weight"]
                ),
                width='stretch',
                hide_index=True
            )

    st.divider()

    if not factor_currency_exposure.empty:
        st.subheader("Currency Exposure")

        currency = factor_currency_exposure.copy()
        currency["Weight (%)"] = currency["value"] * 100

        fig = px.bar(
            currency.sort_values("Weight (%)"),
            x="Weight (%)",
            y="category",
            orientation="h",
            title="Effective Currency Exposure (Look-Through)"
        )
        fig.update_layout(
            height=420,
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=20)
        )
        st.plotly_chart(fig, width='stretch')

    st.divider()

    col_country, col_sector = st.columns(2)

    with col_country:
        if not factor_country_exposure.empty:
            country = factor_country_exposure.copy()
            country["Weight (%)"] = country["value"] * 100

            fig = px.bar(
                country.sort_values("Weight (%)"),
                x="Weight (%)",
                y="category",
                orientation="h",
                title="Country Exposure"
            )
            fig.update_layout(
                height=460,
                yaxis_title="",
                margin=dict(l=10, r=10, t=60, b=20)
            )
            st.plotly_chart(fig, width='stretch')

    with col_sector:
        if not factor_sector_exposure.empty:
            sector = factor_sector_exposure.copy()
            sector["Weight (%)"] = sector["value"] * 100

            fig = px.bar(
                sector.sort_values("Weight (%)"),
                x="Weight (%)",
                y="category",
                orientation="h",
                title="Sector Exposure"
            )
            fig.update_layout(
                height=460,
                yaxis_title="",
                margin=dict(l=10, r=10, t=60, b=20)
            )
            st.plotly_chart(fig, width='stretch')