import pandas as pd
import plotly.express as px
import streamlit as st

from app import explanations as ex
from app.formatting import clean_table, format_pct_fraction, format_pct_points


def render(
    portfolio_risk_metrics,
    portfolio_risk_timeseries,
    portfolio_concentration,
    portfolio_concentration_metrics,
    portfolio_risk_contribution,
    backtest_cumulative_growth,
    stress_test_scenario_summary,
    stress_test_loss_by_class,
    factor_market_beta
):
    ...
    if not portfolio_risk_timeseries.empty:
        st.subheader("Backtested Portfolio Performance")

        if "actual_portfolio_cum_return_pct" in portfolio_risk_timeseries.columns:
            performance_chart = portfolio_risk_timeseries[
                ["actual_portfolio_cum_return_pct"]
            ].rename(
                columns={"actual_portfolio_cum_return_pct": "Current Portfolio"}
            )

            if (
                not backtest_cumulative_growth.empty
                and "Benchmark (FTSE All-World)" in backtest_cumulative_growth.columns
            ):
                benchmark_cum_return_pct = (
                    backtest_cumulative_growth["Benchmark (FTSE All-World)"] - 1
                ) * 100

                performance_chart = performance_chart.join(
                    benchmark_cum_return_pct.rename("Benchmark (FTSE All-World)"),
                    how="left"
                )

            fig = px.line(
                performance_chart,
                x=performance_chart.index,
                y=performance_chart.columns.tolist(),
                title="Backtested Current Portfolio vs. Benchmark"
            )

            fig.update_layout(
                height=450,
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                legend_title_text="",
                margin=dict(l=10, r=10, t=60, b=20)
            )

            st.plotly_chart(fig, width='stretch')
            st.caption("All strategies, rebalancing and DCA: see the Backtest tab.")

    if not portfolio_risk_metrics.empty and "Current Portfolio" in portfolio_risk_metrics.index:
        current_metrics = portfolio_risk_metrics.loc["Current Portfolio"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Return p.a.",
            format_pct_points(current_metrics["Annualized Return"]),
            help=ex.RETURN_PA
        )

        col2.metric(
            "Volatility p.a.",
            format_pct_points(current_metrics["Annualized Volatility"]),
            help=ex.VOLATILITY_PA
        )

        col3.metric(
            "Sharpe Ratio",
            f"{current_metrics['Sharpe Ratio']:.2f}",
            help=ex.sharpe_help()
        )

        col4.metric(
            "Max Drawdown",
            format_pct_points(current_metrics["Max Drawdown"]),
            help=ex.MAX_DRAWDOWN
        )

    if not factor_market_beta.empty:
        col5, col6 = st.columns(2)
        col5.metric(
            "Market Beta",
            f"{float(factor_market_beta.loc['Market Beta'].iloc[0]):.2f}",
            help=ex.BETA
        )
        col6.metric(
            "R² vs. Benchmark",
            f"{float(factor_market_beta.loc['R-squared'].iloc[0]):.2f}",
            help=ex.R_SQUARED
        )
        st.caption(
            f"Measured against {factor_market_beta.loc['Benchmark'].iloc[0]} on weekly returns "
            f"(Notebook 10): current weights held static over "
            f"{factor_market_beta.loc['Window'].iloc[0]} "
            "— a hypothetical beta of today's portfolio, not the realized one."
        )

    st.divider()

    if not portfolio_risk_timeseries.empty:
        st.subheader("Drawdown")

        if "actual_drawdown" in portfolio_risk_timeseries.columns:
            drawdown_chart = (
                portfolio_risk_timeseries[["actual_drawdown"]] * 100
            ).rename(
                columns={
                    "actual_drawdown": "Current Portfolio"
                }
            )

            fig = px.line(
                drawdown_chart,
                x=drawdown_chart.index,
                y="Current Portfolio",
                title="Portfolio Drawdown"
            )

            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                margin=dict(l=10, r=10, t=60, b=20)
            )

            st.plotly_chart(fig, width='stretch')

        st.subheader("Rolling Volatility")

        if "rolling_vol_actual" in portfolio_risk_timeseries.columns:
            rolling_vol_chart = (
                portfolio_risk_timeseries[["rolling_vol_actual"]] * 100
            ).rename(
                columns={
                    "rolling_vol_actual": "Current Portfolio"
                }
            )

            fig = px.line(
                rolling_vol_chart,
                x=rolling_vol_chart.index,
                y="Current Portfolio",
                title="Rolling 3-Month Annualized Volatility"
            )

            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                margin=dict(l=10, r=10, t=60, b=20)
            )

            st.plotly_chart(fig, width='stretch')

    st.divider()

    if not portfolio_concentration_metrics.empty:
        st.subheader("Concentration")

        cm = portfolio_concentration_metrics["value"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("HHI", f"{cm.loc['HHI']:.3f}", help=ex.HHI)
        col2.metric(
            "Effective N Positions",
            f"{cm.loc['Effective Number of Positions']:.1f}",
            help=ex.EFFECTIVE_N
        )
        col3.metric("Top 3 Weight", format_pct_fraction(cm.loc["Top 3 Weight"]), help=ex.TOP_N)
        col4.metric("Top 5 Weight", format_pct_fraction(cm.loc["Top 5 Weight"]), help=ex.TOP_N)

    if not portfolio_concentration.empty and not portfolio_risk_contribution.empty:
        st.subheader("Weight vs Risk Contribution")
        st.caption(ex.RISK_CONTRIBUTION)

        risk_compare = portfolio_risk_contribution.merge(
            portfolio_concentration[["name", "actual_weight"]],
            on="name",
            how="left",
            suffixes=("_risk", "_portfolio")
        )

        weight_col = (
            "actual_weight_portfolio"
            if "actual_weight_portfolio" in risk_compare.columns
            else "actual_weight"
        )

        risk_compare["Portfolio Weight"] = risk_compare[weight_col] * 100
        risk_compare["Risk Contribution"] = risk_compare["risk_contribution_pct"] * 100

        risk_compare_long = risk_compare.melt(
            id_vars="name",
            value_vars=[
                "Portfolio Weight",
                "Risk Contribution"
            ],
            var_name="Metric",
            value_name="Percent"
        )

        fig = px.bar(
            risk_compare_long,
            x="Percent",
            y="name",
            color="Metric",
            orientation="h",
            barmode="group",
            title="Portfolio Weight vs Risk Contribution"
        )

        fig.update_layout(
            height=650,
            xaxis_title="Percent (%)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=20)
        )

        st.plotly_chart(fig, width='stretch')

        with st.expander("Risk Contribution Table"):
            risk_table = clean_table(
                risk_compare.sort_values("Risk Contribution", ascending=False),
                columns=[
                    "name",
                    "Portfolio Weight",
                    "Risk Contribution"
                ],
                rename_map={
                    "name": "Asset"
                },
                pct_point_cols=[
                    "Portfolio Weight",
                    "Risk Contribution"
                ]
            )

            st.dataframe(
                risk_table,
                width='stretch',
                hide_index=True
            )

    st.divider()


    if not stress_test_scenario_summary.empty:
        st.subheader("Stress Test")
        st.caption(
            "Notebook 09 — instantaneous shocks on today's holdings, not a forecast. "
            "Hypothetical scenarios are judgement-based what-ifs; historical ones replay each "
            "holding's actual EUR return over that window; the FX scenario is the pure "
            "currency-translation effect of a stronger EUR on the USD share."
        )

        scenario_plot_df = stress_test_scenario_summary.sort_values("total_loss_pct").reset_index()
        scenario_plot_df = scenario_plot_df.rename(columns={"index": "Scenario"})
        scenario_plot_df["Loss (%)"] = scenario_plot_df["total_loss_pct"] * 100

        scenario_fig = px.bar(
            scenario_plot_df,
            x="Loss (%)",
            y="Scenario",
            orientation="h",
            title="Portfolio Loss by Stress Scenario"
        )
        scenario_fig.update_layout(
            height=450,
            xaxis_title="Portfolio Loss (%)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=20)
        )
        st.plotly_chart(scenario_fig, width='stretch')

        if not stress_test_loss_by_class.empty:
            selected_scenario = st.selectbox(
                "Scenario Detail",
                scenario_plot_df["Scenario"].tolist()
            )

            class_detail = clean_table(
                stress_test_loss_by_class[stress_test_loss_by_class["scenario"] == selected_scenario],
                columns=["asset_class", "market_value", "loss_eur", "loss_pct_of_class"],
                rename_map={
                    "asset_class": "Asset Class",
                    "market_value": "Market Value",
                    "loss_eur": "Loss (EUR)",
                    "loss_pct_of_class": "Loss (%)"
                },
                euro_cols=["market_value", "loss_eur"],
                pct_fraction_cols=["loss_pct_of_class"]
            )

            st.dataframe(
                class_detail,
                width='stretch',
                hide_index=True
            )

