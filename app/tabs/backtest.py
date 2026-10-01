import pandas as pd
import plotly.express as px
import streamlit as st

from app import explanations as ex
from app.formatting import clean_table, format_eur, format_pct_fraction


def render(
    backtest_cumulative_growth,
    backtest_metrics_summary,
    backtest_rebalance_frequency,
    backtest_dca_summary,
    optimization_efficient_frontier,
    optimization_expected_metrics
):
    """Historical what-ifs (Notebook 07) and ex-ante optimization (Notebook 08)."""
    if not backtest_cumulative_growth.empty:
        st.subheader("Backtested Strategies")

        performance_chart = (backtest_cumulative_growth - 1) * 100

        if not performance_chart.empty:
            fig = px.line(
                performance_chart,
                x=performance_chart.index,
                y=performance_chart.columns.tolist(),
                title="Backtested Strategies vs. Benchmark"
            )

            # In-sample strategies (today's scores applied to the past) are
            # an upper bound, not an achievable path — draw them dashed.
            for trace in fig.data:
                if "In-Sample" in trace.name:
                    trace.update(line=dict(dash="dash"))

            fig.update_layout(
                height=450,
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                legend_title_text="",
                margin=dict(l=10, r=10, t=60, b=20)
            )

            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Today's weights applied to the full history (Notebook 07) — a what-if, "
                "not a real track record. The dashed selection-score line uses today's "
                "scores retroactively (look-ahead bias): an upper bound, not a strategy."
            )

    if not backtest_metrics_summary.empty:
        st.subheader("Strategy Comparison")

        strategy_table = clean_table(
            backtest_metrics_summary.reset_index().rename(columns={"index": "Strategy"}),
            columns=[
                "Strategy",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Max Drawdown",
                "Recovery Time (Days)"
            ],
            rename_map={
                "Annualized Return": "Return p.a.",
                "Annualized Volatility": "Volatility p.a.",
                "Sharpe Ratio": "Sharpe",
                "Sortino Ratio": "Sortino",
                "Calmar Ratio": "Calmar",
                "Recovery Time (Days)": "Recovery (Days)"
            },
            pct_fraction_cols=[
                "Annualized Return",
                "Annualized Volatility",
                "Max Drawdown"
            ],
            round_cols=[
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio"
            ]
        )

        if "Recovery (Days)" in strategy_table.columns:
            strategy_table["Recovery (Days)"] = strategy_table["Recovery (Days)"].round().astype("Int64")

        st.dataframe(strategy_table, width='stretch', hide_index=True)
        st.caption(ex.strategy_metrics_explanation())

    if not backtest_rebalance_frequency.empty:
        st.subheader("Rebalancing Frequency")

        frequency_table = clean_table(
            backtest_rebalance_frequency.reset_index(),
            columns=[
                "Strategy",
                "Annualized Return",
                "Sharpe Ratio",
                "Max Drawdown",
                "Average Turnover",
                "Total Turnover"
            ],
            rename_map={
                "Strategy": "Frequency",
                "Annualized Return": "Return p.a.",
                "Sharpe Ratio": "Sharpe",
                "Average Turnover": "Avg. Turnover per Rebalance",
                "Total Turnover": "Total Turnover"
            },
            pct_fraction_cols=[
                "Annualized Return",
                "Max Drawdown",
                "Average Turnover",
                "Total Turnover"
            ],
            round_cols=["Sharpe Ratio"]
        )

        st.dataframe(frequency_table, width='stretch', hide_index=True)
        st.caption(ex.REBALANCING_FREQUENCY)

    if not backtest_dca_summary.empty:
        st.subheader("DCA Simulation")

        dca = backtest_dca_summary["value"]
        simulated_amount = float(dca.loc["monthly_amount"])

        monthly_amount = st.number_input(
            "Monthly amount (€)",
            min_value=10.0,
            value=simulated_amount,
            step=50.0,
            format="%.0f"
        )

        # Every contribution is split the same way and compounds with the same
        # returns, so invested amount and final value scale linearly with the
        # monthly amount — no re-simulation needed. XIRR is unchanged.
        scale = monthly_amount / simulated_amount

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Invested", format_eur(float(dca.loc["total_invested"]) * scale))
        col2.metric(
            "Final Value",
            format_eur(float(dca.loc["final_value"]) * scale),
            format_eur(float(dca.loc["gain"]) * scale)
        )
        col3.metric("XIRR", format_pct_fraction(float(dca.loc["xirr"])))
        st.caption(
            f"Monthly contribution into the target weights since {dca.loc['start_date']} "
            "(Notebook 07). XIRR = money-weighted annual return — independent of the amount."
        )

    st.divider()

    if not optimization_efficient_frontier.empty and not optimization_expected_metrics.empty:
        st.subheader("Efficient Frontier")
        st.caption(
            "Ex-ante (Notebook 08) — expected return/volatility from historical mu/covariance, "
            "not backtested. The frontier respects the same position and asset-class caps as the "
            "optimized portfolios (drawn up to Max-Sharpe — beyond it the caps only add risk). "
            "Equal Weight ignores the caps, so it can sit above the frontier."
        )

        frontier = optimization_efficient_frontier.copy()
        if "Efficient" not in frontier.columns:
            frontier["Efficient"] = True

        # Add the min-variance point to both branches so they meet without a gap.
        if "Min-Variance" in optimization_expected_metrics.index:
            min_var_point = optimization_expected_metrics.loc[
                ["Min-Variance"], ["Expected Return", "Expected Volatility"]
            ]
            frontier = pd.concat([
                frontier,
                min_var_point.assign(Efficient=True),
                min_var_point.assign(Efficient=False)
            ], ignore_index=True)

        frontier = frontier.drop_duplicates().sort_values("Expected Return")
        efficient_branch = frontier[frontier["Efficient"]]

        # Drawn up to the Max-Sharpe portfolio: beyond it the position caps bind
        # and the frontier only adds volatility for almost no extra return.
        if "Max-Sharpe" in optimization_expected_metrics.index:
            max_sharpe_point = optimization_expected_metrics.loc[
                ["Max-Sharpe"], ["Expected Return", "Expected Volatility"]
            ]
            efficient_branch = pd.concat([
                efficient_branch[
                    efficient_branch["Expected Return"] < max_sharpe_point["Expected Return"].iloc[0]
                ],
                max_sharpe_point.assign(Efficient=True)
            ], ignore_index=True)
        inefficient_branch = frontier[~frontier["Efficient"]]

        frontier_fig = px.line(
            efficient_branch,
            x="Expected Volatility",
            y="Expected Return"
        )
        frontier_fig.update_traces(name="Efficient Frontier", showlegend=True)

        if len(inefficient_branch) > 1:
            inefficient_fig = px.line(
                inefficient_branch,
                x="Expected Volatility",
                y="Expected Return"
            )
            inefficient_fig.update_traces(
                name="Inefficient branch",
                showlegend=True,
                line=dict(dash="dash"),
                opacity=0.5
            )
            frontier_fig.add_trace(inefficient_fig.data[0])

        strategy_points = optimization_expected_metrics.reset_index().rename(columns={"index": "Strategy"})
        # Strategies that land on the same point (e.g. both Max-Sharpe variants
        # at the same corner of the caps) would hide each other: merge them
        # into one point named "A = B".
        coords = strategy_points[["Expected Volatility", "Expected Return"]].round(6)
        strategy_points = (
            strategy_points.assign(_vol=coords["Expected Volatility"], _ret=coords["Expected Return"])
            .groupby(["_vol", "_ret"], sort=False, as_index=False)
            .agg({"Strategy": " = ".join, "Expected Volatility": "first", "Expected Return": "first"})
        )
        scatter_fig = px.scatter(
            strategy_points,
            x="Expected Volatility",
            y="Expected Return",
            color="Strategy"
        )
        scatter_fig.update_traces(marker=dict(size=11))

        for trace in scatter_fig.data:
            frontier_fig.add_trace(trace)

        frontier_fig.update_layout(
            height=500,
            title="Efficient Frontier vs. Current / Target / Optimized Portfolios",
            xaxis_title="Expected Volatility (annualized)",
            yaxis_title="Expected Return (annualized)",
            margin=dict(l=10, r=10, t=60, b=20)
        )

        st.plotly_chart(frontier_fig, width='stretch')
