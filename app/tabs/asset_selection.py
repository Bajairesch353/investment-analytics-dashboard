import plotly.express as px
import streamlit as st

from app import explanations as ex
from app.formatting import clean_table


RATING_COLORS = {
    "Strong fundamentals": "#2ca02c",
    "Average fundamentals": "#8c8c8c",
    "Weak fundamentals": "#d62728",
}


def _dimension_scatter(df, x, y, title, x_title, y_title):
    """Scatter of two fundamental dimension scores (0-100), colored by rating.

    Percentile scores of a small universe often tie, so several stocks can sit
    on the same point: better ratings are drawn last (on top) and stocks on
    one point share a single label."""
    data = df.dropna(subset=[x, y]).copy()
    data["label"] = data.groupby([x, y])["yf_ticker"].transform(", ".join)
    # one label per point, on the trace drawn last (best rating on top)
    rank = data["rating"].map({r: i for i, r in enumerate(RATING_COLORS)}).fillna(len(RATING_COLORS))
    data["label"] = data["label"].where(rank == rank.groupby([data[x], data[y]]).transform("min"), "")
    draw_order = list(RATING_COLORS)[::-1]  # weak first, strong last = on top
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color="rating",
        color_discrete_map=RATING_COLORS,
        category_orders={"rating": draw_order},
        text="label",
        hover_name="name",
        hover_data={"fundamental_score": ":.0f", "label": False},
        title=title
    )
    fig.add_hline(y=50, line_dash="dot", line_color="grey")
    fig.add_vline(x=50, line_dash="dot", line_color="grey")
    fig.update_traces(textposition="top center", marker=dict(size=12))
    fig.update_layout(
        height=420,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_range=[-5, 105],
        yaxis_range=[-5, 105],
        legend_title_text="",
        legend_traceorder="reversed",  # legend still lists Strong first
        margin=dict(l=10, r=10, t=60, b=20)
    )
    return fig


def render(
    asset_selection_decision_summary,
    asset_selection_ranking,
    asset_selection_review_table,
    asset_selection_candidate_table,
    fundamental_scoring
):
    st.header("Asset Selection")

    if not asset_selection_decision_summary.empty:
        st.subheader("Decision Summary")
        st.caption(ex.decision_explanation())

        decision_summary = asset_selection_decision_summary.copy()

        decision_col = None

        for candidate_col in [
            "decision_bucket",
            "selection_decision",
            "decision",
            "recommendation",
            "bucket"
        ]:
            if candidate_col in decision_summary.columns:
                decision_col = candidate_col
                break

        if decision_col is not None:
            decision_table = clean_table(
                decision_summary,
                columns=[
                    decision_col,
                    "n_assets",
                    "avg_selection_score",
                    "avg_current_weight",
                    "total_current_weight"
                ],
                rename_map={
                    decision_col: "Decision",
                    "n_assets": "Assets",
                    "avg_selection_score": "Avg. Score",
                    "avg_current_weight": "Avg. Weight",
                    "total_current_weight": "Total Weight"
                },
                pct_fraction_cols=[
                    "avg_current_weight",
                    "total_current_weight"
                ],
                round_cols=["avg_selection_score"]
            )
        else:
            decision_table = clean_table(
                decision_summary,
                columns=decision_summary.columns.tolist(),
                rename_map={
                    "n_assets": "Assets",
                    "avg_selection_score": "Avg. Score",
                    "avg_current_weight": "Avg. Weight",
                    "total_current_weight": "Total Weight"
                },
                pct_fraction_cols=[
                    "avg_current_weight",
                    "total_current_weight"
                ],
                round_cols=["avg_selection_score"]
            )

        st.dataframe(
            decision_table,
            width='stretch',
            hide_index=True
        )

    if not asset_selection_ranking.empty:
        st.subheader("Selection Ranking")
        st.caption(ex.selection_score_explanation())

        ranking = asset_selection_ranking.copy()

        score_col = None

        for candidate_col in [
            "selection_score",
            "score",
            "total_score"
        ]:
            if candidate_col in ranking.columns:
                score_col = candidate_col
                break

        name_col = "name" if "name" in ranking.columns else "symbol"

        if score_col is not None and name_col in ranking.columns:
            ranking_chart = ranking.sort_values(
                score_col,
                ascending=False
            ).head(15)

            fig = px.bar(
                ranking_chart.sort_values(score_col, ascending=True),
                x=score_col,
                y=name_col,
                orientation="h",
                title="Top Asset Selection Scores"
            )

            fig.update_layout(
                height=560,
                xaxis_title="Selection Score",
                yaxis_title="",
                margin=dict(l=10, r=10, t=60, b=20)
            )

            st.plotly_chart(
                fig,
                width='stretch'
            )

        ranking_table = clean_table(
            ranking.head(50),
            columns=[
                "symbol",
                "name",
                "asset_class",
                "selection_score",
                "decision_bucket",
                "current_weight",
                "target_weight",
                "review_reason"
            ],
            rename_map={
                "symbol": "Symbol",
                "name": "Asset",
                "asset_class": "Class",
                "selection_score": "Score",
                "decision_bucket": "Decision",
                "current_weight": "Current Weight",
                "target_weight": "Target Weight",
                "review_reason": "Reason"
            },
            pct_fraction_cols=[
                "current_weight",
                "target_weight"
            ],
            round_cols=["selection_score"]
        )

        st.dataframe(
            ranking_table,
            width='stretch',
            hide_index=True
        )

    else:
        st.info("No asset selection ranking available.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Review Assets")

        if not asset_selection_review_table.empty:
            review_table = clean_table(
                asset_selection_review_table,
                columns=[
                    "symbol",
                    "name",
                    "asset_class",
                    "selection_score",
                    "current_weight",
                    "review_reason"
                ],
                rename_map={
                    "symbol": "Symbol",
                    "name": "Asset",
                    "asset_class": "Class",
                    "selection_score": "Score",
                    "current_weight": "Current Weight",
                    "review_reason": "Reason"
                },
                pct_fraction_cols=["current_weight"],
                round_cols=["selection_score"]
            )

            st.dataframe(
                review_table,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No review assets available.")

    with col2:
        st.subheader("Candidate Assets")

        if not asset_selection_candidate_table.empty:
            candidate_table = clean_table(
                asset_selection_candidate_table,
                columns=[
                    "symbol",
                    "name",
                    "asset_class",
                    "selection_score",
                    "candidate_reason"
                ],
                rename_map={
                    "symbol": "Symbol",
                    "name": "Asset",
                    "asset_class": "Class",
                    "selection_score": "Score",
                    "candidate_reason": "Reason"
                },
                round_cols=["selection_score"]
            )

            st.dataframe(
                candidate_table,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No candidate assets available.")

    st.divider()

    st.subheader("Fundamentals (Individual Stocks)")

    if not fundamental_scoring.empty:
        st.caption(
            ex.fundamental_score_explanation()
            + "  \nFunds and crypto have no comparable fundamentals and are excluded."
        )

        fundamentals_table = clean_table(
            fundamental_scoring,
            columns=[
                "yf_ticker",
                "name",
                "role",
                "source",
                "trailing_pe",
                "price_to_book",
                "return_on_equity",
                "revenue_growth",
                "debt_to_equity",
                "current_ratio",
                "valuation_score",
                "quality_score",
                "growth_score",
                "health_score",
                "fundamental_score",
                "rating"
            ],
            rename_map={
                "yf_ticker": "Ticker",
                "name": "Stock",
                "role": "Role",
                "source": "Source",
                "trailing_pe": "P/E",
                "price_to_book": "P/B",
                "return_on_equity": "ROE",
                "revenue_growth": "Rev. Growth (3y CAGR)",
                "debt_to_equity": "Debt/Equity",
                "current_ratio": "Current Ratio",
                "valuation_score": "Valuation",
                "quality_score": "Quality",
                "growth_score": "Growth",
                "health_score": "Health",
                "fundamental_score": "Fundamental Score",
                "rating": "Rating"
            },
            pct_fraction_cols=["return_on_equity", "revenue_growth"],
            pp_cols=[
                "valuation_score",
                "quality_score",
                "growth_score",
                "health_score",
                "fundamental_score"
            ],
            round_cols=[
                "trailing_pe",
                "price_to_book",
                "debt_to_equity",
                "current_ratio"
            ]
        )

        st.dataframe(
            fundamentals_table,
            width='stretch',
            hide_index=True
        )

        fundamentals_chart = fundamental_scoring.dropna(subset=["fundamental_score"])

        fig = px.bar(
            fundamentals_chart.sort_values("fundamental_score", ascending=True),
            x="fundamental_score",
            y="name",
            orientation="h",
            title="Fundamental Score by Stock"
        )
        fig.update_layout(
            height=max(300, 32 * len(fundamentals_chart)),
            xaxis_title="Fundamental Score (%)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=60, b=20)
        )
        st.plotly_chart(fig, width='stretch')

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(
                _dimension_scatter(
                    fundamentals_chart, "valuation_score", "quality_score",
                    "Valuation vs. Quality",
                    "Valuation Score (%, higher = cheaper)", "Quality Score (%)"
                ),
                width='stretch'
            )
        with col_right:
            st.plotly_chart(
                _dimension_scatter(
                    fundamentals_chart, "growth_score", "health_score",
                    "Growth vs. Health",
                    "Growth Score (%)", "Health Score (%, balance sheet)"
                ),
                width='stretch'
            )

    else:
        st.info("No fundamental scoring data available.")

