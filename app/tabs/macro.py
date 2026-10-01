import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import explanations as ex
from app.formatting import clean_table

BANK_COLORS = {"Fed": "#1f77b4", "ECB": "#ff7f0e"}


def render(macro_sentiment, macro_topics, macro_summary, macro_policy_rates):
    st.header("Central Bank Policy Sentiment")
    st.caption(
        "LLM-based assessment (qwen3:4b) of Fed and ECB speeches. "
        "Only speeches classified as monetary policy are included. "
        "Score ranges from -1 (dovish, favors easing) to +1 (hawkish, favors tightening)."
    )

    if macro_sentiment.empty:
        st.info(
            "No macro sentiment data available. "
            "Run notebook 11 to generate it (needs a local Ollama model; skipped in demo mode)."
        )
        return

    df = macro_sentiment.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["stance_score"]).sort_values("date")

    if df.empty:
        st.info("No monetary policy speeches classified yet.")
        return

    banks = [b for b in ["Fed", "ECB"] if b in df["central_bank"].unique()]

    # --- KPI row (computed in notebook 11) ---------------------------------
    if not macro_summary.empty:
        summary = macro_summary.set_index("central_bank")
        days = int(summary["window_days"].iloc[0])
        as_of = pd.to_datetime(summary["as_of"].iloc[0])
        st.subheader(f"Current Stance (last {days} days to {as_of:%d.%m.%Y})")

        cols = st.columns(len(banks) + 1)
        for col, bank in zip(cols, banks):
            if bank not in summary.index:
                continue
            row = summary.loc[bank]
            if pd.isna(row["current_mean"]):
                col.metric(bank, "n/a", help="No monetary policy speech in this window")
                continue
            delta = None if pd.isna(row["previous_mean"]) else f"{row['current_mean'] - row['previous_mean']:+.2f}"
            col.metric(
                bank,
                f"{row['current_mean']:+.2f}",
                delta,
                help="Positive = hawkish, negative = dovish. Delta vs. the preceding window.",
            )

        cols[-1].metric(
            "Speeches Analyzed",
            f"{int(summary['n_current'].sum())}",
            help=f"Monetary policy speeches in the last {days} days "
            f"({int(summary['n_monetary_policy_total'].sum())} in total).",
        )

        freshness = " · ".join(
            f"{bank}: latest speech {pd.to_datetime(summary.loc[bank, 'latest_speech']):%d.%m.%Y}, "
            f"latest on monetary policy {pd.to_datetime(summary.loc[bank, 'latest_monetary_policy_speech']):%d.%m.%Y}"
            for bank in banks if bank in summary.index
        )
        st.caption(
            f"{freshness}. The ECB speech export is updated about once a month, "
            "so recent ECB speeches can be missing; slide decks without text are skipped."
        )

    st.divider()

    # --- Stance over time --------------------------------------------------
    st.subheader("Stance Over Time")
    st.caption(ex.stance_explanation(
        int(macro_summary["window_days"].iloc[0]) if not macro_summary.empty else None
    ))
    st.caption(
        "Dotted: actual policy rate (Fed funds target upper bound, ECB deposit rate) "
        "as a reference point — a hawkish lean should precede or accompany hikes."
    )

    fig = go.Figure()

    rates = macro_policy_rates.copy()
    if not rates.empty:
        rates["date"] = pd.to_datetime(rates["date"])
        start = df["date"].min()
        # the rate in force at the chart start, then all changes after it
        in_force = rates[rates["date"] < start].sort_values("date").groupby("central_bank").tail(1)
        rates = pd.concat([in_force.assign(date=start), rates[rates["date"] >= start]])

    for bank in banks:
        bank_df = df[df["central_bank"] == bank].set_index("date")

        fig.add_trace(
            go.Scatter(
                x=bank_df.index,
                y=bank_df["stance_rolling"],
                mode="lines+markers",
                name=f"{bank} (90-day avg)",
                line=dict(color=BANK_COLORS.get(bank), width=2),
                marker=dict(size=5),
            )
        )

        # single speeches as faint context behind the smoothed line
        fig.add_trace(
            go.Scatter(
                x=bank_df.index,
                y=bank_df["stance_score"],
                mode="markers",
                name=f"{bank} (single speeches)",
                marker=dict(size=5, color=BANK_COLORS.get(bank), opacity=0.25),
                showlegend=False,
                hovertext=bank_df.get("title"),
            )
        )

        bank_rates = rates[rates["central_bank"] == bank] if not rates.empty else rates
        if not bank_rates.empty:
            fig.add_trace(
                go.Scatter(
                    x=bank_rates["date"],
                    y=bank_rates["policy_rate"],
                    mode="lines",
                    line=dict(color=BANK_COLORS.get(bank), width=1.5, dash="dot", shape="hv"),
                    name=f"{bank} policy rate (%)",
                    yaxis="y2",
                )
            )

    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    # Symmetric around 0 but fitted to the actual spread: with a fixed [-1, 1]
    # the smoothed lines stick to the zero line, since scores rarely exceed |0.5|.
    span = max(0.3, float(df["stance_score"].abs().max()) * 1.15)

    fig.update_layout(
        yaxis_title="Stance (dovish … hawkish)",
        xaxis_title=None,
        yaxis_range=[-span, span],
        yaxis2=dict(title="Policy rate (%)", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    st.plotly_chart(fig, width="stretch")

    # --- Topics ------------------------------------------------------------
    if not macro_topics.empty:
        st.subheader("Dominant Topics")
        st.caption(
            "Keyword frequency across the extracted key topics. "
            "Topics are free-form LLM output, so this is indicative rather than exact."
        )

        topics = macro_topics.copy()

        available_banks = [b for b in banks if b in topics["central_bank"].unique()]
        selected = st.multiselect(
            "Central bank",
            options=available_banks,
            default=available_banks,
            key="macro_topic_banks",
        )

        if selected:
            topics = topics[topics["central_bank"].isin(selected)]

        topics = (
            topics.groupby("keyword", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(15)
        )

        if topics.empty:
            st.info("No topics for the current selection.")
        else:
            fig_topics = px.bar(
                topics.sort_values("count"),
                x="count",
                y="keyword",
                orientation="h",
                labels={"count": "Mentions", "keyword": ""},
            )
            fig_topics.update_layout(height=max(300, 26 * len(topics)))
            st.plotly_chart(fig_topics, width="stretch")

    # --- Recent speeches ---------------------------------------------------
    st.subheader("Recent Speeches")

    recent = df.sort_values("date", ascending=False).head(12).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["stance_score"] = recent["stance_score"].map(lambda v: f"{v:+.2f}")

    display_cols = {
        "date": "Date",
        "central_bank": "Bank",
        "speaker": "Speaker",
        "title": "Title",
        "stance_score": "Stance",
        "rationale": "Assessment",
    }

    # st.table wraps long text; st.dataframe would cut the assessments off
    st.table(
        clean_table(
            recent,
            columns=list(display_cols),
            rename_map=display_cols,
        ).set_index("Date")
    )
