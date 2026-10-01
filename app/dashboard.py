import os
import sys
from pathlib import Path

# `streamlit run app/dashboard.py` only puts this file's own directory
# (app/) on sys.path, not the project root — so `app` itself isn't
# importable as a package yet. Add the project root explicitly, same
# fix the notebooks use via sys.path.append("..").
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.formatting import load_csv
from app.tabs import asset_selection, backtest, cashflows, exposure, macro, newsletter, overview, portfolio, risk
from src.paths import DATA_DIR, PROJECT_ROOT

# Newsletters: $NEWSLETTER_OUTPUT_DIR (same variable as the newsletter agent),
# else <data dir>/newsletter if present (the demo's example newsletter),
# else the agent's default output folder.
_demo_newsletters = DATA_DIR / "newsletter"
NEWSLETTER_PATH = Path(
    os.environ.get(
        "NEWSLETTER_OUTPUT_DIR",
        _demo_newsletters if _demo_newsletters.is_dir()
        else PROJECT_ROOT / "newsletter-agent" / "Newsletter_output"
    )
)

st.set_page_config(
    page_title="Personal Investment Dashboard",
    layout="wide"
)

st.title("Personal Investment Dashboard")

portfolio_kpis = load_csv("portfolio_kpis.csv")
portfolio_dashboard = load_csv("portfolio_dashboard.csv")
actual_vs_target = load_csv("actual_vs_target_weights.csv")

monthly_transactions = load_csv("monthly_transaction_dashboard.csv")
dca_summary = load_csv("dca_summary.csv")

asset_selection_ranking = load_csv("asset_selection_ranking.csv")
asset_selection_ranking_display = load_csv("asset_selection_ranking_display.csv")
asset_selection_decision_summary = load_csv("asset_selection_decision_summary.csv")
asset_selection_review_table = load_csv("asset_selection_review_table.csv")
asset_selection_candidate_table = load_csv("asset_selection_candidate_table.csv")
fundamental_scoring = load_csv("fundamental_scoring_display.csv")

consumption_kpis = load_csv("consumption_kpis.csv")
consumption_monthly = load_csv("consumption_monthly_summary.csv")
consumption_category_monthly = load_csv("consumption_category_monthly.csv")
consumption_category_summary = load_csv("consumption_category_summary.csv")
consumption_merchant_summary = load_csv("consumption_merchant_summary.csv")
consumption_large_transactions = load_csv("consumption_large_transactions.csv")
consumption_final_unclear = load_csv("consumption_final_unclear_summary.csv")

portfolio_risk_metrics = load_csv(
    "portfolio_risk_performance_metrics_display.csv",
    index_col=0
)

portfolio_risk_timeseries = load_csv(
    "portfolio_risk_timeseries.csv",
    index_col=0,
    parse_dates=True
)

portfolio_concentration = load_csv("portfolio_concentration.csv")
portfolio_concentration_metrics = load_csv("portfolio_concentration_metrics.csv", index_col=0)
portfolio_risk_contribution = load_csv("portfolio_risk_contribution.csv")
asset_universe_overview = load_csv("asset_universe_overview.csv")
backtest_cumulative_growth = load_csv(
    "backtest_cumulative_growth.csv",
    index_col=0,
    parse_dates=True
)
backtest_metrics_summary = load_csv("backtest_metrics_summary.csv", index_col=0)
backtest_rebalance_frequency = load_csv("backtest_rebalance_frequency_summary.csv", index_col=0)
backtest_dca_summary = load_csv("backtest_dca_summary.csv", index_col=0)

optimization_efficient_frontier = load_csv("optimization_efficient_frontier.csv")
optimization_expected_metrics = load_csv("optimization_expected_metrics.csv", index_col=0)
stress_test_scenario_summary = load_csv("stress_test_scenario_summary.csv", index_col=0)
stress_test_loss_by_class = load_csv("stress_test_loss_by_class.csv")

factor_single_stock = load_csv("factor_single_stock_exposure.csv", index_col=0)
factor_country_exposure = load_csv("factor_country_exposure.csv")
factor_sector_exposure = load_csv("factor_sector_exposure.csv")
factor_currency_exposure = load_csv("factor_currency_exposure.csv")
factor_market_beta = load_csv("factor_market_beta.csv", index_col=0)
factor_exposure_summary = load_csv("factor_exposure_summary.csv", index_col=0)

macro_sentiment = load_csv("macro_sentiment_timeseries.csv", optional=True)
macro_topics = load_csv("macro_sentiment_topics.csv", optional=True)
macro_summary = load_csv("macro_sentiment_summary.csv", optional=True)
macro_policy_rates = load_csv("macro_policy_rates.csv", optional=True)

tab_overview, tab_portfolio, tab_selection, tab_cash, tab_exposure, tab_risk, tab_backtest, tab_macro, tab_newsletter = st.tabs(
    [
        "Overview",
        "Portfolio",
        "Asset Selection",
        "Cashflows",
        "Exposure",
        "Risk & Performance",
        "Backtest",
        "Macro",
        "Newsletter"
    ]
)


with tab_overview:
    overview.render(
        portfolio_kpis,
        asset_universe_overview,
        portfolio_dashboard,
        monthly_transactions
    )

with tab_portfolio:
    portfolio.render(
        portfolio_dashboard,
        actual_vs_target,
        dca_summary
    )

with tab_selection:
    asset_selection.render(
        asset_selection_decision_summary,
        asset_selection_ranking,
        asset_selection_review_table,
        asset_selection_candidate_table,
        fundamental_scoring
    )

with tab_cash:
    cashflows.render(
        monthly_transactions,
        consumption_monthly,
        consumption_category_monthly,
        consumption_merchant_summary,
        consumption_large_transactions,
        consumption_final_unclear
    )

with tab_exposure:
    exposure.render(
        factor_single_stock,
        factor_country_exposure,
        factor_sector_exposure,
        factor_currency_exposure,
        factor_exposure_summary
    )

with tab_risk:
    risk.render(
        portfolio_risk_metrics,
        portfolio_risk_timeseries,
        portfolio_concentration,
        portfolio_concentration_metrics,
        portfolio_risk_contribution,
        backtest_cumulative_growth,
        stress_test_scenario_summary,
        stress_test_loss_by_class,
        factor_market_beta
    )

with tab_backtest:
    backtest.render(
        backtest_cumulative_growth,
        backtest_metrics_summary,
        backtest_rebalance_frequency,
        backtest_dca_summary,
        optimization_efficient_frontier,
        optimization_expected_metrics
    )

with tab_macro:
    macro.render(
        macro_sentiment,
        macro_topics,
        macro_summary,
        macro_policy_rates
    )

with tab_newsletter:
    newsletter.render(NEWSLETTER_PATH)