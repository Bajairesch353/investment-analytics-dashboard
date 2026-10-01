"""Portfolio and reader-profile configuration for the newsletter agent.

Reads target weights and Yahoo tickers from the same gitignored CSVs the
notebook pipeline uses (`data/manual/`) instead of hardcoding the portfolio.
This keeps newsletter, dashboard and rebalancing logic consistent, and the
tracked code contains no real positions.

If the CSVs are missing (e.g. a fresh clone of the public repo), a generic
example portfolio is used.
"""

import os
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Same data directory as the notebook pipeline (src/paths.py): data/ by
# default, or INVESTMENT_DATA_DIR, e.g. the bundled sample data.
DATA_DIR = Path(os.environ.get("INVESTMENT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
MANUAL_DIR = DATA_DIR / "manual"
PROCESSED_DIR = DATA_DIR / "processed"
# Where newsletters are written (make_newsletter.sh) and the previous one is read from
OUTPUT_DIR = Path(
    os.environ.get("NEWSLETTER_OUTPUT_DIR", PROJECT_ROOT / "newsletter-agent" / "Newsletter_output")
).resolve()

TARGET_WEIGHTS_CSV = MANUAL_DIR / "target_weights_manual.csv"
TICKER_MAP_CSV = MANUAL_DIR / "ticker_map_manual.csv"
# Optional: yf_ticker,note -- binding risk characteristic per position
ASSET_NOTES_CSV = MANUAL_DIR / "newsletter_asset_notes_manual.csv"
HOLDINGS_CSV = PROCESSED_DIR / "current_holdings_all.csv"
# Free text: reader background + career/learning topics for the prompt
PROFILE_MD = MANUAL_DIR / "newsletter_profile_manual.md"

# Same example positions as the seeds in notebook 02 (fictitious allocation)
EXAMPLE_PORTFOLIO = [
    {"name": "Global Equity ETF (example)", "yf_ticker": "VWCE.DE", "target_weight": 0.65,
     "price_currency": "EUR", "asset_class": "FUND", "note": ""},
    {"name": "Emerging Markets ETF (example)", "yf_ticker": "IEMA.AS", "target_weight": 0.18,
     "price_currency": "EUR", "asset_class": "FUND", "note": ""},
    {"name": "Gold ETC (example)", "yf_ticker": "4GLD.DE", "target_weight": 0.09,
     "price_currency": "EUR", "asset_class": "FUND",
     "note": "classic defensive diversifier and inflation hedge."},
    {"name": "Apple", "yf_ticker": "AAPL", "target_weight": 0.03,
     "price_currency": "USD", "asset_class": "STOCK", "note": ""},
    {"name": "Microsoft", "yf_ticker": "MSFT", "target_weight": 0.03,
     "price_currency": "USD", "asset_class": "STOCK", "note": ""},
    {"name": "Bitcoin", "yf_ticker": "BTC-EUR", "target_weight": 0.02,
     "price_currency": "EUR", "asset_class": "CRYPTO", "note": ""},
]

EXAMPLE_PROFILE = """I follow financial markets, macroeconomics and AI/ML and invest for the long
term with a passive buy-and-hold approach. Technical background: statistics/data science.

Career/learning topics: statistics, machine learning, data science jobs."""


def load_profile() -> str:
    """Reader profile for the prompt (who reads, background, career/learning topics).

    Free text in `data/manual/newsletter_profile_manual.md` (gitignored); if the
    file is missing, a generic example profile is used.
    """
    if PROFILE_MD.exists():
        return PROFILE_MD.read_text(encoding="utf-8").strip()
    return EXAMPLE_PROFILE


def output_language() -> str:
    """Newsletter language: English unless the NEWSLETTER_LANGUAGE environment
    variable (e.g. in newsletter-agent/.env) sets another one, e.g. "German"."""
    return os.environ.get("NEWSLETTER_LANGUAGE") or "English"


def load_portfolio() -> list[dict]:
    """Portfolio positions (name, ticker, target weight, currency, note), sorted by weight descending."""
    if not (TARGET_WEIGHTS_CSV.exists() and TICKER_MAP_CSV.exists()):
        return EXAMPLE_PORTFOLIO

    weights = pd.read_csv(TARGET_WEIGHTS_CSV)[["symbol", "target_weight"]]
    tickers = pd.read_csv(TICKER_MAP_CSV)[["symbol", "yf_ticker", "price_currency"]]
    df = weights.merge(tickers, on="symbol", how="left")

    if HOLDINGS_CSV.exists():
        meta = pd.read_csv(HOLDINGS_CSV)[["symbol", "name", "asset_class"]].drop_duplicates("symbol")
        df = df.merge(meta, on="symbol", how="left")
    df["name"] = df.get("name", pd.Series(dtype=str)).fillna(df["symbol"])
    df["asset_class"] = df.get("asset_class", pd.Series(dtype=str)).fillna("")

    if ASSET_NOTES_CSV.exists():
        notes = pd.read_csv(ASSET_NOTES_CSV)[["yf_ticker", "note"]]
        df = df.merge(notes, on="yf_ticker", how="left")
    else:
        df["note"] = ""
    df["note"] = df["note"].fillna("")

    df = df.dropna(subset=["yf_ticker"]).sort_values("target_weight", ascending=False)
    return df[["name", "yf_ticker", "target_weight", "price_currency", "asset_class", "note"]].to_dict("records")


def format_allocation(portfolio: list[dict]) -> str:
    """Target allocation as a prompt block, e.g. 'VWCE.DE / Global Equity ETF 65%'."""
    lines = []
    for p in portfolio:
        pct = f"{p['target_weight'] * 100:.1f}".rstrip("0").rstrip(".")
        lines.append(f"{p['yf_ticker']} / {p['name']} {pct}%")
    return "\n".join(lines)


def format_asset_notes(portfolio: list[dict]) -> str:
    """Risk characteristic per position as a prompt block (only positions with a note)."""
    lines = [f"- {p['yf_ticker']} / {p['name']}: {p['note']}" for p in portfolio if p["note"]]
    return "\n".join(lines) if lines else "- (no position-specific notes provided)"


def stock_names(portfolio: list[dict]) -> str:
    """Single-stock names for the search query, e.g. 'Apple Microsoft'."""
    names = [re.sub(r"\s*\(.*?\)", "", p["name"]) for p in portfolio if p["asset_class"] == "STOCK"]
    return " ".join(names) if names else "Big Tech"
