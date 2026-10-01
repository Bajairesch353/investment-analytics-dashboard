# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
from google.adk.tools import google_search
import yfinance as yf
import pandas as pd
import os
import signal

from .portfolio import (
    OUTPUT_DIR,
    format_allocation,
    format_asset_notes,
    load_portfolio,
    load_profile,
    output_language,
    stock_names,
)

PORTFOLIO = load_portfolio()
OUTPUT_LANGUAGE = output_language()
PRICE_CURRENCY = {p["yf_ticker"]: p["price_currency"] for p in PORTFOLIO}

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

# Shared session with browser TLS impersonation (curl_cffi) for all Yahoo
# requests. Otherwise yfinance opens a new session per call, a pattern Yahoo
# tends to rate-limit with HTTP 429 after ~19 tickers in a row. A reused
# session keeps its cookies and lowers the risk of being blocked.
try:
    from yfinance._http import new_session as _yf_new_session
    _SESSION = _yf_new_session()
except Exception:
    _SESSION = None


# Table labels, section headings and number format follow the newsletter
# language (German or English). Headings are set here, not left to the model
# to translate -- it tends to copy English headings verbatim.
IS_GERMAN = OUTPUT_LANGUAGE.lower() in ("german", "deutsch")
_LABELS_EN = {
    "close": "Close",
    "change": "% Change",
    "as_of": "As of",
    "no_data": "no data",
    "data_as_of": "Price data as of (last completed session)",
    "source": "Source",
    "portfolio_block": "A) My Portfolio & Benchmarks",
    "market_block": "B) Broader Market Overview",
    "s1": "Executive Summary",
    "s2": "Portfolio-Relevant Market Overview (previous day's date)",
    "s2_l1": "Level 1: Portfolio & Direct Market Moves",
    "s2_l2": "Level 2: Placing It in the Broader Market",
    "s3": "Macro & Politics",
    "s4": "AI, Tech & Data Science",
    "s5": "Startup & Venture Sector",
    "s6": "Career & Learning Signals",
    "s7": "Deep Dive",
    "s8": "Actionable Watchlist",
    "s9": "Sources / Data Coverage",
}
_LABELS_DE = {
    "close": "Schlusskurs",
    "change": "%-Veränderung",
    "as_of": "Stand",
    "no_data": "keine Daten",
    "data_as_of": "Datenstand der Kurse (letzte abgeschlossene Sitzung)",
    "source": "Quelle",
    "portfolio_block": "A) Mein Portfolio & Benchmarks",
    "market_block": "B) Allgemeiner Marktüberblick",
    "s1": "Executive Summary",
    "s2": "Portfolio-relevanter Marktüberblick (Datum Vortag)",
    "s2_l1": "Ebene 1: Portfolio & direkte Marktbewegungen",
    "s2_l2": "Ebene 2: Einordnung in den größeren Markt",
    "s3": "Makro & Politik",
    "s4": "AI, Tech & Data Science",
    "s5": "Startup- & Venture-Sektor",
    "s6": "Karriere- & Lern-relevante Signale",
    "s7": "Deep Dive",
    "s8": "Actionable Watchlist",
    "s9": "Quellen / Datenlage",
}
LABELS = _LABELS_DE if IS_GERMAN else _LABELS_EN


def localize_number(text):
    """German style (1.234,56) if the newsletter is German, else 1,234.56."""
    if not IS_GERMAN:
        return text
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_price(value, ticker):
    currency = PRICE_CURRENCY.get(ticker)
    if not currency:
        if "USD" in ticker or ticker == "CL=F":
            currency = "USD"
        elif ticker.endswith(".DE"):
            currency = "EUR"
        elif ticker.endswith(".L"):
            currency = "GBP"
    text = f"{value:,.2f}"
    return localize_number(f"{text} {currency}" if currency else text)


def format_change(pct):
    sign = "🟢" if pct > 0 else "🔴" if pct < 0 else "⚪"
    return localize_number(f"{sign} {pct:+.2f}%")


def get_last_two_valid_closes(data):
    """Finds the last two valid (non-NaN) closes in the price history.

    Skips a running "today" bar and single NaN bars. For European tickers
    (.DE/.L) Yahoo occasionally returns NaN in the close field for the last
    trading day, even though open/high/low/volume are present.
    """
    close = data["Close"].squeeze()

    today = pd.Timestamp.now(tz=None).normalize()
    if data.index[-1].normalize() == today:
        close = close.iloc[:-1]

    valid = close.dropna()
    if len(valid) < 2:
        return None

    return float(valid.iloc[-1]), float(valid.iloc[-2]), valid.index[-1]


def get_price_via_quote(ticker):
    """Fallback via the Yahoo quote endpoint (yfinance fast_info).

    Used when the price history (yf.download) has no valid close. fast_info
    queries a different Yahoo endpoint than the price history and is not
    affected by its NaN problem.
    """
    try:
        fast_info = yf.Ticker(ticker, session=_SESSION).fast_info
        last_price = fast_info.get("lastPrice")
        prev_close = fast_info.get("regularMarketPreviousClose")
    except Exception:
        return None

    if last_price is None or prev_close is None:
        return None
    return float(last_price), float(prev_close), pd.Timestamp.now(tz=None)


def build_table(assets_dict):
    """Builds the price table and returns (markdown, all price dates).

    Each ticker uses its own last valid close. If Yahoo is missing a session
    for some exchanges (e.g. Xetra), rows come from different sessions -- the
    table then gets an "as of" column instead of one date for everything.
    """
    rows = []
    dates = []

    for name, ticker in assets_dict.items():
        data = yf.download(ticker, period="7d", interval="1d", progress=False,
                           auto_adjust=True, session=_SESSION, timeout=8)

        result = None if data.empty else get_last_two_valid_closes(data)
        if result is None:
            result = get_price_via_quote(ticker)

        if result is None:
            rows.append({
                "Asset": name,
                "Ticker": ticker,
                LABELS["close"]: LABELS["no_data"],
                LABELS["change"]: LABELS["no_data"],
                LABELS["as_of"]: "–",
            })
            continue

        last_close, prev_close, idx = result
        date = idx.normalize()
        dates.append(date)
        pct_change = (last_close / prev_close - 1) * 100

        rows.append({
            "Asset": f"{name:<35}",
            "Ticker": f"{ticker:<10}",
            LABELS["close"]: f"{format_price(last_close, ticker):<18}",
            LABELS["change"]: f"{format_change(pct_change):<10}",
            LABELS["as_of"]: date.strftime("%d.%m."),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Asset")
    if len(set(dates)) <= 1:
        df = df.drop(columns=LABELS["as_of"])
    return df.to_markdown(index=False), dates


def get_previous_newsletter_context() -> str:
    """Loads the executive summary of the most recently saved newsletter.

    Lets the agent check recurring themes (e.g. tech capex sentiment, oil price
    drivers, the Fed's path) against the previous day's assessment instead of
    judging every day in isolation.
    """
    output_dir = str(OUTPUT_DIR)

    if not os.path.isdir(output_dir):
        return "No previous newsletter found (output folder missing)."

    md_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("newsletter_") and f.endswith(".md")
    )
    if not md_files:
        return "No previous newsletter found."

    latest_file = md_files[-1]
    try:
        with open(os.path.join(output_dir, latest_file), "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return "Previous newsletter could not be read."

    # Match section numbers only: headings are written in the output language
    start = content.find("## 1.")
    end = content.find("## 3.")
    excerpt = content[start:end].strip() if start != -1 and end != -1 else content[:2000].strip()
    excerpt = excerpt[:2500]

    date_label = latest_file.replace("newsletter_", "").replace(".md", "")
    return f"Newsletter from {date_label} (executive summary, shortened):\n\n{excerpt}"


def create_market_table() -> str:
    portfolio_assets = {p["name"]: p["yf_ticker"] for p in PORTFOLIO}

    market_assets = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "DAX": "^GDAXI",
        "US 10Y Treasury Yield": "^TNX",
        "EUR/USD": "EURUSD=X",
        "USD Index": "DX-Y.NYB",
        "WTI Crude Oil": "CL=F",
    }

    portfolio_table, portfolio_dates = build_table(portfolio_assets)
    market_table, market_dates = build_table(market_assets)

    all_dates = portfolio_dates + market_dates
    if not all_dates:
        date_line = "unknown"
    elif len(set(all_dates)) == 1:
        date_line = all_dates[0].strftime("%d.%m.%Y")
    else:
        date_line = (
            f"{max(all_dates).strftime('%d.%m.%Y')} -- **not all prices are from "
            "the same trading day** (Yahoo is missing a session for some "
            f"exchanges), date per asset in the \"{LABELS['as_of']}\" column"
        )

    return f"""
**{LABELS["data_as_of"]}:** {date_line}
**{LABELS["source"]}:** Yahoo Finance (adjusted close)

---

### {LABELS["portfolio_block"]}

{portfolio_table}

---

### {LABELS["market_block"]}

{market_table}
"""

today_weekday = datetime.datetime.now().weekday()
# Monday = 0, Tuesday = 1, ..., Sunday = 6

def _build_market_table_safe(hard_timeout=45):
    """Builds the price table with a hard time limit and a fallback.

    This runs at module import. If it hangs or fails (Yahoo slow/blocking,
    no network), the agents-cli dev server would hit its 30-40s start timeout
    and return an empty newsletter. So: abort after `hard_timeout` seconds and
    continue with a notice instead of failing the whole agent import.
    """
    def _on_timeout(signum, frame):
        raise TimeoutError(f"price download > {hard_timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(hard_timeout)
    try:
        return create_market_table()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, import must never fail
        return (
            "**Price table not available today** "
            f"(data download failed: {type(exc).__name__}: {exc}).\n\n"
            "Work without the price table in the analysis and state this "
            "transparently in the newsletter."
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if today_weekday in [6, 0]:
    market_table = "No price table in the Sunday/Monday brief."
else:
    market_table = _build_market_table_safe()

previous_context = get_previous_newsletter_context()


if today_weekday == 6:   # Sunday
    mode_text = """

ACTIVATE: WEEKLY REVIEW BRIEF (Sunday)

IMPORTANT: This mode instruction takes precedence over later prompt rules wherever they conflict with the weekly review brief.

- Do NOT show a price table.
- Do NOT analyse the last trading session.
- No regular daily newsletter.

Focus:
- Review of the past week.
- What were the most important market moves?
- Which AI/tech/VC developments mattered?
- Which macro narratives have shifted?
- What does this mean for my portfolio?
- Close with a longer deep dive.
- Clear focus on the review; outlook only briefly at the end.

Length:
15–20 minutes
"""

elif today_weekday == 0:   # Monday
    mode_text = """

ACTIVATE: WEEK AHEAD BRIEF (Monday)

IMPORTANT: This mode instruction takes precedence over later prompt rules wherever they conflict with the week ahead brief.

- Do NOT show a price table.
- Do NOT analyse old Friday prices.
- No regular daily newsletter.

Focus:
- Outlook on the coming week.
- Which macro data releases are due?
- Which earnings matter?
- Which central bank meetings or speeches are relevant?
- Which AI/tech/VC events could become relevant?
- Which portfolio risks and opportunities should I watch?
- Close with a watchlist for the week.
- Clear focus on the outlook; review only briefly for context.

Length:
10–15 minutes
"""

else:
    mode_text = ""

newsletter_instruction="""
You are a personalised newsletter agent.

IMPORTANT:
- Use the following price table unchanged.
- The table is already correctly calculated and formatted.
- Do not change any numbers, the order or the formatting.
- Only analyse it.
- If the table has a date column ("As of" / "Stand"), the changes come from different trading days. Then do NOT present them as moves of the same session and do not compare them directly; name the trading day for the affected assets.
- Use google_search only for news, macro, AI, tech and startup developments.
- Reading time: approx. 15–20 minutes

DEPTH:
- Every section starts with a framing paragraph (macro perspective)
- Then a detailed analysis (not just bullet points)
- Explain connections (WHY, not just WHAT)
- Draw links between markets, AI and macro
- Use economic mechanisms deliberately (e.g. rates → tech valuations → capex), but only for the most important drivers
- Where useful, link the level of interest rates to fiscal sustainability (e.g. effects on government debt and yields)

Build in structural implications deliberately:
- Focus on the 1–2 most important implications per section
- Only where they add real value
- Avoid forced or trivial implications

Current price table from Yahoo Finance:

MARKET_TABLE_PLACEHOLDER

Context from the previous newsletter (to keep consistency, not to repeat):

PREVIOUS_CONTEXT_PLACEHOLDER

MOST IMPORTANT RULES:
- The entire output MUST be written in OUTPUT_LANGUAGE_PLACEHOLDER, except for the section headings: use the headings of the structure below exactly as given (they are already in the right language). Keep the title line "# Daily Personal Finance & AI Brief" exactly as it is.
- Copy the values from the embedded price table above unchanged.
- Use google_search only for current news, macro, AI/tech, company news and interpretation.
- Do not invent news, numbers or sources.
- If no source is found, say so transparently.
- Strictly separate facts from interpretation.
- No investment advice, no buy/sell recommendations.
- Write analytically, precisely and without clickbait.
- Avoid overdramatic or final narratives such as "turning point", "new era", "the end of ...", unless clearly supported by several sources.
- Phrase strong market theses in a balanced way: first observation, then interpretation, then possible implication.
- Examine the political situation (especially of the last day) closely and connect it to relevant developments.
- For commodity and currency moves (especially WTI oil, gold, the USD index): name the concrete political/geopolitical event as the cause (actor, place, decision, date) – no generic phrases like "geopolitical tensions in the Middle East" without a link to a trigger from that day. If no concrete trigger can be found, say explicitly that the move had no clear trigger instead of generalising.
- Consistency with the previous day: check your assessment of recurring themes (e.g. tech capex sentiment, the Fed's path, oil price drivers) against the embedded context of the previous newsletter. If your assessment has changed, say so explicitly in a half-sentence with the concrete new trigger (e.g. "unlike yesterday ... because ..."). Do not judge the same move in contradictory ways without relating them.
- Grammar check before output (applies when writing in German): check the simple past forms of irregular (strong) verbs in particular. Example: "leiden" → "litt/litten" (not "leidete/leideten"); likewise for other strong verbs (e.g. "stehen" → "stand/standen", "gelten" → "galt/galten"). When in doubt, choose a simpler phrasing that is certainly correct.

STYLE:
- Write like a professional macro/tech analyst (FT, Bloomberg, a16z).
- Mainly continuous prose, bullet points only occasionally.
- Every section starts with a short paragraph, bullet points optional only afterwards.
- Avoid plain enumerations without context.
- Focus on connections, not isolated facts.
- Balance readability and depth: approx. 70 percent context/story, 30 percent analytical implications

LIST FORMAT (strict):
- Never use inline lists such as:
  "Short term: * point 1 * point 2"
- Never use "*" as a bullet symbol.
- Every bullet point starts on a new line.
- Always use this visual format for lists:

Example:
Heading
(paragraph)
- Point 1
(paragraph)
- Point 2
(paragraph)
- Point 3

- There is a blank line before every list.
- Lists should look calm, like a professional FT/Bloomberg newsletter.
- If a list has only 1 item, write a normal paragraph instead.
- The concrete heading of a list depends on the respective section.

Search strategy:
Run at most 4–5 targeted google_search queries, for example:
1. "latest global markets macro news central banks inflation interest rates"
2. "latest AI news LLM model releases machine learning research arXiv"
3. "PORTFOLIO_STOCKS_PLACEHOLDER latest AI capex earnings stock news"
4. "AI infrastructure data centers chips power cooling semiconductor trends"
5. "latest Germany Europe US emerging markets economic news"

Task:
Create a personalised daily newsletter on finance and AI based on:
1. the embedded price table above
2. current public information from google_search

Context (reader profile):
READER_PROFILE_PLACEHOLDER

Main focus:
Financial markets, macroeconomics, AI/ML/statistics, tech companies, AI infrastructure and its practical use, portfolio-relevant developments, the startup sector.

Target asset allocation:
PORTFOLIO_ALLOCATION_PLACEHOLDER

Risk profile characteristics of the positions (binding – do not contradict):
PORTFOLIO_NOTES_PLACEHOLDER
- Single stocks: risk profile differs by business model – do not classify them all the same way.
- Regional or style diversifiers (e.g. EM, Europe, small caps): assess each one on its own based on the day's move instead of labelling them across the board.
Before giving a position a label such as "defensive", "cyclical", "growth" or "value": check it against this list. If the label contradicts the characteristic, correct the sentence.

Length:
30 minutes reading time, but analytically dense rather than artificially long.
The consistency and causality rules above (reference to the previous day, concrete political triggers) should sharpen existing sentences, not add extra text. Do not increase the overall length.

Structure:

# Daily Personal Finance & AI Brief
Date: use today's system date and the current time.

## 1. SECTION_TITLE_s1
Start with 1–2 sentences stating the day's central market thesis.
Then 4–6 concise supporting bullet points.

## 2. SECTION_TITLE_s2

Use the embedded price table unchanged:
- Copy the table exactly.
- Do not change any numbers.
- Do not reformat.
- Do not recalculate any values.

Then a detailed analysis on two levels:

### SECTION_TITLE_s2_l1
Discuss relevant moves from the price table:

- Which positions or indices stand out?
- Which short-term drivers explain the moves?
- Which moves look more like noise?
- Where could structural trends be emerging?
- Are there signs of sector rotation within my portfolio?
- Look not only at the previous day's moves but also at the development over the last days/weeks.
- For larger moves, search the sources for the most precise reasons for exactly these moves.

### SECTION_TITLE_s2_l2
Use the table as a starting point and extend the analysis to broader developments:

- Which global market moves matter today, even if they are not in the table?
- Which sectors are leading or lagging?
- Which companies outside my portfolio are sending relevant signals?
- Which narratives currently dominate?
  (AI, energy, deficits, China, commodities, labour market, geopolitics, fiscal policy, etc.)
- Which second-round effects could emerge?
- Which market moves appear short-term, which structural?

Analyse explicitly:
- Sector rotation:
  - Growth vs value
  - Tech vs energy
  - Software vs infrastructure
  - Small caps vs mega caps
  - US vs Europe vs emerging markets

- Interpretation of index moves:
  - Why are the DAX, Nasdaq or S&P moving?
  - Which components drive the move?

Balance:
- Approx. 50 percent focus on the portfolio/price table
- Approx. 50 percent broader market analysis
- The market overview should read like a professional Bloomberg/FT report, not like a pure portfolio analysis.

To close:
A short deep dive into an interesting stock or company outside my portfolio:

Criteria:
- current relevance
- not always the same company
- focus on:
  - business model
  - current trigger
  - opportunities
  - risks
  - structural significance

## 3. SECTION_TITLE_s3
Inflation, interest rates, central banks, labour market, geopolitics.
Focus on Germany, Europe, the US and emerging markets.

## 4. SECTION_TITLE_s4
- New AI models, research, chips, compute, cloud, energy, cooling, agents, regulation, business applications.
- If available: relevant developments (announcements, models, research, trends)

## 5. SECTION_TITLE_s5

Focus:
- Venture capital activity (fundraising, large rounds)
- Startup trends (AI, climate, fintech, deep tech)
- Big Tech vs startups (e.g. buy vs build)
- M&A activity
- IPO market (if relevant)

Analysis:
- What does this say about risk appetite in the market?
- Early-cycle vs late-cycle?
- Link to interest rates / liquidity
- Describe capital cycles, not just deals
- Explain: early cycle vs late cycle
- Link to interest rates, liquidity, IPO window
- Which trends are likely to spill over into public markets

Rules:
- Use google_search for current deals/trends
- No list of random startups → only relevant cases
- Focus on signals, not news noise
- Describe 1–2 concrete examples (deals, companies, moves) and derive a signal from them.

## 6. SECTION_TITLE_s6
Only if really relevant:
Developments related to the career/learning topics from the reader profile (see context).

## 7. SECTION_TITLE_s7
One relevant topic in more detail:
Background, current development, significance, second-round effects, outlook.
- At least 2–3 paragraphs
- Clear structure: background → current trigger → implications → outlook
- Ideally one economic mechanism

## 8. SECTION_TITLE_s8
- 3 short-term observations or relevant events (1–4 weeks)
- 3 long-term trends
- optional: 1 data science / coding / portfolio project idea

## 9. SECTION_TITLE_s9
List all sources/links used.
State transparently if the source coverage is thin.
""".replace("MARKET_TABLE_PLACEHOLDER", market_table).replace("PREVIOUS_CONTEXT_PLACEHOLDER", previous_context) \
    .replace("PORTFOLIO_ALLOCATION_PLACEHOLDER", format_allocation(PORTFOLIO)) \
    .replace("PORTFOLIO_NOTES_PLACEHOLDER", format_asset_notes(PORTFOLIO)) \
    .replace("PORTFOLIO_STOCKS_PLACEHOLDER", stock_names(PORTFOLIO)) \
    .replace("READER_PROFILE_PLACEHOLDER", load_profile()) \
    .replace("OUTPUT_LANGUAGE_PLACEHOLDER", OUTPUT_LANGUAGE)

for _key in ("s2_l1", "s2_l2", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"):
    newsletter_instruction = newsletter_instruction.replace(f"SECTION_TITLE_{_key}", LABELS[_key])

newsletter_instruction = mode_text + "\n" + newsletter_instruction

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=newsletter_instruction,
    tools=[google_search],
)

app = App(
    root_agent=root_agent,
    name="app",
)
