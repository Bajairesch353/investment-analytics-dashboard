"""Central bank speech sentiment (notebook 11): fetch Fed/ECB speeches,
classify their monetary policy stance with a local LLM (Ollama), and derive
scores, rolling averages and a per-bank summary; plus the actual policy rates
as an external reference point for the LLM output."""

import csv
import io
import json
import re
import warnings
from datetime import datetime

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

FED_SPEECHES_RSS = "https://www.federalreserve.gov/feeds/speeches.xml"

# Official ECB full-text export of all speeches (pipe-separated, includes the
# complete speech text). Replaces the earlier mixed press RSS feed, which only
# returned ~15 recent items, mostly 0-2 speeches, some as slide-deck PDFs
# without usable text. The export is updated about once a month, so the
# latest few weeks of ECB speeches are usually missing.
# Source: https://www.ecb.europa.eu/press/key/html/downloads.en.html
ECB_SPEECHES_CSV = (
    "https://www.ecb.europa.eu/press/key/shared/data/all_ECB_speeches.csv"
)

# Policy rates as external reference for the LLM stance (no API key needed):
# Fed funds target range upper limit (FRED) and ECB deposit facility rate.
FED_POLICY_RATE_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU&cosd={start}"
ECB_POLICY_RATE_CSV = (
    "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV"
    "?startPeriod={start}&format=csvdata"
)

DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def fetch_fed_speeches(limit=20):
    """Fed speeches from the RSS feed (only the latest ~15). The full text is
    NOT included and has to be loaded via extract_speech_text() -> text=None."""
    feed = feedparser.parse(FED_SPEECHES_RSS)
    return [
        {
            "central_bank": "Fed",
            "speaker": e.title.split(",")[0].strip(),
            "title": e.title,
            "url": e.link,
            "date": datetime(*e.published_parsed[:6]),
            "text": None,
        }
        for e in feed.entries[:limit]
    ]


# Columns of the ECB export -> internal names. The schema is neither
# documented nor versioned, so a renamed column fails loudly instead of
# silently producing empty speeches.
ECB_COLUMNS = {"date": "date", "speakers": "speaker", "title": "title", "contents": "text"}


def _normalize_ecb_columns(df):
    """Rename the export's columns to internal names; KeyError if any is missing."""
    missing = set(ECB_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(
            f"ECB speech export lacks columns {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )
    return df.rename(columns=ECB_COLUMNS)


# ECB export: the speech body comes first; footnotes/references are appended
# as separate blocks, separated by runs of whitespace. Without stripping them,
# the tail of the prompt excerpt (see _build_excerpt) shows the model mostly
# references instead of the speech's conclusion.
_ECB_BLOCK_SEP = re.compile(r"\s{4,}")
_CLOSING = re.compile(
    r"(?:I\s+)?[Tt]hank you(?: very much)?(?: (?:all|again))?"
    r"(?: for your (?:kind )?attention| for listening)?[^.!\n]{0,80}[.!]"
)
_REFERENCE = re.compile(
    r"\(\d{4}[a-z]?\)|\bpp?\.\s*\d|\bVol\.|\bibid\b|https?://|\bet al\."
    r"|^\s*(?:See|Cf\.|For (?:more|further|a )|Source)\b"
    r"|Working Paper|Occasional Paper|Economic Bulletin|Discussion Paper",
    re.I,
)


def strip_ecb_references(text, max_reference_len=900):
    """Cut the appended footnotes/references off an ECB export text.

    1. A closing formula ("Thank you for your attention.") in the last 40% of
       the text marks the end of the speech -> cut right after it.
    2. Otherwise drop whitespace-separated blocks from the end as long as they
       look like references (short, with a year in brackets, "See ...",
       "pp.", a URL, ...). Long prose footnotes can survive this step.
    """
    text = text.strip()
    closings = [m for m in _CLOSING.finditer(text) if m.start() > 0.6 * len(text)]
    if closings:
        return text[: closings[-1].end()]

    blocks = _ECB_BLOCK_SEP.split(text)
    keep = len(blocks)
    while keep > 1:
        block = blocks[keep - 1].strip()
        if block == "" or (
            len(block) <= max_reference_len and (_REFERENCE.search(block) or len(block) < 120)
        ):
            keep -= 1
        else:
            break
    return "    ".join(blocks[:keep]).strip()


def fetch_ecb_speeches(limit=20, since=None):
    """ECB speeches incl. full text from the official CSV export.

    Unlike fetch_fed_speeches, the CSV already contains the speech text, so
    extract_speech_text() is not needed for the ECB. Entries without text
    (slide decks: the export only has title + "Slides by ...") are dropped
    with a warning listing them.

    since: optional date (str or Timestamp), e.g. "2024-01-01".
    limit: newest N speeches after filtering (None = all).
    """
    df = pd.read_csv(
        ECB_SPEECHES_CSV,
        sep="|",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
        dtype=str,
    )

    df = _normalize_ecb_columns(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.sort_values("date", ascending=False)
    if limit is not None:
        df = df.head(limit)

    has_text = df["text"].fillna("").astype(str).str.len() > 200
    if not has_text.all():
        dropped = df.loc[~has_text, ["date", "title"]]
        listing = "; ".join(f"{d:%Y-%m-%d} {t}" for d, t in dropped.itertuples(index=False))
        warnings.warn(f"{len(dropped)} ECB speech(es) without text in the export (slides), skipped: {listing}")
    df = df[has_text]

    speeches = []
    for _, r in df.iterrows():
        # The export has no URL -> synthetic, stable key, so that the cache
        # dedup on "url" keeps working.
        speaker_slug = re.sub(r"\W+", "-", str(r["speaker"]).lower()).strip("-")
        url = f"ecb-speech://{r['date'].date()}/{speaker_slug}"

        speeches.append(
            {
                "central_bank": "ECB",
                "speaker": str(r["speaker"]).strip(),
                "title": str(r["title"]).strip(),
                "url": url,
                "date": r["date"].to_pydatetime(),
                "text": strip_ecb_references(str(r["text"])),
            }
        )

    return speeches


def extract_fed_speech_body(html):
    """Speech text of a federalreserve.gov speech page: the paragraphs of the
    main article column up to the <hr> that separates the footnotes.
    Returns None if the page doesn't have that structure."""
    soup = BeautifulSoup(html, "lxml")
    columns = soup.select("#article > div.col-xs-12.col-sm-8.col-md-8")
    body = [c for c in columns if "heading" not in (c.get("class") or [])]
    if not body:
        return None

    paragraphs = []
    for element in body[-1].find_all(recursive=False):
        if element.name == "hr":
            break  # footnotes follow
        text = element.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs) or None


def extract_speech_text(url, timeout=30):
    """Load the full text of a speech. Only needed for the Fed (the ECB text
    already comes with fetch_ecb_speeches).

    HTML: the article body without navigation boilerplate and footnotes
    (extract_fed_speech_body); if the page layout differs, all <p> elements
    as a fallback, with a warning. The PDF branch stays as a fallback in case
    a speech is only published as PDF.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    if url.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    body = extract_fed_speech_body(resp.content)
    if body:
        return body

    warnings.warn(f"Unknown page layout, falling back to all <p> elements (incl. footnotes): {url}")
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 40]  # drop boilerplate lines
    return "\n\n".join(paragraphs)


# Translation of the categorical classification into a numeric score.
# Deliberately here and not in the model: small models pick categories
# reliably but give inconsistent values on a continuous scale (with
# llama3.2:3b the sign often contradicted its own chosen direction). Changing
# these numbers lets the score be recomputed from the existing cache --
# without classifying the speeches again.
STANCE_SCORE_MAP = {
    ("hawkish", "slight"): 0.25,
    ("hawkish", "moderate"): 0.55,
    ("hawkish", "strong"): 0.85,
    ("dovish", "slight"): -0.25,
    ("dovish", "moderate"): -0.55,
    ("dovish", "strong"): -0.85,
}


def derive_stance_score(direction, intensity):
    """Numeric stance score from direction and intensity.

    Returns None if the speech is not about monetary policy or the
    combination is unknown -- the score is then filtered out, instead of
    silently counting as 0.0 (= "explicitly neutral").
    """
    if direction == "balanced":
        return 0.0
    if direction in (None, "not_applicable"):
        return None
    return STANCE_SCORE_MAP.get((direction, intensity))


def _build_excerpt(text, head=3500, tail=2500, threshold=6000):
    """Text excerpt for the prompt: beginning + end instead of beginning only.

    Speeches often state their concrete policy conclusion near the end -- a
    plain text[:threshold] cut (earlier behaviour) shows the model only the
    introduction for most speeches (median length > threshold).
    """
    if len(text) <= threshold:
        return text
    return text[:head] + "\n\n[... middle of the speech omitted ...]\n\n" + text[-tail:]


def classify_speech_stance(text, speaker, central_bank, model=DEFAULT_MODEL):
    text_excerpt = _build_excerpt(text)

    prompt = f"""
You are analyzing a central bank speech for monetary policy stance.

Speaker: {speaker}
Central bank: {central_bank}

Speech text:
{text_excerpt}

Task -- steps, in this order:

Step 1: First name the speech's main topic in a few words (e.g. "rate path /
inflation outlook", "digital euro / payments", "AI governance in banking
supervision", "bank capital regulation"). Deciding the topic FIRST, before
the yes/no gate below, matters: committing to yes/no cold, without first
naming what the speech is actually about, is what causes off-topic speeches
to get misclassified.

Then decide whether this speech is about monetary policy at all, i.e. about
interest rates, inflation, the policy stance, or the economic outlook that
feeds directly into those decisions.
Speeches about payment systems, the digital euro, banking supervision and
regulation, financial inclusion, climate/green transition, AI, or ceremonial
and welcoming remarks are NOT monetary policy -- even when delivered by a
central banker, even if they mention inflation or rates in passing, and even
if the speaker is a well-known monetary policymaker. Judge the topic you just
named, not the speaker's usual role.
Set is_monetary_policy accordingly.

Step 2: Only if is_monetary_policy is true, write a short rationale (2-4
sentences): what does the speaker actually say about rates, inflation, growth
or the outlook? Reach your conclusion about direction and intensity here --
the categorical fields below must follow from this rationale, not the other
way round. If is_monetary_policy is false, keep rationale to one short clause
repeating the topic (e.g. "About the digital euro, not monetary policy.").

Step 3: Decide the DIRECTION:
  "hawkish"  -> the speech points towards HIGHER rates / tighter policy
  "dovish"   -> the speech points towards LOWER rates / easier policy
  "balanced" -> genuinely no directional lean
If is_monetary_policy is false, use "not_applicable".

CRITICAL -- do not confuse tone with direction. Both of these sound worried,
but they point in OPPOSITE directions:
  - Worried that inflation is too high / not falling fast enough
    -> HAWKISH (positive score). This is the most common mistake: it sounds
       negative, but wanting to fight inflation means wanting HIGHER rates.
  - Worried about weak growth, rising unemployment, recession risk
    -> DOVISH (negative score).
The emotional tone of the speech is irrelevant. The only question is: does
this speaker want rates higher or lower?

Step 4: Choose the INTENSITY of that lean -- this is about how insistently
the speaker presses the lean, NOT about whether numbers/figures are
mentioned (plenty of "moderate" speeches cite inflation data; that alone
does not make them more intense). Judge it by what the speaker actually
does in the text, not by how strongly worded the speech feels. Do NOT
default to "moderate" when unsure -- check each level in order and stop at
the first one that fits:
  "slight"   -> mentions the lean in passing; mostly descriptive or analytical;
                no clear preference for a policy move.
  "moderate" -> a clear preference is recognisable, but expressed in a
                measured, unhurried way -- concern or optimism is present,
                without urgency or a sense that action may be needed soon.
  "strong"   -> the lean is expressed with real conviction or urgency and is
                tied fairly directly to current or near-term policy -- e.g.
                explicitly calls for, announces, or defends a concrete change
                in rates; cites concrete already-taken policy actions (a rate
                hike, quantitative tightening, a specific rate level) to
                justify the current stance; or uses language like "we must
                remain vigilant", "further tightening may well be needed",
                "I would not hesitate to act further".
If stance_direction is "balanced" or "not_applicable", use "none".

Step 5: Independently of the categories above, give your own numeric estimate
of the stance in stance_score_llm, from -1.0 (very dovish) to +1.0 (very
hawkish), or null if the speech is not about monetary policy. This field is
only used to cross-check the categories; it does not need to match them.

Examples (illustrative only, not from real speeches):

1) "Given that core inflation remains stubbornly above target, I believe we
must raise the policy rate by another 50 basis points at our next meeting to
restore price stability."
-> rationale: "Explicitly calls for a concrete 50bp rate hike to fight
above-target inflation." | stance_direction: hawkish | stance_intensity:
strong

2) "Growth has been softening in recent quarters, and while this bears
watching, it is too early to draw firm conclusions about the appropriate
path for policy."
-> rationale: "Notes softening growth but draws no firm policy conclusion,
only a mild lean towards caution." | stance_direction: dovish |
stance_intensity: slight

3) "Inflation has come down from its peak, but at 3.4% it remains well above
our 2% goal, and I don't think we've yet done enough to be confident it will
get there."
-> rationale: "Signals continued concern that inflation is too high and more
work is needed, without calling for a specific rate move." | stance_direction:
hawkish | stance_intensity: moderate

3b) "Inflation has proven more persistent than we hoped, and I want to be
clear: further tightening may well be necessary in the months ahead if price
pressures do not ease."
-> rationale: "Goes beyond general concern to express real urgency and open
willingness to tighten further soon." | stance_direction: hawkish |
stance_intensity: strong

4) "We continue to weigh the risk of persistent inflation against the risk of
an unnecessarily sharp slowdown in employment; neither risk currently
dominates the other."
-> rationale: "Explicitly weighs inflation risk against employment risk as
equally balanced, with no directional lean." | stance_direction: balanced |
stance_intensity: none

5) "The digital euro would give European citizens a public, risk-free digital
means of payment, safeguarding monetary sovereignty in an increasingly
digitalised economy."
-> topic: "digital euro / payments infrastructure" | is_monetary_policy:
false | stance_direction: not_applicable | stance_intensity: none

6) "Financial institutions adopting AI must embed robust governance, model
risk management and cybersecurity safeguards from the outset, and supervisors
need to adapt their toolkits accordingly."
-> topic: "AI governance in banking supervision" | is_monetary_policy: false |
stance_direction: not_applicable | stance_intensity: none

Return only valid JSON:
{{
  "topic": "...",
  "is_monetary_policy": true,
  "rationale": "...",
  "stance_direction": "hawkish",
  "stance_intensity": "moderate",
  "stance_score_llm": 0.0,
  "confidence": 0.0,
  "key_topics": ["...", "..."]
}}

Rules:
- Judge only what is actually said, not the speaker's reputation.
- "balanced" is for speeches that explicitly weigh inflation and growth risks
  equally. It does NOT mean "not about policy" -- that case is covered by
  is_monetary_policy = false and stance_direction = "not_applicable".
- confidence reflects how clear the signal is, not how strong the stance is.
  A clearly dovish speech gets a HIGH confidence and a dovish direction.
"""

    schema = {
        "type": "object",
        "properties": {
            # Order matters: Ollama generates the fields in schema order.
            # "topic" deliberately comes BEFORE is_monetary_policy: without
            # it the model decided True/False cold, before "knowing" what the
            # speech is about -- speeches on the digital euro/AI/banking
            # supervision slipped through despite the exclusion list in the
            # prompt (practically every speech came back as
            # is_monetary_policy=True, regardless of topic).
            "topic": {"type": "string"},
            "is_monetary_policy": {"type": "boolean"},
            # rationale BEFORE the categories (used to be last): forces a
            # reasoning step before the model commits to direction/intensity,
            # instead of justifying an already chosen category afterwards.
            "rationale": {"type": "string"},
            # Categorical direction BEFORE the number: llama3.2:3b inverted the
            # scale (inflation worry = worried tone = scored negative, although
            # inflation worry is hawkish, i.e. positive). An enum choice is much
            # more robust than a continuous scale; the score is derived from it.
            "stance_direction": {
                "type": "string",
                "enum": ["hawkish", "dovish", "balanced", "not_applicable"],
            },
            "stance_intensity": {
                "type": "string",
                "enum": ["slight", "moderate", "strong", "none"],
            },
            # Control field only: the model's number is NOT used, just recorded
            # to cross-check it against the derived score (e.g. when comparing
            # larger models).
            "stance_score_llm": {"type": ["number", "null"]},
            "confidence": {"type": "number"},
            "key_topics": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "topic",
            "is_monetary_policy",
            "rationale",
            "stance_direction",
            "stance_intensity",
            "stance_score_llm",
            "confidence",
            "key_topics",
        ],
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            # Low temperature for more consistent category choice -- Ollama's
            # default (~0.8) adds needless variance for a classification task.
            "options": {"temperature": 0.2},
        },
        timeout=300,
    )
    response.raise_for_status()

    result = json.loads(response.json()["response"])

    # Authoritative score: derived deterministically from the categories,
    # not taken from the model.
    result["stance_score"] = derive_stance_score(
        result.get("stance_direction"),
        result.get("stance_intensity"),
    )

    return result


def parse_policy_rate_csv(text, date_col, value_col):
    """Policy rate CSV (FRED or ECB data portal) -> Series date -> rate,
    reduced to the dates where the rate changes (plus the first one)."""
    df = pd.read_csv(io.StringIO(text))
    series = (
        pd.Series(
            pd.to_numeric(df[value_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col]),
        )
        .dropna()
        .sort_index()
    )
    return series[series.ne(series.shift())]


def fetch_policy_rates(start, timeout=30):
    """Fed target rate (upper bound) and ECB deposit rate since `start`,
    long format: date, central_bank, policy_rate. The last observed value is
    repeated at the latest date so step charts run up to today."""
    sources = {
        "Fed": (FED_POLICY_RATE_CSV, "observation_date", "DFEDTARU"),
        "ECB": (ECB_POLICY_RATE_CSV, "TIME_PERIOD", "OBS_VALUE"),
    }
    frames = []
    for bank, (url, date_col, value_col) in sources.items():
        resp = requests.get(url.format(start=start), timeout=timeout)
        resp.raise_for_status()
        full = pd.read_csv(io.StringIO(resp.text))
        last_date = pd.to_datetime(full[date_col]).max()
        changes = parse_policy_rate_csv(resp.text, date_col, value_col)
        if last_date not in changes.index:
            changes.loc[last_date] = changes.iloc[-1]
        frames.append(pd.DataFrame({
            "date": changes.index, "central_bank": bank, "policy_rate": changes.values,
        }))
    return pd.concat(frames, ignore_index=True)


def rolling_stance(sentiment, window="90D"):
    """Per bank: rolling mean of stance_score over a calendar window, one
    value per speech (aligned to sentiment's index). Input: monetary policy
    speeches with date, central_bank, stance_score."""
    out = pd.Series(index=sentiment.index, dtype=float)
    for _, group in sentiment.groupby("central_bank"):
        group = group.sort_values("date")
        rolled = group.set_index("date")["stance_score"].rolling(window).mean()
        out.loc[group.index] = rolled.values
    return out


def stance_summary(sentiment, all_speeches, as_of, days=90):
    """Per bank: mean stance over the last `days` up to `as_of` and over the
    window before, number of monetary policy speeches in the current window,
    and the dates of the latest speech (any topic) and latest monetary
    policy speech — so a stale source is visible."""
    as_of = pd.Timestamp(as_of)
    window = pd.Timedelta(days=days)
    rows = []
    for bank in sorted(all_speeches["central_bank"].unique()):
        mp = sentiment[sentiment["central_bank"] == bank]
        current = mp[(mp["date"] > as_of - window) & (mp["date"] <= as_of)]
        previous = mp[(mp["date"] > as_of - 2 * window) & (mp["date"] <= as_of - window)]
        speeches = all_speeches[all_speeches["central_bank"] == bank]
        rows.append({
            "central_bank": bank,
            "as_of": as_of,
            "window_days": days,
            "current_mean": current["stance_score"].mean() if len(current) else None,
            "previous_mean": previous["stance_score"].mean() if len(previous) else None,
            "n_current": len(current),
            "n_monetary_policy_total": len(mp),
            "latest_speech": speeches["date"].max(),
            "latest_monetary_policy_speech": mp["date"].max() if len(mp) else None,
        })
    return pd.DataFrame(rows)
