import pandas as pd
import pytest
import requests

import src.macro_sentiment as macro_sentiment

from src.macro_sentiment import (
    STANCE_SCORE_MAP,
    _build_excerpt,
    _normalize_ecb_columns,
    derive_stance_score,
    extract_fed_speech_body,
    fallback_policy_rates,
    fetch_policy_rates,
    parse_policy_rate_csv,
    rolling_stance,
    stance_summary,
    strip_ecb_references,
)


def test_derive_stance_score_balanced_is_zero():
    assert derive_stance_score("balanced", "none") == 0.0


@pytest.mark.parametrize("direction", [None, "not_applicable"])
def test_derive_stance_score_not_monetary_policy_is_none(direction):
    assert derive_stance_score(direction, "moderate") is None


@pytest.mark.parametrize(("direction", "intensity"), STANCE_SCORE_MAP.keys())
def test_derive_stance_score_matches_map_for_known_combinations(direction, intensity):
    assert derive_stance_score(direction, intensity) == STANCE_SCORE_MAP[(direction, intensity)]


def test_derive_stance_score_hawkish_outranks_dovish_at_same_intensity():
    hawkish = derive_stance_score("hawkish", "moderate")
    dovish = derive_stance_score("dovish", "moderate")
    assert hawkish > 0
    assert dovish < 0
    assert hawkish == pytest.approx(-dovish)


def test_derive_stance_score_unknown_combination_is_none():
    assert derive_stance_score("hawkish", "unknown_intensity") is None


def test_normalize_ecb_columns_renames_to_internal_names():
    df = pd.DataFrame(columns=["date", "speakers", "title", "subtitle", "contents"])
    assert list(_normalize_ecb_columns(df).columns) == ["date", "speaker", "title", "subtitle", "text"]


def test_normalize_ecb_columns_missing_column_raises_with_available_columns():
    df = pd.DataFrame(columns=["date", "speakers", "title"])
    with pytest.raises(KeyError, match="contents"):
        _normalize_ecb_columns(df)


def test_build_excerpt_returns_full_text_below_threshold():
    text = "x" * 100
    assert _build_excerpt(text, head=30, tail=30, threshold=6000) == text


def test_build_excerpt_truncates_middle_above_threshold():
    text = "A" * 3500 + "B" * 1000 + "C" * 2500
    excerpt = _build_excerpt(text, head=3500, tail=2500, threshold=6000)
    assert excerpt.startswith("A" * 3500)
    assert excerpt.endswith("C" * 2500)
    assert "B" not in excerpt
    assert "[... middle of the speech omitted ...]" in excerpt


def test_parse_policy_rate_csv_keeps_only_changes():
    text = (
        "observation_date,DFEDTARU\n"
        "2025-12-10,4.00\n2025-12-11,3.75\n2025-12-12,3.75\n"
        "2026-09-17,4.00\n2026-09-18,.\n"
    )
    rates = parse_policy_rate_csv(text, "observation_date", "DFEDTARU")
    assert list(rates.index.strftime("%Y-%m-%d")) == ["2025-12-10", "2025-12-11", "2026-09-17"]
    assert list(rates.values) == [4.00, 3.75, 4.00]



PREVIOUS_RATES = pd.DataFrame({
    "date": ["2026-06-11", "2026-09-17", "2026-06-05"],
    "central_bank": ["ECB", "ECB", "Fed"],
    "policy_rate": [2.25, 2.50, 4.00],
})

FED_CSV = "observation_date,DFEDTARU\n2026-09-16,3.75\n2026-09-17,4.00\n2026-10-02,4.00\n"


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _fed_ok_ecb_down(url, timeout):
    if "data-api.ecb.europa.eu" in url:
        raise requests.ReadTimeout("read timed out")
    return _FakeResponse(FED_CSV)


def test_fallback_policy_rates_returns_only_that_bank():
    rows = fallback_policy_rates(PREVIOUS_RATES, "ECB")
    assert list(rows["policy_rate"]) == [2.25, 2.50]
    assert set(rows["central_bank"]) == {"ECB"}


def test_fallback_policy_rates_without_previous_is_empty():
    assert fallback_policy_rates(None, "ECB").empty


def test_fetch_policy_rates_keeps_other_bank_and_previous_values(monkeypatch):
    monkeypatch.setattr(macro_sentiment.requests, "get", _fed_ok_ecb_down)
    monkeypatch.setattr(macro_sentiment.time, "sleep", lambda s: None)
    with pytest.warns(UserWarning, match="ECB policy rate not available"):
        rates = fetch_policy_rates("2026-01-01", previous=PREVIOUS_RATES)
    fed = rates[rates["central_bank"] == "Fed"]
    ecb = rates[rates["central_bank"] == "ECB"]
    assert list(fed["policy_rate"]) == [3.75, 4.00, 4.00]  # fresh, last value repeated at latest date
    assert list(ecb["policy_rate"]) == [2.25, 2.50]         # from the previous export
    assert rates.attrs["stale_banks"] == ["ECB"]


def test_fetch_policy_rates_all_down_without_previous_is_empty(monkeypatch):
    def all_down(url, timeout):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(macro_sentiment.requests, "get", all_down)
    monkeypatch.setattr(macro_sentiment.time, "sleep", lambda s: None)
    with pytest.warns(UserWarning):
        rates = fetch_policy_rates("2026-01-01")
    assert rates.empty
    assert list(rates.columns) == ["date", "central_bank", "policy_rate"]
    assert rates.attrs["stale_banks"] == ["Fed", "ECB"]

SPEECHES = pd.DataFrame({
    "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-06-01", "2026-01-15", "2026-06-20"]),
    "central_bank": ["ECB", "ECB", "ECB", "Fed", "Fed"],
    "stance_score": [0.2, 0.4, -0.2, 0.55, 0.25],
})


def test_rolling_stance_is_per_bank_and_calendar_based():
    rolled = rolling_stance(SPEECHES, window="90D")
    assert rolled.iloc[1] == pytest.approx(0.3)   # ECB Jan + Feb
    assert rolled.iloc[2] == pytest.approx(-0.2)  # ECB June: Jan/Feb outside 90 days
    assert rolled.iloc[3] == pytest.approx(0.55)  # Fed not mixed with ECB


def test_stance_summary_windows_and_latest_dates():
    # plus a later non-monetary-policy speech (no score)
    all_speeches = pd.concat([
        SPEECHES[["date", "central_bank"]],
        pd.DataFrame({"date": pd.to_datetime(["2026-08-28"]), "central_bank": ["ECB"]}),
    ])
    summary = stance_summary(SPEECHES, all_speeches, as_of="2026-06-30", days=90).set_index("central_bank")
    assert summary.loc["ECB", "current_mean"] == pytest.approx(-0.2)
    # previous window (01.01., 01.04.] is left-open: only the February speech
    assert summary.loc["ECB", "previous_mean"] == pytest.approx(0.4)
    assert summary.loc["ECB", "n_current"] == 1
    assert summary.loc["ECB", "latest_speech"] == pd.Timestamp("2026-08-28")
    assert summary.loc["ECB", "latest_monetary_policy_speech"] == pd.Timestamp("2026-06-01")
    assert summary.loc["Fed", "current_mean"] == pytest.approx(0.25)


FED_PAGE = """
<html><body><nav>Official website of the United States Government</nav>
<div id="article">
  <div class="heading col-xs-12 col-sm-8 col-md-8"><h3>Economic Conditions</h3></div>
  <div class="col-xs-12 col-sm-4 col-md-4 hidden-sm"></div>
  <div class="col-xs-12 col-sm-8 col-md-8">
    <p>Thank you for having me.</p>
    <h4>Inflation</h4>
    <p>Further policy adjustments are likely to be needed.</p>
    <hr>
    <p>1. The views expressed here are my own.</p>
    <p>2. See Smith (2024), pp. 12-15.</p>
  </div>
</div></body></html>
"""


def test_extract_fed_speech_body_skips_boilerplate_and_footnotes():
    body = extract_fed_speech_body(FED_PAGE)
    assert body.startswith("Thank you for having me.")
    assert "Further policy adjustments" in body
    assert "Inflation" in body
    assert "views expressed" not in body
    assert "Official website" not in body


def test_extract_fed_speech_body_unknown_layout_returns_none():
    assert extract_fed_speech_body("<html><body><p>Some text</p></body></html>") is None


def test_strip_ecb_references_cuts_after_closing_formula():
    speech = "Intro. " * 50 + "We stand ready to act. Thank you for your attention."
    text = speech + "     Lane, P. (2026), Outlook, speech.     See also Figure 2."
    assert strip_ecb_references(text) == speech


def test_strip_ecb_references_drops_reference_blocks_without_closing():
    body = "The disinflation process is well advanced but services inflation remains high. " * 5
    text = body + "     Schnabel, I. (2025), R-star, Economic Bulletin.     See Chart 3.     ibid, pp. 4-5."
    assert strip_ecb_references(text) == body.strip()


def test_strip_ecb_references_keeps_text_without_references():
    body = "A speech without footnotes. " * 20
    assert strip_ecb_references(body) == body.strip()
