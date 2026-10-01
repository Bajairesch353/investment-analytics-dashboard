"""Tests für app/portfolio.py (ohne Netzwerk/LLM)."""

import pandas as pd

from app import portfolio as pf


def _point_to(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "TARGET_WEIGHTS_CSV", tmp_path / "target_weights_manual.csv")
    monkeypatch.setattr(pf, "TICKER_MAP_CSV", tmp_path / "ticker_map_manual.csv")
    monkeypatch.setattr(pf, "ASSET_NOTES_CSV", tmp_path / "newsletter_asset_notes_manual.csv")
    monkeypatch.setattr(pf, "HOLDINGS_CSV", tmp_path / "current_holdings_all.csv")
    monkeypatch.setattr(pf, "PROFILE_MD", tmp_path / "newsletter_profile_manual.md")


def test_load_portfolio_falls_back_to_example(tmp_path, monkeypatch):
    _point_to(tmp_path, monkeypatch)
    assert pf.load_portfolio() == pf.EXAMPLE_PORTFOLIO


def test_example_portfolio_weights_sum_to_one():
    assert abs(sum(p["target_weight"] for p in pf.EXAMPLE_PORTFOLIO) - 1) < 1e-9


def test_load_portfolio_from_csvs(tmp_path, monkeypatch):
    _point_to(tmp_path, monkeypatch)
    pd.DataFrame([
        {"symbol": "A", "target_weight": 0.3},
        {"symbol": "B", "target_weight": 0.7},
        {"symbol": "C", "target_weight": 0.0},  # kein Ticker-Mapping -> fällt raus
    ]).to_csv(pf.TARGET_WEIGHTS_CSV, index=False)
    pd.DataFrame([
        {"symbol": "A", "yf_ticker": "AAA.DE", "price_currency": "EUR"},
        {"symbol": "B", "yf_ticker": "BBB", "price_currency": "USD"},
    ]).to_csv(pf.TICKER_MAP_CSV, index=False)
    pd.DataFrame([
        {"symbol": "B", "name": "Beta Corp (Class A)", "asset_class": "STOCK"},
    ]).to_csv(pf.HOLDINGS_CSV, index=False)
    pd.DataFrame([{"yf_ticker": "AAA.DE", "note": "defensiv"}]).to_csv(pf.ASSET_NOTES_CSV, index=False)

    result = pf.load_portfolio()

    assert [p["yf_ticker"] for p in result] == ["BBB", "AAA.DE"]  # absteigend nach Gewicht
    assert result[1]["name"] == "A"  # ohne Holdings-Eintrag -> Symbol als Name
    assert result[1]["note"] == "defensiv"
    assert result[0]["note"] == ""
    assert pf.stock_names(result) == "Beta Corp"  # Klammerzusatz entfernt


def test_format_allocation_and_notes():
    portfolio = [
        {"name": "X", "yf_ticker": "X.DE", "target_weight": 0.655, "note": ""},
        {"name": "Y", "yf_ticker": "Y", "target_weight": 0.3, "note": "zyklisch"},
    ]
    assert pf.format_allocation(portfolio) == "X.DE / X 65.5%\nY / Y 30%"
    assert pf.format_asset_notes(portfolio) == "- Y / Y: zyklisch"
    assert "no position-specific" in pf.format_asset_notes(portfolio[:1])


def test_stock_names_without_stocks():
    assert pf.stock_names([{"name": "Fonds", "asset_class": "FUND"}]) == "Big Tech"


def test_load_profile(tmp_path, monkeypatch):
    _point_to(tmp_path, monkeypatch)
    assert pf.load_profile() == pf.EXAMPLE_PROFILE
    pf.PROFILE_MD.write_text("  Mein Profil\n", encoding="utf-8")
    assert pf.load_profile() == "Mein Profil"


def test_output_language(tmp_path, monkeypatch):
    _point_to(tmp_path, monkeypatch)
    monkeypatch.delenv("NEWSLETTER_LANGUAGE", raising=False)
    assert pf.output_language() == "English"  # default
    pf.PROFILE_MD.write_text("Profil", encoding="utf-8")
    assert pf.output_language() == "English"  # a profile alone doesn't switch language
    monkeypatch.setenv("NEWSLETTER_LANGUAGE", "German")
    assert pf.output_language() == "German"
    monkeypatch.setenv("NEWSLETTER_LANGUAGE", "French")
    assert pf.output_language() == "French"


def test_data_and_output_dirs_follow_env(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("INVESTMENT_DATA_DIR", str(tmp_path / "sample"))
    monkeypatch.setenv("NEWSLETTER_OUTPUT_DIR", str(tmp_path / "out"))
    reloaded = importlib.reload(pf)
    assert reloaded.MANUAL_DIR == (tmp_path / "sample" / "manual").resolve()
    assert reloaded.TARGET_WEIGHTS_CSV.parent == reloaded.MANUAL_DIR
    assert reloaded.OUTPUT_DIR == (tmp_path / "out").resolve()

    monkeypatch.delenv("INVESTMENT_DATA_DIR")
    monkeypatch.delenv("NEWSLETTER_OUTPUT_DIR")
    reloaded = importlib.reload(pf)
    assert reloaded.MANUAL_DIR == reloaded.PROJECT_ROOT / "data" / "manual"
