import pandas as pd

from src.formatting import (
    clean_table,
    color_change,
    format_asset_class,
    format_eur,
    format_pct_fraction,
    format_pct_points,
    format_pp,
)


def test_format_eur_uses_german_thousands_and_decimal_separators():
    assert format_eur(1234.5) == "1.234,50 €"


def test_format_eur_handles_nan():
    assert format_eur(float("nan")) == ""


def test_format_eur_negative_value():
    assert format_eur(-42.1) == "-42,10 €"


def test_format_pct_fraction_multiplies_by_100():
    assert format_pct_fraction(0.1234) == "12.3%"


def test_format_pct_fraction_handles_nan():
    assert format_pct_fraction(float("nan")) == ""


def test_format_pct_points_does_not_multiply():
    assert format_pct_points(12.34) == "12.3%"


def test_format_pp_matches_format_pct_points():
    assert format_pp(5.0) == format_pct_points(5.0) == "5.0%"


def test_format_asset_class_known_values():
    assert format_asset_class("FUND") == "Fund"
    assert format_asset_class("STOCK") == "Stock"
    assert format_asset_class("CRYPTO") == "Crypto"


def test_format_asset_class_unknown_value_is_titlecased():
    assert format_asset_class("etf") == "Etf"


def test_format_asset_class_handles_nan():
    assert format_asset_class(float("nan")) == ""


def test_color_change_positive_negative_zero():
    assert color_change(1.0) == "color: #2ecc71"
    assert color_change(-1.0) == "color: #ff6b6b"
    assert color_change(0.0) == ""


def test_color_change_handles_nan():
    assert color_change(float("nan")) == ""


def test_clean_table_formats_and_renames_selected_columns():
    df = pd.DataFrame({
        "name": ["MSCI World", "Bitcoin"],
        "market_value": [1000.0, 500.5],
        "actual_weight": [0.6, 0.4],
        "irrelevant": ["x", "y"],
    })

    out = clean_table(
        df,
        columns=["name", "market_value", "actual_weight"],
        rename_map={"name": "Asset", "market_value": "Market Value", "actual_weight": "Weight"},
        euro_cols=["market_value"],
        pct_fraction_cols=["actual_weight"],
    )

    assert list(out.columns) == ["Asset", "Market Value", "Weight"]
    assert out.loc[0, "Market Value"] == "1.000,00 €"
    assert out.loc[0, "Weight"] == "60.0%"
    assert "irrelevant" not in out.columns


def test_clean_table_ignores_missing_columns():
    df = pd.DataFrame({"a": [1, 2]})

    out = clean_table(df, columns=["a", "does_not_exist"])

    assert list(out.columns) == ["a"]


def test_clean_table_round_cols():
    df = pd.DataFrame({"shares": [1.23456, 2.0]})

    out = clean_table(df, columns=["shares"], round_cols=["shares"])

    assert out.loc[0, "shares"] == 1.235
