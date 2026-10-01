"""Shared display-formatting helpers for tables shown in the Streamlit dashboard.

Extracted from app/dashboard.py so the same formatting logic can be unit
tested and reused without pulling in a Streamlit dependency.
"""

import pandas as pd


def format_eur(value):
    if pd.isna(value):
        return ""
    return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct_fraction(value):
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def format_pct_points(value):
    if pd.isna(value):
        return ""
    return f"{value:.1f}%"


def format_pp(value):
    if pd.isna(value):
        return ""
    return f"{value:.1f}%"


def format_asset_class(value):
    mapping = {
        "FUND": "Fund",
        "STOCK": "Stock",
        "CRYPTO": "Crypto"
    }

    if pd.isna(value):
        return ""

    return mapping.get(value, str(value).title())


def color_change(value):
    if pd.isna(value):
        return ""

    if value > 0:
        return "color: #2ecc71"

    if value < 0:
        return "color: #ff6b6b"

    return ""


def clean_table(
    df,
    columns,
    rename_map=None,
    euro_cols=None,
    pct_fraction_cols=None,
    pct_point_cols=None,
    pp_cols=None,
    round_cols=None
):
    rename_map = rename_map or {}
    euro_cols = euro_cols or []
    pct_fraction_cols = pct_fraction_cols or []
    pct_point_cols = pct_point_cols or []
    pp_cols = pp_cols or []
    round_cols = round_cols or []

    existing_cols = [col for col in columns if col in df.columns]
    out = df[existing_cols].copy()

    for col in euro_cols:
        if col in out.columns:
            out[col] = out[col].apply(format_eur)

    for col in pct_fraction_cols:
        if col in out.columns:
            out[col] = out[col].apply(format_pct_fraction)

    for col in pct_point_cols:
        if col in out.columns:
            out[col] = out[col].apply(format_pct_points)

    for col in pp_cols:
        if col in out.columns:
            out[col] = out[col].apply(format_pp)

    for col in round_cols:
        if col in out.columns:
            out[col] = out[col].round(3)

    out = out.rename(columns=rename_map)

    return out
