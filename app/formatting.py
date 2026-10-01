"""App-side formatting/loading helpers used by app/dashboard.py and app/tabs/*.

The pure formatting functions live in src/formatting.py (no Streamlit
dependency, so they're unit-testable in isolation) and are re-exported
here for convenience alongside load_csv, which does depend on Streamlit
(st.warning) and therefore has to stay app-side.
"""

import pandas as pd
import streamlit as st

from src.formatting import (
    clean_table,
    color_change,
    format_asset_class,
    format_eur,
    format_pct_fraction,
    format_pct_points,
    format_pp,
)
from src.paths import PROCESSED_PATH

__all__ = [
    "clean_table",
    "color_change",
    "format_asset_class",
    "format_eur",
    "format_pct_fraction",
    "format_pct_points",
    "format_pp",
    "load_csv",
]

# data/processed by default, or $INVESTMENT_DATA_DIR/processed (e.g. sample data)
DATA_PATH = PROCESSED_PATH


def load_csv(filename, optional=False, **kwargs):
    """Read a processed CSV; empty DataFrame if missing. `optional` files
    (e.g. macro data when notebook 11 isn't run) skip the warning -- the tab
    explains the missing data itself."""
    path = DATA_PATH / filename

    if not path.exists():
        if not optional:
            st.warning(f"Missing file: {filename}")
        return pd.DataFrame()

    return pd.read_csv(path, **kwargs)
