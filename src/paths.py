"""Data directories for notebooks and dashboard.

Everything reads and writes below DATA_DIR: `data/` by default, or the
directory in the INVESTMENT_DATA_DIR environment variable -- e.g. the
bundled sample data (`scripts/run_pipeline.sh --demo`), which then never
touches your own files in `data/`.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("INVESTMENT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
RAW_PATH = DATA_DIR / "raw"
MANUAL_PATH = DATA_DIR / "manual"
PROCESSED_PATH = DATA_DIR / "processed"
