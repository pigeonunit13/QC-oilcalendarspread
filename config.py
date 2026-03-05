"""
Configuration constants for the Oil Calendar Spread algorithm.

API keys are read from environment variables or a local secrets file that is
NOT committed to version control.  The algorithm itself (main_algorithm.py)
never calls external APIs at run-time; all data is loaded from pre-downloaded
CSV files in the data/ directory.
"""

import os

# ---------------------------------------------------------------------------
# API keys – used only by data_downloader.py (offline, one-time use)
# ---------------------------------------------------------------------------
# Set these as environment variables before running data_downloader.py:
#   export EIA_API_KEY="your_key_here"
#   export FRED_API_KEY="your_key_here"
EIA_API_KEY: str = os.environ.get("EIA_API_KEY", "")
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
CUSHING_CSV: str = os.path.join(DATA_DIR, "cushing_inventory.csv")
FRED_CSV: str = os.path.join(DATA_DIR, "fred_data.csv")

# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------
# Front-month and back-month futures contracts traded on NYMEX
FRONT_MONTH_TICKER: str = "CL1"   # Crude Oil front month
BACK_MONTH_TICKER: str = "CL2"    # Crude Oil second month

# Spread signal thresholds (contango / backwardation)
ENTRY_Z_SCORE: float = 1.5        # Enter when |z-score| exceeds this value
EXIT_Z_SCORE: float = 0.5         # Exit when |z-score| falls below this value
LOOKBACK_DAYS: int = 252          # Rolling window for z-score calculation

# Position sizing
MAX_POSITION_SIZE: float = 0.25   # Maximum fraction of portfolio per leg

# Inventory signal parameters
INVENTORY_LOOKBACK_WEEKS: int = 52  # Weeks of inventory history for normalisation
