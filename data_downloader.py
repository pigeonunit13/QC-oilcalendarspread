"""
One-time data downloader script.

Run this script **before** uploading the algorithm to QuantConnect to
pre-download EIA Cushing crude-oil inventory data and selected FRED
economic series.  The downloaded data is saved as CSV files in the
data/ directory and then uploaded to QC's Object Store or Object
Storage so that main_algorithm.py can load them without making live
network calls.

Usage
-----
    python data_downloader.py

Environment variables required
-------------------------------
    EIA_API_KEY   – EIA Open Data API key (https://www.eia.gov/opendata/)
    FRED_API_KEY  – FRED API key (https://fred.stlouisfed.org/docs/api/)

Both keys can also be placed in a local `secrets.env` file (not committed):
    EIA_API_KEY=your_eia_key
    FRED_API_KEY=your_fred_key
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from config import EIA_API_KEY, FRED_API_KEY, DATA_DIR, CUSHING_CSV, FRED_CSV
except ImportError:
    EIA_API_KEY = os.environ.get("EIA_API_KEY", "")
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    CUSHING_CSV = os.path.join(DATA_DIR, "cushing_inventory.csv")
    FRED_CSV = os.path.join(DATA_DIR, "fred_data.csv")

# EIA series for Cushing, OK crude oil stocks (weekly)
EIA_CUSHING_SERIES_ID = "PET.W_EPC0_SAX_YCUOK_MBBL.W"

# FRED series to download
FRED_SERIES: Dict[str, str] = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI)",
    "DCOILBRENTEU": "Crude Oil Prices: Brent - Europe",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch a URL and return the raw bytes body."""
    log.info("GET %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "QC-oilcalendarspread/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ensure_data_dir() -> None:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# EIA downloader
# ---------------------------------------------------------------------------


def download_cushing_inventory(api_key: str, output_path: str) -> bool:
    """
    Download weekly Cushing crude-oil inventory from the EIA v2 API and
    save the result as a two-column CSV (date, inventory_mbbl).

    Returns True on success, False on failure.
    """
    if not api_key:
        log.warning("EIA_API_KEY is not set – skipping Cushing inventory download.")
        return False

    # EIA v2 API endpoint
    url = (
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        f"?api_key={api_key}"
        "&frequency=weekly"
        "&data[0]=value"
        "&facets[series][]=W_EPC0_SAX_YCUOK_MBBL"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        "&offset=0"
        "&length=5000"
    )

    try:
        raw = _fetch_url(url)
        payload = json.loads(raw)
        records: List[dict] = payload.get("response", {}).get("data", [])
        if not records:
            log.error("EIA returned no records for Cushing inventory series.")
            return False

        _ensure_data_dir()
        with open(output_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "inventory_mbbl"])
            for rec in records:
                date_str = rec.get("period", "")
                value = rec.get("value", "")
                if date_str and value not in ("", None):
                    writer.writerow([date_str, value])

        log.info("Saved Cushing inventory → %s  (%d rows)", output_path, len(records))
        return True

    except Exception as exc:  # noqa: BLE001
        log.error("Failed to download Cushing inventory: %s", exc)
        return False


# ---------------------------------------------------------------------------
# FRED downloader
# ---------------------------------------------------------------------------


def download_fred_series(api_key: str, series_dict: Dict[str, str], output_path: str) -> bool:
    """
    Download one or more FRED series and save them as a CSV with columns:
        date, <series_id1>, <series_id2>, ...

    Series are aligned by date (outer join) and missing values are left blank.

    Returns True on success, False on failure.
    """
    if not api_key:
        log.warning("FRED_API_KEY is not set – skipping FRED data download.")
        return False

    all_data: Dict[str, Dict[str, str]] = {}  # {date: {series_id: value}}

    for series_id in series_dict:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={api_key}"
            "&file_type=json"
            "&sort_order=asc"
        )
        try:
            raw = _fetch_url(url)
            payload = json.loads(raw)
            observations = payload.get("observations", [])
            for obs in observations:
                date = obs.get("date", "")
                value = obs.get("value", ".")
                if date:
                    if date not in all_data:
                        all_data[date] = {}
                    # FRED uses "." for missing values
                    all_data[date][series_id] = "" if value == "." else value
            log.info("  Series %s – %d observations", series_id, len(observations))
        except Exception as exc:  # noqa: BLE001
            log.warning("  Failed to download FRED series %s: %s", series_id, exc)

    if not all_data:
        log.error("No FRED data was downloaded.")
        return False

    _ensure_data_dir()
    series_ids = list(series_dict.keys())
    sorted_dates = sorted(all_data.keys())

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date"] + series_ids)
        for date in sorted_dates:
            row = [date] + [all_data[date].get(sid, "") for sid in series_ids]
            writer.writerow(row)

    log.info(
        "Saved FRED data → %s  (%d dates, %d series)",
        output_path,
        len(sorted_dates),
        len(series_ids),
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("=== QC Oil Calendar Spread – Data Downloader ===")

    # Load keys from secrets.env if present (not committed to version control)
    secrets_file = os.path.join(os.path.dirname(__file__), "secrets.env")
    if os.path.isfile(secrets_file):
        with open(secrets_file) as fh:
            for line in fh:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    eia_key = os.environ.get("EIA_API_KEY", EIA_API_KEY)
    fred_key = os.environ.get("FRED_API_KEY", FRED_API_KEY)

    success_eia = download_cushing_inventory(eia_key, CUSHING_CSV)
    success_fred = download_fred_series(fred_key, FRED_SERIES, FRED_CSV)

    if not success_eia and not success_fred:
        log.error(
            "Both downloads failed.  "
            "Ensure EIA_API_KEY and FRED_API_KEY are set and retry."
        )
        sys.exit(1)

    log.info("Download complete.  Upload the data/ directory to your QC project.")


if __name__ == "__main__":
    main()
