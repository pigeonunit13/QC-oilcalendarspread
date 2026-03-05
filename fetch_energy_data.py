"""
fetch_energy_data.py
--------------------
Downloads comprehensive energy and economic data from the FRED and EIA APIs,
aligns everything to a weekly (Friday) frequency, and saves clean CSVs that are
ready for use in QuantConnect backtesting.

Output files
------------
data/cushing_inventory.csv    – Cushing-specific metrics
data/economic_indicators.csv  – FRED economic indicators
data/supply_demand.csv        – EIA supply/demand metrics
data/combined_features.csv    – All variables merged on weekly dates

Data sources
------------
FRED (Federal Reserve Bank of St. Louis)
  DTWEXBGS  – Nominal Broad US Dollar Index (daily)
  DGS10     – 10-Year Treasury Constant Maturity Rate (daily)
  VIXCLS    – CBOE Volatility Index: VIX (daily)
  GASDESW   – US Regular Conventional Gas Price (weekly, used for crack-spread proxy)
  DCOILWTICO – WTI Crude Oil Price (daily, used for crack-spread proxy)

EIA (US Energy Information Administration) – API v2
  PET.W_EPC0_SAX_YCUOK_MBBL.W   – Cushing, OK crude oil ending stocks (weekly, Mbbl)
  PET.W_EPC0_SAX_NUS_MBBL.W     – US total crude oil ending stocks (weekly, Mbbl)
  PET.W_EPC0_YOP_NUS_PER.W      – US refinery utilization rate (weekly, %)
  PET.W_EPC0_FPF_NUS_MBBLD.W    – US crude oil field production (weekly, Mbbl/d)
  PET.W_ERRPUS_SAX_NUS_MBBL.W   – US Strategic Petroleum Reserve (weekly, Mbbl)

Calculation methods
-------------------
cushing_seasonal_dev  – Cushing stocks minus the 5-year weekly seasonal average
cushing_delta_dev     – Week-over-week change in Cushing stocks
total_inventory_dev   – Total crude stocks minus the 5-year weekly seasonal average
production_change     – Week-over-week change in US crude production
crack_spread          – (GASDESW * 42 / 1000) - DCOILWTICO  ($/bbl, 1:1 crack proxy)
                        42 gallons per barrel; GASDESW is in $/gallon
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# API keys can be supplied via environment variables; the defaults below are
# the project keys provided in the issue.  Set FRED_API_KEY / EIA_API_KEY in
# your environment (or a .env file) to override them without editing source.
FRED_API_KEY = os.getenv("FRED_API_KEY", "0a2fa8c363e8e38521c2d80a895c7f8f")
EIA_API_KEY = os.getenv("EIA_API_KEY", "30ucZNJA1sPn9OcTEyfBWTMO11O19gfa9PA2qyaL")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
EIA_BASE_URL = "https://api.eia.gov/v2/seriesid/{series_id}"

# Observation start – pull full history from inception
OBS_START = "1900-01-01"
OBS_END = datetime.today().strftime("%Y-%m-%d")

# Output directory (created if absent)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Number of years used to compute seasonal averages
SEASONAL_WINDOW_YEARS = 5

# Seconds to sleep between API calls to avoid rate-limit errors
API_SLEEP = 0.5

# Max retries on transient HTTP errors
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_with_retry(url: str, params: dict, label: str) -> Optional[dict]:
    """GET *url* with *params*, retrying on 5xx / connection errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * attempt * 2
                log.warning("%s: rate-limited (429), waiting %ss …", label, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = RETRY_BACKOFF * attempt
                log.warning(
                    "%s: server error %s, waiting %ss …",
                    label,
                    resp.status_code,
                    wait,
                )
                time.sleep(wait)
                continue
            log.error(
                "%s: HTTP %s – %s", label, resp.status_code, resp.text[:200]
            )
            return None
        except requests.exceptions.RequestException as exc:
            wait = RETRY_BACKOFF * attempt
            log.warning("%s: request error (%s), waiting %ss …", label, exc, wait)
            time.sleep(wait)
    log.error("%s: all %s retries exhausted", label, MAX_RETRIES)
    return None


def _to_weekly_friday(series: pd.Series) -> pd.Series:
    """
    Resample a daily/monthly series to weekly-Friday frequency.

    Strategy
    --------
    * If original frequency appears daily-ish (median gap <= 7 days) → forward-fill
      to daily, then resample to Friday using the last observation of each week.
    * Otherwise (monthly etc.) → resample to Friday using forward-fill.
    """
    if series.empty:
        return pd.Series(dtype=float, name=series.name)

    series = series.sort_index().dropna()

    if len(series) > 1:
        gaps = series.index.to_series().diff().median()
        is_daily = gaps <= timedelta(days=7)
    else:
        is_daily = True

    if is_daily:
        daily = series.resample("D").ffill()
        weekly = daily.resample("W-FRI").last()
    else:
        weekly = series.resample("W-FRI").last().ffill()

    return weekly


def _compute_seasonal_deviation(
    weekly: pd.Series, window_years: int = SEASONAL_WINDOW_YEARS
) -> pd.Series:
    """
    Return the difference between *weekly* and its rolling seasonal average.

    The seasonal average for ISO-week *w* of year *y* is the mean of the
    observations for week *w* over the preceding *window_years* years.
    Weeks with fewer than 2 valid prior observations are returned as NaN.
    """
    if weekly.empty:
        return weekly.copy()

    df = weekly.to_frame("value")
    df["week"] = df.index.isocalendar().week.astype(int)
    df["year"] = df.index.year

    # Build a lookup table: (year, week) -> value
    lookup = df.set_index(["year", "week"])["value"]

    def _row_avg(row: pd.Series) -> float:
        y = int(row["year"])
        w = int(row["week"])
        prior_vals = [
            lookup.get((py, w))
            for py in range(y - window_years, y)
        ]
        valid = [v for v in prior_vals if v is not None and not np.isnan(v)]
        return float(np.mean(valid)) if len(valid) >= 2 else np.nan

    avg_series = df[["year", "week"]].apply(_row_avg, axis=1)
    avg_series.index = df.index
    deviation = df["value"] - avg_series
    deviation.name = weekly.name
    return deviation


# ---------------------------------------------------------------------------
# FRED fetcher
# ---------------------------------------------------------------------------


def fetch_fred(series_id: str) -> pd.Series:
    """
    Download a FRED series and return a *daily*-indexed pd.Series.
    Missing-value sentinel "." is converted to NaN.
    """
    log.info("FRED  →  %s", series_id)
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": OBS_START,
        "observation_end": OBS_END,
    }
    data = _request_with_retry(FRED_BASE_URL, params, f"FRED/{series_id}")
    time.sleep(API_SLEEP)

    if data is None or "observations" not in data:
        log.error("FRED/%s: no observations returned", series_id)
        return pd.Series(dtype=float, name=series_id)

    rows = [
        {"date": obs["date"], "value": obs["value"]}
        for obs in data["observations"]
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date").sort_index()
    series = df["value"].rename(series_id)
    log.info("  ✓  %s rows for %s  (%s → %s)",
             len(series.dropna()), series_id,
             series.first_valid_index(), series.last_valid_index())
    return series


# ---------------------------------------------------------------------------
# EIA v2 fetcher
# ---------------------------------------------------------------------------


def fetch_eia(series_id: str, label: str) -> pd.Series:
    """
    Download an EIA v2 series and return a weekly-indexed pd.Series.

    EIA v2 endpoint pattern:
        https://api.eia.gov/v2/seriesid/{series_id}?api_key=...&data[]=value
    """
    log.info("EIA   →  %s  (%s)", series_id, label)
    url = EIA_BASE_URL.format(series_id=series_id)
    params = {
        "api_key": EIA_API_KEY,
        "data[]": "value",
        "start": OBS_START,
        "end": OBS_END,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }

    all_rows: list[dict] = []
    while True:
        data = _request_with_retry(url, params, f"EIA/{series_id}")
        time.sleep(API_SLEEP)
        if data is None:
            break
        response = data.get("response", {})
        rows = response.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        total = response.get("total", len(all_rows))
        if len(all_rows) >= int(total):
            break
        params["offset"] = len(all_rows)

    if not all_rows:
        log.error("EIA/%s: no data returned", series_id)
        return pd.Series(dtype=float, name=label)

    df = pd.DataFrame(all_rows)
    # Period can be "2023-01-06" (weekly) or "2023-01" (monthly)
    df["period"] = pd.to_datetime(df["period"], infer_datetime_format=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("period").sort_index()
    series = df["value"].rename(label)
    log.info("  ✓  %s rows for %s  (%s → %s)",
             len(series.dropna()), label,
             series.first_valid_index(), series.last_valid_index())
    return series


# ---------------------------------------------------------------------------
# Data validation helpers
# ---------------------------------------------------------------------------


def _validate(df: pd.DataFrame, name: str) -> None:
    """Log basic quality stats for a DataFrame before saving."""
    total = len(df)
    if total == 0:
        log.warning("%s: DataFrame is EMPTY – check API responses", name)
        return
    pct_missing = df.isna().mean() * 100
    log.info("─── %s quality report (%s rows) ───", name, total)
    for col, pct in pct_missing.items():
        if pct > 0:
            log.info("  %-30s  %.1f %% missing", col, pct)

    # Gap detection: flag runs of consecutive NaN > 12 weeks
    for col in df.columns:
        null_runs = (
            df[col]
            .isna()
            .astype(int)
            .groupby((~df[col].isna()).cumsum())
            .sum()
        )
        long_gaps = null_runs[null_runs > 12]
        if not long_gaps.empty:
            log.warning(
                "%s / %s: %s gap(s) longer than 12 weeks detected",
                name, col, len(long_gaps),
            )


def _save(df: pd.DataFrame, filename: str) -> None:
    """Save *df* to *DATA_DIR/filename* as CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    log.info("Saved  %s  (%s rows × %s cols)", path, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    log.info("=" * 60)
    log.info("Energy & Economic Data Downloader")
    log.info("Target date range: %s → %s", OBS_START, OBS_END)
    log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1.  EIA weekly series                                               #
    # ------------------------------------------------------------------ #

    # Cushing, OK crude oil ending stocks (Mbbl)
    cushing_raw = fetch_eia("PET.W_EPC0_SAX_YCUOK_MBBL.W", "cushing_stocks")

    # US total crude oil ending stocks (Mbbl)
    total_inv_raw = fetch_eia("PET.W_EPC0_SAX_NUS_MBBL.W", "total_crude_stocks")

    # US refinery utilization (%)
    refinery_util = fetch_eia("PET.W_EPC0_YOP_NUS_PER.W", "refinery_util")

    # US crude oil field production (Mbbl/d)
    production_raw = fetch_eia("PET.W_EPC0_FPF_NUS_MBBLD.W", "production")

    # Strategic Petroleum Reserve (Mbbl)
    spr_level = fetch_eia("PET.W_ERRPUS_SAX_NUS_MBBL.W", "spr_level")

    # ------------------------------------------------------------------ #
    # 2.  FRED series                                                     #
    # ------------------------------------------------------------------ #

    dollar_index_raw = fetch_fred("DTWEXBGS")    # Nominal Broad US Dollar Index
    treasury_yield_raw = fetch_fred("DGS10")     # 10-Year Treasury CMT
    vix_raw = fetch_fred("VIXCLS")              # CBOE VIX
    gas_price_raw = fetch_fred("GASDESW")        # Regular Conventional Gas ($/gal, weekly)
    wti_raw = fetch_fred("DCOILWTICO")           # WTI Crude Oil Spot ($/bbl, daily)

    # ------------------------------------------------------------------ #
    # 3.  Frequency alignment → weekly Friday                             #
    # ------------------------------------------------------------------ #

    log.info("Aligning all series to weekly (Friday) frequency …")

    cushing_w = _to_weekly_friday(cushing_raw)
    total_inv_w = _to_weekly_friday(total_inv_raw)
    refinery_util_w = _to_weekly_friday(refinery_util)
    production_w = _to_weekly_friday(production_raw)
    spr_w = _to_weekly_friday(spr_level)

    dollar_index_w = _to_weekly_friday(dollar_index_raw)
    treasury_yield_w = _to_weekly_friday(treasury_yield_raw)
    vix_w = _to_weekly_friday(vix_raw)

    # Gas/WTI weekly for crack spread
    gas_w = _to_weekly_friday(gas_price_raw)
    wti_w = _to_weekly_friday(wti_raw)

    # ------------------------------------------------------------------ #
    # 4.  Derived variables                                               #
    # ------------------------------------------------------------------ #

    log.info("Computing derived features …")

    # Seasonal deviations
    cushing_seasonal_dev = _compute_seasonal_deviation(cushing_w).rename(
        "cushing_seasonal_dev"
    )
    total_inventory_dev = _compute_seasonal_deviation(total_inv_w).rename(
        "total_inventory_dev"
    )

    # Weekly changes
    cushing_delta_dev = cushing_w.diff().rename("cushing_delta_dev")
    production_change = production_w.diff().rename("production_change")

    # Crack spread proxy: 1-bbl gasoline value minus 1-bbl WTI
    # GASDESW is $/gallon → multiply by 42 to get $/bbl
    crack_spread = (gas_w * 42 - wti_w).rename("crack_spread")

    # ------------------------------------------------------------------ #
    # 5.  Build output DataFrames                                         #
    # ------------------------------------------------------------------ #

    # --- cushing_inventory.csv ---
    cushing_df = pd.concat(
        [cushing_seasonal_dev, cushing_delta_dev], axis=1
    )
    cushing_df.index.name = "date"
    _validate(cushing_df, "cushing_inventory")
    _save(cushing_df, "cushing_inventory.csv")

    # --- economic_indicators.csv ---
    econ_df = pd.concat(
        [
            dollar_index_w.rename("dollar_index"),
            treasury_yield_w.rename("treasury_yield"),
            vix_w.rename("vix"),
            crack_spread,
        ],
        axis=1,
    )
    econ_df.index.name = "date"
    _validate(econ_df, "economic_indicators")
    _save(econ_df, "economic_indicators.csv")

    # --- supply_demand.csv ---
    supply_df = pd.concat(
        [
            refinery_util_w.rename("refinery_util"),
            production_change,
            spr_w.rename("spr_level"),
            total_inventory_dev,
        ],
        axis=1,
    )
    supply_df.index.name = "date"
    _validate(supply_df, "supply_demand")
    _save(supply_df, "supply_demand.csv")

    # --- combined_features.csv ---
    combined_df = pd.concat(
        [
            cushing_seasonal_dev,
            cushing_delta_dev,
            total_inventory_dev,
            refinery_util_w.rename("refinery_util"),
            production_change,
            dollar_index_w.rename("dollar_index"),
            treasury_yield_w.rename("treasury_yield"),
            vix_w.rename("vix"),
            crack_spread,
            spr_w.rename("spr_level"),
        ],
        axis=1,
    )
    combined_df.index.name = "date"

    # Drop rows where ALL features are NaN (e.g. very early sparse dates)
    combined_df = combined_df.dropna(how="all")

    _validate(combined_df, "combined_features")
    _save(combined_df, "combined_features.csv")

    # ------------------------------------------------------------------ #
    # 6.  Summary                                                         #
    # ------------------------------------------------------------------ #

    log.info("=" * 60)
    log.info("Download complete.")
    log.info(
        "combined_features spans %s → %s  (%s weekly observations)",
        combined_df.index.min(),
        combined_df.index.max(),
        len(combined_df),
    )
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(0)
    except Exception:
        log.exception("Unexpected error during data download")
        sys.exit(1)
