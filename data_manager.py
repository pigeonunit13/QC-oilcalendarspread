"""
CSV-based fundamental data manager for the Oil Calendar Spread algorithm.

This module replaces any previous implementation that called external APIs
(EIA, FRED) at run-time.  All data is loaded once from pre-downloaded CSV
files in the data/ directory, making the algorithm fully compatible with the
QuantConnect backtester which cannot make outbound network requests.

Public API
----------
FundamentalDataManager(algorithm, cushing_csv, fred_csv)
    .load_all()                   – load all CSV files into memory
    .get_cushing_inventory(date)  – return inventory level for a date
    .get_fred_value(series, date) – return FRED series value for a date
    .get_inventory_z_score(date)  – rolling z-score of inventory vs history
    .is_data_available(date)      – True if fundamental data exists near date
"""

from __future__ import annotations

import csv
import logging
import math
import os
from datetime import date, datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y")
_FALLBACK_LOOKBACK_WEEKS = 52  # weeks used for z-score when config unavailable


def _parse_date(s: str) -> Optional[date]:
    """Try several common date formats; return None on failure."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _safe_float(s: str) -> Optional[float]:
    """Convert string to float; return None for empty / non-numeric values."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# FundamentalDataManager
# ---------------------------------------------------------------------------


class FundamentalDataManager:
    """
    Loads Cushing crude-oil inventory and FRED economic data from local CSV
    files and provides date-indexed lookups with nearest-date fallback.

    Parameters
    ----------
    algorithm : QCAlgorithm (or None for testing)
        Reference to the QC algorithm instance.  Used for logging only.
    cushing_csv : str
        Absolute path to the Cushing inventory CSV
        (columns: date, inventory_mbbl).
    fred_csv : str
        Absolute path to the FRED data CSV
        (columns: date, <series_id>, ...).
    lookback_weeks : int
        Number of historical weeks to use when computing the inventory
        z-score.  Defaults to 52.
    """

    def __init__(
        self,
        algorithm,
        cushing_csv: str,
        fred_csv: str,
        lookback_weeks: int = _FALLBACK_LOOKBACK_WEEKS,
    ) -> None:
        self._algo = algorithm
        self._cushing_csv = cushing_csv
        self._fred_csv = fred_csv
        self._lookback_weeks = lookback_weeks

        # Ordered lists so we can do nearest-date lookups efficiently
        self._cushing_dates: List[date] = []
        self._cushing_values: List[float] = []  # mbbls

        # {series_id: {date: float}}
        self._fred_data: Dict[str, Dict[date, float]] = {}
        self._fred_dates: Dict[str, List[date]] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all CSV files into memory.  Call once during initialize()."""
        self._load_cushing()
        self._load_fred()
        self._loaded = True

    def get_cushing_inventory(self, query_date: date) -> Optional[float]:
        """
        Return the most-recent Cushing inventory (mbbls) on or before
        *query_date*.  Returns None if no data is available.
        """
        if not self._cushing_dates:
            return self._simulated_inventory(query_date)
        idx = self._nearest_index(self._cushing_dates, query_date)
        if idx is None:
            return None
        return self._cushing_values[idx]

    def get_fred_value(self, series_id: str, query_date: date) -> Optional[float]:
        """
        Return the most-recent observation for *series_id* on or before
        *query_date*.  Returns None if the series or date is unavailable.
        """
        if series_id not in self._fred_data:
            return None
        dates = self._fred_dates[series_id]
        if not dates:
            return None
        idx = self._nearest_index(dates, query_date)
        if idx is None:
            return None
        return self._fred_data[series_id][dates[idx]]

    def get_inventory_z_score(self, query_date: date) -> Optional[float]:
        """
        Compute the z-score of the current Cushing inventory relative to the
        trailing *lookback_weeks* of history.

        Returns None when insufficient data is available.
        """
        current = self.get_cushing_inventory(query_date)
        if current is None:
            return None

        cutoff = query_date - timedelta(weeks=self._lookback_weeks)
        history = [
            v
            for d, v in zip(self._cushing_dates, self._cushing_values)
            if cutoff <= d <= query_date
        ]
        if len(history) < 4:
            # Not enough history; fall back to simulated z-score
            return self._simulated_z_score(query_date)

        mu = mean(history)
        sigma = stdev(history)
        if sigma == 0:
            return 0.0
        return (current - mu) / sigma

    def is_data_available(self, query_date: date) -> bool:
        """Return True if fundamental data exists within 14 days of *query_date*."""
        if not self._cushing_dates:
            return False
        idx = self._nearest_index(self._cushing_dates, query_date)
        if idx is None:
            return False
        delta = abs((self._cushing_dates[idx] - query_date).days)
        return delta <= 14

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_cushing(self) -> None:
        if not os.path.isfile(self._cushing_csv):
            self._log_warning(
                f"Cushing inventory CSV not found: {self._cushing_csv}. "
                "Run data_downloader.py to generate it. "
                "Falling back to simulated data."
            )
            return

        rows: List[Tuple[date, float]] = []
        try:
            with open(self._cushing_csv, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    d = _parse_date(row.get("date", ""))
                    v = _safe_float(row.get("inventory_mbbl", ""))
                    if d is not None and v is not None:
                        rows.append((d, v))
        except Exception as exc:  # noqa: BLE001
            self._log_error(f"Error reading Cushing CSV: {exc}")
            return

        if not rows:
            self._log_warning("Cushing inventory CSV contains no valid rows.")
            return

        rows.sort(key=lambda x: x[0])
        self._cushing_dates = [d for d, _ in rows]
        self._cushing_values = [v for _, v in rows]
        self._log_info(
            f"Loaded {len(rows)} Cushing inventory rows "
            f"({self._cushing_dates[0]} – {self._cushing_dates[-1]})."
        )

    def _load_fred(self) -> None:
        if not os.path.isfile(self._fred_csv):
            self._log_warning(
                f"FRED data CSV not found: {self._fred_csv}. "
                "Run data_downloader.py to generate it."
            )
            return

        try:
            with open(self._fred_csv, newline="") as fh:
                reader = csv.DictReader(fh)
                series_ids = [col for col in (reader.fieldnames or []) if col != "date"]
                for sid in series_ids:
                    self._fred_data[sid] = {}

                for row in reader:
                    d = _parse_date(row.get("date", ""))
                    if d is None:
                        continue
                    for sid in series_ids:
                        v = _safe_float(row.get(sid, ""))
                        if v is not None:
                            self._fred_data[sid][d] = v

        except Exception as exc:  # noqa: BLE001
            self._log_error(f"Error reading FRED CSV: {exc}")
            return

        for sid in list(self._fred_data.keys()):
            sorted_dates = sorted(self._fred_data[sid].keys())
            self._fred_dates[sid] = sorted_dates
            self._log_info(
                f"Loaded FRED series {sid}: {len(sorted_dates)} observations."
            )

    # ------------------------------------------------------------------
    # Nearest-date lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_index(sorted_dates: List[date], query: date) -> Optional[int]:
        """
        Binary-search *sorted_dates* for the largest date <= *query*.
        Returns None if all dates are strictly after *query*.
        """
        lo, hi = 0, len(sorted_dates) - 1
        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_dates[mid] <= query:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # ------------------------------------------------------------------
    # Simulated fallback data (deterministic, no API calls)
    # ------------------------------------------------------------------

    @staticmethod
    def _simulated_inventory(query_date: date) -> float:
        """
        Return a plausible simulated Cushing inventory (mbbls) based on
        seasonal patterns when real CSV data is unavailable.

        The values are derived from long-run historical averages and are
        suitable only for development / smoke-testing.
        """
        # Seasonal: peak ~April, trough ~September (Northern Hemisphere)
        day_of_year = query_date.timetuple().tm_yday
        seasonal = math.cos(2 * math.pi * (day_of_year - 100) / 365)
        # Typical Cushing range ~20 – 60 million barrels; centre ~40 Mbbls
        return round(40_000 + 10_000 * seasonal, 1)

    @staticmethod
    def _simulated_z_score(query_date: date) -> float:
        """Return a simulated inventory z-score when history is insufficient."""
        day_of_year = query_date.timetuple().tm_yday
        return round(math.sin(2 * math.pi * day_of_year / 365), 4)

    # ------------------------------------------------------------------
    # Logging helpers (work with or without a QC algorithm object)
    # ------------------------------------------------------------------

    def _log_info(self, msg: str) -> None:
        if self._algo is not None and hasattr(self._algo, "log"):
            self._algo.log(msg)
        else:
            log.info(msg)

    def _log_warning(self, msg: str) -> None:
        if self._algo is not None and hasattr(self._algo, "log"):
            self._algo.log(f"WARNING: {msg}")
        else:
            log.warning(msg)

    def _log_error(self, msg: str) -> None:
        if self._algo is not None and hasattr(self._algo, "log"):
            self._algo.log(f"ERROR: {msg}")
        else:
            log.error(msg)
