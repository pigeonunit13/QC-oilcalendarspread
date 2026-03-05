"""
Tests for data_manager.py and config.py.

These tests run without a QuantConnect environment and without any network
access.  They validate:
  - CSV parsing and date lookup logic
  - Fallback to simulated data when CSVs are absent
  - Z-score computation
  - Config constants have expected types
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import date

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cushing_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "inventory_mbbl"])
        writer.writerows(rows)


def _write_fred_csv(path: str, series: list, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date"] + series)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_imports() -> None:
    from config import (
        CUSHING_CSV,
        ENTRY_Z_SCORE,
        EXIT_Z_SCORE,
        FRED_CSV,
        LOOKBACK_DAYS,
        MAX_POSITION_SIZE,
    )

    assert isinstance(ENTRY_Z_SCORE, float)
    assert isinstance(EXIT_Z_SCORE, float)
    assert ENTRY_Z_SCORE > EXIT_Z_SCORE, "ENTRY threshold must exceed EXIT threshold"
    assert isinstance(LOOKBACK_DAYS, int) and LOOKBACK_DAYS > 0
    assert 0 < MAX_POSITION_SIZE <= 1.0
    assert CUSHING_CSV.endswith(".csv")
    assert FRED_CSV.endswith(".csv")


def test_no_api_keys_in_config_defaults() -> None:
    """API keys must not be hard-coded; defaults must be empty strings."""
    import importlib

    # Unset any real keys so we test the hard-coded default
    saved_eia = os.environ.pop("EIA_API_KEY", None)
    saved_fred = os.environ.pop("FRED_API_KEY", None)
    try:
        import config
        importlib.reload(config)
        assert config.EIA_API_KEY == "", "EIA_API_KEY should default to empty string"
        assert config.FRED_API_KEY == "", "FRED_API_KEY should default to empty string"
    finally:
        if saved_eia is not None:
            os.environ["EIA_API_KEY"] = saved_eia
        if saved_fred is not None:
            os.environ["FRED_API_KEY"] = saved_fred


# ---------------------------------------------------------------------------
# FundamentalDataManager – CSV loading tests
# ---------------------------------------------------------------------------


class TestCushingLoad:
    def test_loads_valid_csv(self, tmp_path):
        from data_manager import FundamentalDataManager

        csv_path = str(tmp_path / "cushing.csv")
        _write_cushing_csv(csv_path, [
            ("2022-01-07", "35000.5"),
            ("2022-01-14", "36000.0"),
            ("2022-01-21", "34500.0"),
        ])
        mgr = FundamentalDataManager(None, csv_path, str(tmp_path / "fred.csv"))
        mgr.load_all()

        assert mgr.get_cushing_inventory(date(2022, 1, 14)) == pytest.approx(36000.0)

    def test_nearest_date_before_query(self, tmp_path):
        from data_manager import FundamentalDataManager

        csv_path = str(tmp_path / "cushing.csv")
        _write_cushing_csv(csv_path, [
            ("2022-01-07", "35000.0"),
            ("2022-01-14", "36000.0"),
        ])
        mgr = FundamentalDataManager(None, csv_path, str(tmp_path / "fred.csv"))
        mgr.load_all()

        # Query on a date between rows -> should return the earlier row
        val = mgr.get_cushing_inventory(date(2022, 1, 10))
        assert val == pytest.approx(35000.0)

    def test_query_before_all_data_returns_none(self, tmp_path):
        from data_manager import FundamentalDataManager

        csv_path = str(tmp_path / "cushing.csv")
        _write_cushing_csv(csv_path, [("2022-06-01", "40000.0")])
        mgr = FundamentalDataManager(None, csv_path, str(tmp_path / "fred.csv"))
        mgr.load_all()

        assert mgr.get_cushing_inventory(date(2021, 1, 1)) is None

    def test_missing_csv_uses_simulated_fallback(self, tmp_path):
        from data_manager import FundamentalDataManager

        mgr = FundamentalDataManager(
            None,
            str(tmp_path / "nonexistent.csv"),
            str(tmp_path / "fred_nonexistent.csv"),
        )
        mgr.load_all()

        # Should return a positive float (simulated value)
        val = mgr.get_cushing_inventory(date(2022, 6, 1))
        assert val is not None and val > 0

    def test_is_data_available_true(self, tmp_path):
        from data_manager import FundamentalDataManager

        csv_path = str(tmp_path / "cushing.csv")
        _write_cushing_csv(csv_path, [("2022-01-07", "35000.0")])
        mgr = FundamentalDataManager(None, csv_path, str(tmp_path / "fred.csv"))
        mgr.load_all()

        assert mgr.is_data_available(date(2022, 1, 10)) is True

    def test_is_data_available_false_when_no_csv(self, tmp_path):
        from data_manager import FundamentalDataManager

        mgr = FundamentalDataManager(
            None,
            str(tmp_path / "nonexistent.csv"),
            str(tmp_path / "fred_nonexistent.csv"),
        )
        mgr.load_all()
        assert mgr.is_data_available(date(2022, 1, 10)) is False


class TestFredLoad:
    def test_loads_valid_fred_csv(self, tmp_path):
        from data_manager import FundamentalDataManager

        fred_path = str(tmp_path / "fred.csv")
        _write_fred_csv(
            fred_path,
            ["DCOILWTICO", "DGS10"],
            [
                ["2022-01-03", "76.50", "1.63"],
                ["2022-01-04", "77.00", "1.70"],
            ],
        )
        mgr = FundamentalDataManager(None, str(tmp_path / "cushing.csv"), fred_path)
        mgr.load_all()

        assert mgr.get_fred_value("DCOILWTICO", date(2022, 1, 4)) == pytest.approx(77.0)
        assert mgr.get_fred_value("DGS10", date(2022, 1, 3)) == pytest.approx(1.63)

    def test_missing_series_returns_none(self, tmp_path):
        from data_manager import FundamentalDataManager

        fred_path = str(tmp_path / "fred.csv")
        _write_fred_csv(fred_path, ["DCOILWTICO"], [["2022-01-03", "76.50"]])
        mgr = FundamentalDataManager(None, str(tmp_path / "cushing.csv"), fred_path)
        mgr.load_all()

        assert mgr.get_fred_value("NONEXISTENT", date(2022, 1, 3)) is None

    def test_blank_cells_are_skipped(self, tmp_path):
        from data_manager import FundamentalDataManager

        fred_path = str(tmp_path / "fred.csv")
        _write_fred_csv(
            fred_path,
            ["DCOILWTICO"],
            [
                ["2022-01-03", ""],    # blank value - should be skipped
                ["2022-01-04", "77.0"],
            ],
        )
        mgr = FundamentalDataManager(None, str(tmp_path / "cushing.csv"), fred_path)
        mgr.load_all()

        assert mgr.get_fred_value("DCOILWTICO", date(2022, 1, 3)) is None
        assert mgr.get_fred_value("DCOILWTICO", date(2022, 1, 4)) == pytest.approx(77.0)


class TestZScore:
    def test_z_score_with_sufficient_history(self, tmp_path):
        import datetime
        from data_manager import FundamentalDataManager

        csv_path = str(tmp_path / "cushing.csv")
        # Generate 60 weekly data points
        rows = []
        base = date(2021, 1, 1)
        for i in range(60):
            d = base + datetime.timedelta(weeks=i)
            rows.append((d.isoformat(), str(40000.0 + i * 10)))
        _write_cushing_csv(csv_path, rows)

        mgr = FundamentalDataManager(
            None, csv_path, str(tmp_path / "fred.csv"), lookback_weeks=52
        )
        mgr.load_all()

        z = mgr.get_inventory_z_score(date(2022, 1, 1))
        assert z is not None
        assert isinstance(z, float)

    def test_z_score_fallback_to_simulated_when_no_csv(self, tmp_path):
        from data_manager import FundamentalDataManager

        mgr = FundamentalDataManager(
            None,
            str(tmp_path / "nonexistent.csv"),
            str(tmp_path / "fred_nonexistent.csv"),
        )
        mgr.load_all()

        z = mgr.get_inventory_z_score(date(2022, 6, 1))
        assert z is not None
        assert -2 <= z <= 2  # plausible sine-wave value


# ---------------------------------------------------------------------------
# data_manager - private helper: _nearest_index
# ---------------------------------------------------------------------------


class TestNearestIndex:
    def test_exact_match(self):
        from data_manager import FundamentalDataManager

        dates = [date(2022, 1, 1), date(2022, 1, 8), date(2022, 1, 15)]
        assert FundamentalDataManager._nearest_index(dates, date(2022, 1, 8)) == 1

    def test_between_dates_returns_earlier(self):
        from data_manager import FundamentalDataManager

        dates = [date(2022, 1, 1), date(2022, 1, 8), date(2022, 1, 15)]
        assert FundamentalDataManager._nearest_index(dates, date(2022, 1, 5)) == 0

    def test_before_all_returns_none(self):
        from data_manager import FundamentalDataManager

        dates = [date(2022, 1, 8), date(2022, 1, 15)]
        assert FundamentalDataManager._nearest_index(dates, date(2022, 1, 1)) is None

    def test_after_all_returns_last(self):
        from data_manager import FundamentalDataManager

        dates = [date(2022, 1, 1), date(2022, 1, 8)]
        assert FundamentalDataManager._nearest_index(dates, date(2022, 12, 31)) == 1
