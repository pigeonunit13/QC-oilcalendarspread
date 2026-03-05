"""
tests/test_fetch_energy_data.py
-------------------------------
Unit tests for the non-network helper functions in fetch_energy_data.py.
Run with:  python -m pytest tests/ -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_energy_data import (
    _to_weekly_friday,
    _compute_seasonal_deviation,
    _validate,
    SEASONAL_WINDOW_YEARS,
)


# ---------------------------------------------------------------------------
# _to_weekly_friday
# ---------------------------------------------------------------------------


class TestToWeeklyFriday:
    def test_daily_resamples_to_friday(self):
        idx = pd.date_range("2020-01-01", periods=90, freq="D")
        s = pd.Series(range(90), index=idx, name="test")
        result = _to_weekly_friday(s)
        assert result.index.freqstr in ("W-FRI",) or all(
            d.weekday() == 4 for d in result.index
        ), "All dates should be Fridays"

    def test_empty_series_returns_empty(self):
        s = pd.Series(dtype=float, name="empty")
        result = _to_weekly_friday(s)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_weekly_source_stays_weekly(self):
        idx = pd.date_range("2020-01-03", periods=52, freq="W-FRI")
        s = pd.Series(np.random.randn(52), index=idx, name="weekly")
        result = _to_weekly_friday(s)
        assert len(result) == 52
        assert all(d.weekday() == 4 for d in result.index)

    def test_values_preserved_weekly_source(self):
        # A weekly series should not alter values significantly
        idx = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
        vals = np.arange(1.0, 11.0)
        s = pd.Series(vals, index=idx, name="v")
        result = _to_weekly_friday(s)
        np.testing.assert_array_almost_equal(result.values, vals)

    def test_monthly_series_forward_filled(self):
        # Monthly series should be forward-filled into weekly
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        s = pd.Series(range(24), index=idx, name="monthly")
        result = _to_weekly_friday(s)
        # Should have ~24*4 = ~96 weekly rows
        assert len(result) >= 80


# ---------------------------------------------------------------------------
# _compute_seasonal_deviation
# ---------------------------------------------------------------------------


class TestComputeSeasonalDeviation:
    def _make_weekly_series(self, years: int = 8, base_value: float = 100.0) -> pd.Series:
        """Create a constant weekly series spanning *years* years."""
        idx = pd.date_range("2015-01-02", periods=52 * years, freq="W-FRI")
        return pd.Series(base_value, index=idx, name="stocks")

    def test_constant_series_deviation_is_zero(self):
        s = self._make_weekly_series(years=8, base_value=200.0)
        dev = _compute_seasonal_deviation(s, window_years=5)
        # After the initial warm-up period the deviation must be exactly 0
        valid = dev.dropna()
        assert len(valid) > 0, "Should have valid observations"
        np.testing.assert_array_almost_equal(
            valid.values, 0.0, decimal=6,
            err_msg="Constant series should produce zero seasonal deviation",
        )

    def test_returns_series_same_index(self):
        s = self._make_weekly_series(years=8)
        dev = _compute_seasonal_deviation(s)
        assert isinstance(dev, pd.Series)
        assert dev.index.equals(s.index)

    def test_empty_series_returns_empty(self):
        s = pd.Series(dtype=float, name="empty")
        dev = _compute_seasonal_deviation(s)
        assert len(dev) == 0

    def test_insufficient_history_produces_nan(self):
        # Only 2 years of data -> first year has no 5-year lookback -> all NaN
        idx = pd.date_range("2020-01-03", periods=52, freq="W-FRI")
        s = pd.Series(100.0, index=idx, name="short")
        dev = _compute_seasonal_deviation(s, window_years=5)
        # Everything should be NaN since we have zero prior years
        assert dev.isna().all(), "With no prior years all values should be NaN"

    def test_step_change_reflected_in_deviation(self):
        """
        If the series jumps 50 units above its historical average for a given
        week, the seasonal deviation for those weeks should be +50.
        """
        n_years = 8
        idx = pd.date_range("2015-01-02", periods=52 * n_years, freq="W-FRI")
        base = pd.Series(100.0, index=idx, name="stocks")
        # Add 50 to the last full year
        bump_start = pd.Timestamp("2022-01-01")
        bumped = base.copy()
        bumped[bumped.index >= bump_start] = 150.0
        dev = _compute_seasonal_deviation(bumped, window_years=5)
        valid_bump = dev[(dev.index >= bump_start) & dev.notna()]
        # With 5 prior years of 100 the average is 100, so deviation >= 45
        assert (
            valid_bump > 45
        ).all(), "Step-up years should show positive seasonal deviation"


# ---------------------------------------------------------------------------
# _validate  (smoke test - just checks it doesn't raise)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_smoke_no_exception(self):
        idx = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
        df = pd.DataFrame(
            {"a": np.random.randn(20), "b": np.random.randn(20)}, index=idx
        )
        df.iloc[2, 0] = float("nan")  # inject one gap
        _validate(df, "test_df")  # should log but not raise

    def test_empty_dataframe_no_exception(self):
        _validate(pd.DataFrame(), "empty")
