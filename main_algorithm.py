"""
Oil Calendar Spread – Mean-Reversion Algorithm
===============================================
QuantConnect (Lean) algorithm that trades the WTI crude-oil calendar spread
(front month vs second month) using:

  * A z-score of the price spread for entry / exit signals
  * Cushing crude-oil inventory data loaded from a pre-downloaded CSV file
  * FRED economic data loaded from a pre-downloaded CSV file

No external API calls are made at run-time.  All fundamental data is loaded
from local CSV files in the data/ directory during initialize().

Setup
-----
1. Run ``python data_downloader.py`` locally to generate the CSV files.
2. Upload the entire project (including the data/ folder) to QuantConnect.
3. Run the backtest.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# QuantConnect imports (available inside the QC cloud environment)
# ---------------------------------------------------------------------------
try:
    from AlgorithmImports import (
        QCAlgorithm,
        Resolution,
        TradeBar,
        Symbol,
        Slice,
        RollingWindow,
        SimpleMovingAverage,
        StandardDeviation,
    )
except ImportError:
    # Allow the file to be imported outside of QC for unit testing
    QCAlgorithm = object  # type: ignore[assignment,misc]
    Resolution = None  # type: ignore[assignment]

from config import (
    CUSHING_CSV,
    FRED_CSV,
    FRONT_MONTH_TICKER,
    BACK_MONTH_TICKER,
    ENTRY_Z_SCORE,
    EXIT_Z_SCORE,
    LOOKBACK_DAYS,
    MAX_POSITION_SIZE,
    INVENTORY_LOOKBACK_WEEKS,
)
from data_manager import FundamentalDataManager


class OilCalendarSpreadAlgorithm(QCAlgorithm):  # type: ignore[misc]
    """
    Mean-reversion calendar spread strategy on WTI crude oil.

    Entry rules
    -----------
    * Compute z-score of (front_price - back_price) over LOOKBACK_DAYS.
    * If z-score >  ENTRY_Z_SCORE → spread is wide (contango extreme) → short
      front / long back.
    * If z-score < -ENTRY_Z_SCORE → spread is narrow (backwardation extreme)
      → long front / short back.
    * Inventory z-score is used as a secondary filter: only enter when it
      agrees with the spread signal direction.

    Exit rules
    ----------
    * Exit when |z-score| < EXIT_Z_SCORE (spread has mean-reverted).
    * Hard stop-loss at 3× ENTRY_Z_SCORE.
    """

    def initialize(self) -> None:
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(1_000_000)

        # ----------------------------------------------------------------
        # Subscribe to futures data
        # ----------------------------------------------------------------
        self._front = self.add_equity(FRONT_MONTH_TICKER, Resolution.DAILY).symbol
        self._back = self.add_equity(BACK_MONTH_TICKER, Resolution.DAILY).symbol

        # ----------------------------------------------------------------
        # Load fundamental data from CSV files (no live API calls)
        # ----------------------------------------------------------------
        self._data_manager = FundamentalDataManager(
            self,
            CUSHING_CSV,
            FRED_CSV,
            lookback_weeks=INVENTORY_LOOKBACK_WEEKS,
        )
        self._data_manager.load_all()

        # ----------------------------------------------------------------
        # Rolling indicators for z-score calculation
        # ----------------------------------------------------------------
        self._spread_window = RollingWindow[float](LOOKBACK_DAYS)
        self._spread_mean = SimpleMovingAverage(LOOKBACK_DAYS)
        self._spread_std = StandardDeviation(LOOKBACK_DAYS)

        # State
        self._in_position: bool = False
        self._position_direction: int = 0  # +1 long spread, -1 short spread

        self.log(
            f"Algorithm initialized. "
            f"Front: {FRONT_MONTH_TICKER}, Back: {BACK_MONTH_TICKER}, "
            f"Lookback: {LOOKBACK_DAYS} days."
        )

    def on_data(self, data: Slice) -> None:
        """Called on every new bar.  No external API calls are made here."""

        if not (data.contains_key(self._front) and data.contains_key(self._back)):
            return

        front_price: float = data[self._front].close
        back_price: float = data[self._back].close

        if front_price <= 0 or back_price <= 0:
            return

        spread: float = front_price - back_price

        # Feed spread into rolling statistics
        self._spread_mean.update(self.time, spread)
        self._spread_std.update(self.time, spread)

        if not (self._spread_mean.is_ready and self._spread_std.is_ready):
            return

        z_score = self._compute_z_score(spread)
        if z_score is None:
            return

        # Retrieve fundamental filter
        inv_z = self._data_manager.get_inventory_z_score(self.time.date())

        # ----------------------------------------------------------------
        # Trading logic
        # ----------------------------------------------------------------
        if self._in_position:
            self._check_exit(z_score)
        else:
            self._check_entry(z_score, inv_z)

    # ------------------------------------------------------------------
    # Trading helpers
    # ------------------------------------------------------------------

    def _compute_z_score(self, spread: float) -> Optional[float]:
        sigma = self._spread_std.current.value
        mu = self._spread_mean.current.value
        if sigma is None or sigma <= 0:
            return None
        return (spread - mu) / sigma

    def _check_entry(self, z_score: float, inv_z: Optional[float]) -> None:
        """Enter a position when z-score breaches the threshold."""
        direction = 0

        if z_score > ENTRY_Z_SCORE:
            # Spread unusually wide → expect mean reversion downward
            # Short front (sell) / Long back (buy)
            direction = -1
        elif z_score < -ENTRY_Z_SCORE:
            # Spread unusually narrow → expect mean reversion upward
            # Long front (buy) / Short back (sell)
            direction = 1

        if direction == 0:
            return

        # Inventory confirmation filter (optional – skip if data unavailable)
        if inv_z is not None:
            # High inventory → bearish for front month → supports short spread
            # Low inventory → bullish for front month → supports long spread
            if direction == -1 and inv_z < 0:
                return  # inventory doesn't support contango trade
            if direction == 1 and inv_z > 0:
                return  # inventory doesn't support backwardation trade

        size = MAX_POSITION_SIZE
        if direction == 1:
            self.set_holdings(self._front, size)
            self.set_holdings(self._back, -size)
        else:
            self.set_holdings(self._front, -size)
            self.set_holdings(self._back, size)

        self._in_position = True
        self._position_direction = direction
        self.log(
            f"ENTER {'LONG' if direction == 1 else 'SHORT'} spread "
            f"| z={z_score:.2f} | inv_z={inv_z}"
        )

    def _check_exit(self, z_score: float) -> None:
        """Exit when spread has sufficiently mean-reverted or stop is hit."""
        abs_z = abs(z_score)
        stop_z = ENTRY_Z_SCORE * 3

        should_exit = abs_z < EXIT_Z_SCORE or abs_z > stop_z
        if not should_exit:
            return

        self.liquidate(self._front)
        self.liquidate(self._back)
        self._in_position = False
        reason = "mean-reversion" if abs_z < EXIT_Z_SCORE else "stop-loss"
        self.log(f"EXIT ({reason}) | z={z_score:.2f}")
        self._position_direction = 0

    def on_end_of_algorithm(self) -> None:
        self.log("Algorithm finished.")
