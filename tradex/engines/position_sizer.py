"""
Position Sizing Model (Kelly Criterion)

"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Output of position sizing calculation."""
    symbol: str
    full_kelly_fraction: float      # Full Kelly fraction (0.0 to 1.0)
    fractional_kelly: float         # Adjusted fraction (full * kelly_fraction)
    position_size_usd: float        # Dollar amount to risk
    position_size_pct: float        # Percentage of capital
    shares_or_units: float          # Number of shares/units
    stop_loss_distance: float       # Distance to stop in price units
    risk_per_unit: float            # Risk per share/unit
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    edge: float                     # win_rate * avg_win - (1-win_rate) * avg_loss
    method: str                     # KELLY, FIXED_FRACTIONAL, or CAPPED


class PositionSizer:
    """
    Calculates position size using the Kelly Criterion with
    configurable fractional multiplier and maximum risk cap.

    Kelly formula:
        f* = (p * b - q) / b

    Where:
        p = probability of winning (win rate)
        q = probability of losing (1 - p)
        b = ratio of average win to average loss (reward/risk)
        f* = fraction of capital to risk
    """

    def __init__(self, kelly_fraction: float = 0.25,
                 max_risk_pct: float = 5.0,
                 min_risk_pct: float = 0.5,
                 default_risk_pct: float = 2.0):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly).
            max_risk_pct: Maximum position size as % of capital (hard cap).
            min_risk_pct: Minimum position size as % of capital.
            default_risk_pct: Default when insufficient trade history.
        """
        self.kelly_fraction = kelly_fraction
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.default_risk_pct = default_risk_pct

    def calculate_kelly(self, win_rate: float,
                        avg_win: float,
                        avg_loss: float) -> float:
        """
        Calculate full Kelly fraction.

        Args:
            win_rate: Historical win rate (0.0 to 1.0).
            avg_win: Average winning trade return (positive).
            avg_loss: Average losing trade return (positive, absolute value).

        Returns:
            Full Kelly fraction (can be negative if no edge).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0

        p = win_rate
        q = 1.0 - p
        b = avg_win / avg_loss  # Reward-to-risk ratio

        kelly = (p * b - q) / b

        return kelly

    def calculate_from_trades(self, trades: List[Dict],
                               capital: float,
                               entry_price: float,
                               stop_loss: float,
                               symbol: str = "UNKNOWN") -> PositionSizeResult:
        """
        Calculate position size from historical trade records.

        Args:
            trades: List of trade dicts with keys: pnl, pnl_pct, outcome.
            capital: Current portfolio value.
            entry_price: Planned entry price.
            stop_loss: Planned stop loss price.
            symbol: Asset symbol.

        Returns:
            PositionSizeResult with recommended position size.
        """
        sl_distance = abs(entry_price - stop_loss)

        if not trades or len(trades) < 10:
            # Insufficient history -> fixed fractional
            risk_usd = capital * (self.default_risk_pct / 100)
            units = risk_usd / sl_distance if sl_distance > 0 else 0
            return PositionSizeResult(
                symbol=symbol,
                full_kelly_fraction=0.0,
                fractional_kelly=0.0,
                position_size_usd=risk_usd,
                position_size_pct=self.default_risk_pct,
                shares_or_units=units,
                stop_loss_distance=sl_distance,
                risk_per_unit=sl_distance,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                expectancy=0.0,
                edge=0.0,
                method="FIXED_FRACTIONAL",
            )

        # Calculate win/loss statistics
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t["pnl_pct"] for t in losses])) if losses else 1.0

        # Kelly
        full_kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
        fractional = full_kelly * self.kelly_fraction

        # Edge and expectancy
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        expectancy = edge / avg_loss if avg_loss > 0 else 0.0

        # Apply risk caps
        if fractional <= 0:
            # No edge or negative edge -> minimum size
            risk_pct = self.min_risk_pct
            method = "CAPPED"
        else:
            risk_pct = fractional * 100  # Convert to percentage
            if risk_pct > self.max_risk_pct:
                risk_pct = self.max_risk_pct
                method = "CAPPED"
            elif risk_pct < self.min_risk_pct:
                risk_pct = self.min_risk_pct
                method = "CAPPED"
            else:
                method = "KELLY"

        risk_usd = capital * (risk_pct / 100)
        units = risk_usd / sl_distance if sl_distance > 0 else 0

        return PositionSizeResult(
            symbol=symbol,
            full_kelly_fraction=round(full_kelly, 4),
            fractional_kelly=round(fractional, 4),
            position_size_usd=round(risk_usd, 2),
            position_size_pct=round(risk_pct, 2),
            shares_or_units=round(units, 4),
            stop_loss_distance=round(sl_distance, 6),
            risk_per_unit=round(sl_distance, 6),
            win_rate=round(win_rate, 3),
            avg_win=round(avg_win, 3),
            avg_loss=round(avg_loss, 3),
            expectancy=round(expectancy, 3),
            edge=round(edge, 4),
            method=method,
        )

    def calculate_simple(self, capital: float,
                         entry_price: float,
                         stop_loss: float,
                         win_rate: float = 0.55,
                         reward_risk_ratio: float = 2.0,
                         symbol: str = "UNKNOWN") -> PositionSizeResult:
        """
        Simplified calculation with known win rate and R:R.

        Args:
            capital: Current portfolio value.
            entry_price: Planned entry price.
            stop_loss: Planned stop loss.
            win_rate: Expected win rate.
            reward_risk_ratio: Expected reward/risk ratio.
            symbol: Asset symbol.
        """
        sl_distance = abs(entry_price - stop_loss)
        avg_win = reward_risk_ratio
        avg_loss = 1.0

        full_kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
        fractional = full_kelly * self.kelly_fraction

        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        expectancy = edge / avg_loss if avg_loss > 0 else 0.0

        risk_pct = max(self.min_risk_pct,
                       min(self.max_risk_pct, fractional * 100))
        risk_usd = capital * (risk_pct / 100)
        units = risk_usd / sl_distance if sl_distance > 0 else 0

        method = "KELLY" if 0 < fractional * 100 <= self.max_risk_pct else "CAPPED"

        return PositionSizeResult(
            symbol=symbol,
            full_kelly_fraction=round(full_kelly, 4),
            fractional_kelly=round(fractional, 4),
            position_size_usd=round(risk_usd, 2),
            position_size_pct=round(risk_pct, 2),
            shares_or_units=round(units, 4),
            stop_loss_distance=round(sl_distance, 6),
            risk_per_unit=round(sl_distance, 6),
            win_rate=round(win_rate, 3),
            avg_win=round(avg_win, 3),
            avg_loss=round(avg_loss, 3),
            expectancy=round(expectancy, 3),
            edge=round(edge, 4),
            method=method,
        )

    def format_summary(self, result: PositionSizeResult) -> str:
        """Format result as human-readable text."""
        lines = [
            f"Position Size: {result.symbol}",
            f"  Method: {result.method}",
            f"  Full Kelly: {result.full_kelly_fraction:.4f}",
            f"  Fractional Kelly ({self.kelly_fraction:.0%}): "
            f"{result.fractional_kelly:.4f}",
            f"",
            f"  Risk: ${result.position_size_usd:.2f} "
            f"({result.position_size_pct:.2f}% of capital)",
            f"  Units: {result.shares_or_units:.4f}",
            f"  SL distance: {result.stop_loss_distance:.4f}",
            f"",
            f"  Win rate: {result.win_rate:.1%}",
            f"  Avg win: {result.avg_win:.3f}",
            f"  Avg loss: {result.avg_loss:.3f}",
            f"  Edge: {result.edge:.4f}",
            f"  Expectancy: {result.expectancy:.3f}R",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Position Sizing (Kelly Criterion) Test")
    logger.info("=")

    sizer = PositionSizer(kelly_fraction=0.25, max_risk_pct=5.0)

    # Scenario 1: Known win rate
    result1 = sizer.calculate_simple(
        capital=10000,
        entry_price=150.00,
        stop_loss=147.00,
        win_rate=0.55,
        reward_risk_ratio=2.0,
        symbol="AAPL",
    )
    logger.info("\nScenario 1: Known win/loss ratio")
    logger.info(sizer.format_summary(result1))

    # Scenario 2: From trade history
    np.random.seed(42)
    trades = []
    for _ in range(50):
        win = np.random.random() < 0.55
        pnl = np.random.uniform(50, 200) if win else -np.random.uniform(30, 100)
        pnl_pct = pnl / 10000 * 100
        trades.append({
            "pnl": pnl,
            "pnl_pct": abs(pnl_pct),
            "outcome": "WIN" if win else "LOSS",
        })

    result2 = sizer.calculate_from_trades(
        trades=trades,
        capital=10000,
        entry_price=45000,
        stop_loss=44000,
        symbol="BTCUSDT",
    )
    logger.info("\nScenario 2: From 50 historic trades")
    logger.info(sizer.format_summary(result2))

    # Scenario 3: No edge (poor win rate)
    result3 = sizer.calculate_simple(
        capital=10000,
        entry_price=1.0800,
        stop_loss=1.0750,
        win_rate=0.35,
        reward_risk_ratio=1.5,
        symbol="EURUSD",
    )
    logger.info("\nScenario 3: No edge (35% win rate)")
    logger.info(sizer.format_summary(result3))

    logger.info("\nPosition sizing test complete.")
