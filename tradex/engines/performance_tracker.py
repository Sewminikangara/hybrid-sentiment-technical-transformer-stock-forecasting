"""
Signal Performance Tracker (Closed-Loop Feedback)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedSignal:
    """A signal with its tracked outcome."""
    signal_id: str
    symbol: str
    direction: str
    grade: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    entry_time: str
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    outcome: str = "OPEN"        # OPEN, TP1, TP2, TP3, SL, EXPIRED
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    hold_duration_min: int = 0
    conditions_passed: List[str] = field(default_factory=list)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_signals: int
    closed_signals: int
    open_signals: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_r_multiple: float
    total_r: float
    best_r: float
    worst_r: float
    avg_hold_minutes: float
    profit_factor: float
    expectancy_r: float
    streak_current: int           # Positive = win streak, negative = loss
    streak_max_win: int
    streak_max_loss: int
    performance_by_symbol: Dict[str, Dict]
    performance_by_direction: Dict[str, Dict]
    recent_accuracy_pct: float    # Last 20 signals


class SignalPerformanceTracker:
    """
    Monitors signal outcomes and maintains a performance database.
    Persists data to JSON files for analysis across sessions.
    """

    def __init__(self, data_dir: Optional[str] = None,
                 max_hold_bars: int = 100,
                 expiry_hours: int = 72):
        """
        Args:
            data_dir: Directory to store performance data.
            max_hold_bars: Maximum bars before marking signal expired.
            expiry_hours: Hours after which open signals expire.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".tradex" / "performance"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_hold_bars = max_hold_bars
        self.expiry_hours = expiry_hours
        self._signals: List[TrackedSignal] = []
        self._load()

    def _load(self):
        """Load existing performance data from disk."""
        data_file = self.data_dir / "signals.json"
        if data_file.exists():
            try:
                with open(data_file, "r") as f:
                    data = json.load(f)
                self._signals = [TrackedSignal(**s) for s in data]
                logger.info(f"Loaded {len(self._signals)} tracked signals.")
            except Exception as e:
                logger.warning(f"Failed to load performance data: {e}")
                self._signals = []

    def _save(self):
        """Persist performance data to disk."""
        data_file = self.data_dir / "signals.json"
        try:
            data = [asdict(s) for s in self._signals]
            with open(data_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save performance data: {e}")

    def register_signal(self, signal_id: str, symbol: str,
                        direction: str, grade: str,
                        entry_price: float, stop_loss: float,
                        tp1: float, tp2: float, tp3: float,
                        conditions: Optional[List[str]] = None):
        """Register a new signal for tracking."""
        ts = TrackedSignal(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            grade=grade,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            entry_time=datetime.now(timezone.utc).isoformat(),
            conditions_passed=conditions or [],
        )
        self._signals.append(ts)
        self._save()
        logger.info(f"Registered signal {signal_id} for {symbol}")

    def update_outcome(self, signal_id: str, exit_price: float,
                       outcome: str):
        """
        Update the outcome of a tracked signal.

        Args:
            signal_id: ID of the signal to update.
            exit_price: Price at which the position was closed.
            outcome: One of TP1, TP2, TP3, SL, EXPIRED.
        """
        for sig in self._signals:
            if sig.signal_id == signal_id and sig.outcome == "OPEN":
                sig.exit_price = exit_price
                sig.exit_time = datetime.now(timezone.utc).isoformat()
                sig.outcome = outcome

                # Calculate P&L
                if sig.direction == "LONG":
                    sig.pnl_pct = (exit_price - sig.entry_price) / sig.entry_price * 100
                else:
                    sig.pnl_pct = (sig.entry_price - exit_price) / sig.entry_price * 100

                # R-multiple
                risk = abs(sig.entry_price - sig.stop_loss)
                if risk > 0:
                    if sig.direction == "LONG":
                        sig.r_multiple = (exit_price - sig.entry_price) / risk
                    else:
                        sig.r_multiple = (sig.entry_price - exit_price) / risk

                # Hold duration
                try:
                    entry_dt = datetime.fromisoformat(sig.entry_time)
                    exit_dt = datetime.fromisoformat(sig.exit_time)
                    sig.hold_duration_min = int(
                        (exit_dt - entry_dt).total_seconds() / 60
                    )
                except (ValueError, TypeError):
                    pass

                self._save()
                logger.info(f"Signal {signal_id}: {outcome}, R={sig.r_multiple:.2f}")
                return

        logger.warning(f"Signal {signal_id} not found or already closed.")

    def check_and_update(self, symbol: str, current_high: float,
                         current_low: float, current_close: float):
        """
        Check all open signals for the given symbol against current
        price action and update outcomes automatically.

        Args:
            symbol: Asset symbol.
            current_high: Current bar high.
            current_low: Current bar low.
            current_close: Current bar close.
        """
        for sig in self._signals:
            if sig.symbol != symbol or sig.outcome != "OPEN":
                continue

            is_long = sig.direction == "LONG"

            # Check stop loss
            if is_long and current_low <= sig.stop_loss:
                self.update_outcome(sig.signal_id, sig.stop_loss, "SL")
                continue
            elif not is_long and current_high >= sig.stop_loss:
                self.update_outcome(sig.signal_id, sig.stop_loss, "SL")
                continue

            # Check take profits (highest first)
            if is_long:
                if current_high >= sig.take_profit_3:
                    self.update_outcome(sig.signal_id, sig.take_profit_3, "TP3")
                elif current_high >= sig.take_profit_2:
                    self.update_outcome(sig.signal_id, sig.take_profit_2, "TP2")
                elif current_high >= sig.take_profit_1:
                    self.update_outcome(sig.signal_id, sig.take_profit_1, "TP1")
            else:
                if current_low <= sig.take_profit_3:
                    self.update_outcome(sig.signal_id, sig.take_profit_3, "TP3")
                elif current_low <= sig.take_profit_2:
                    self.update_outcome(sig.signal_id, sig.take_profit_2, "TP2")
                elif current_low <= sig.take_profit_1:
                    self.update_outcome(sig.signal_id, sig.take_profit_1, "TP1")

            # Check expiry
            try:
                entry_dt = datetime.fromisoformat(sig.entry_time)
                if (datetime.now(timezone.utc) - entry_dt).total_seconds() > self.expiry_hours * 3600:
                    self.update_outcome(sig.signal_id, current_close, "EXPIRED")
            except (ValueError, TypeError):
                pass

    def get_stats(self, symbol: Optional[str] = None,
                  last_n: Optional[int] = None) -> PerformanceStats:
        """
        Calculate performance statistics.

        Args:
            symbol: Filter by symbol (None = all).
            last_n: Only consider the last N signals.

        Returns:
            PerformanceStats with comprehensive metrics.
        """
        signals = self._signals
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        if last_n:
            signals = signals[-last_n:]

        closed = [s for s in signals if s.outcome != "OPEN"]
        open_sigs = [s for s in signals if s.outcome == "OPEN"]

        wins = [s for s in closed if s.r_multiple > 0]
        losses = [s for s in closed if s.r_multiple <= 0]

        win_rate = len(wins) / len(closed) * 100 if closed else 0.0

        r_multiples = [s.r_multiple for s in closed]
        avg_r = np.mean(r_multiples) if r_multiples else 0.0
        total_r = sum(r_multiples)
        best_r = max(r_multiples) if r_multiples else 0.0
        worst_r = min(r_multiples) if r_multiples else 0.0

        hold_times = [s.hold_duration_min for s in closed if s.hold_duration_min > 0]
        avg_hold = np.mean(hold_times) if hold_times else 0.0

        # Profit factor
        gross_profit = sum(s.r_multiple for s in wins) if wins else 0.0
        gross_loss = abs(sum(s.r_multiple for s in losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Expectancy
        expectancy = avg_r

        # Streak
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0

        for s in closed:
            if s.r_multiple > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

        # Per-symbol breakdown
        by_symbol: Dict[str, Dict] = {}
        for s in closed:
            if s.symbol not in by_symbol:
                by_symbol[s.symbol] = {"wins": 0, "losses": 0, "total_r": 0.0}
            if s.r_multiple > 0:
                by_symbol[s.symbol]["wins"] += 1
            else:
                by_symbol[s.symbol]["losses"] += 1
            by_symbol[s.symbol]["total_r"] += s.r_multiple

        # Per-direction breakdown
        by_dir: Dict[str, Dict] = {}
        for s in closed:
            d = s.direction
            if d not in by_dir:
                by_dir[d] = {"wins": 0, "losses": 0, "total_r": 0.0}
            if s.r_multiple > 0:
                by_dir[d]["wins"] += 1
            else:
                by_dir[d]["losses"] += 1
            by_dir[d]["total_r"] += s.r_multiple

        # Recent accuracy (last 20)
        recent_closed = closed[-20:] if len(closed) >= 20 else closed
        recent_wins = sum(1 for s in recent_closed if s.r_multiple > 0)
        recent_acc = recent_wins / len(recent_closed) * 100 if recent_closed else 0.0

        return PerformanceStats(
            total_signals=len(signals),
            closed_signals=len(closed),
            open_signals=len(open_sigs),
            wins=len(wins),
            losses=len(losses),
            win_rate_pct=round(win_rate, 1),
            avg_r_multiple=round(avg_r, 3),
            total_r=round(total_r, 2),
            best_r=round(best_r, 2),
            worst_r=round(worst_r, 2),
            avg_hold_minutes=round(avg_hold, 1),
            profit_factor=round(profit_factor, 2),
            expectancy_r=round(expectancy, 3),
            streak_current=current_streak,
            streak_max_win=max_win_streak,
            streak_max_loss=max_loss_streak,
            performance_by_symbol=by_symbol,
            performance_by_direction=by_dir,
            recent_accuracy_pct=round(recent_acc, 1),
        )

    def format_stats(self, stats: PerformanceStats) -> str:
        """Format stats as readable text."""
        lines = [
            "Signal Performance Summary",
            "=" * 40,
            f"Total signals: {stats.total_signals} "
            f"({stats.closed_signals} closed, {stats.open_signals} open)",
            f"Win/Loss: {stats.wins}W / {stats.losses}L",
            f"Win rate: {stats.win_rate_pct:.1f}%",
            f"",
            f"Avg R-multiple: {stats.avg_r_multiple:+.3f}R",
            f"Total R: {stats.total_r:+.2f}R",
            f"Best: {stats.best_r:+.2f}R | Worst: {stats.worst_r:+.2f}R",
            f"Profit factor: {stats.profit_factor:.2f}",
            f"Expectancy: {stats.expectancy_r:+.3f}R",
            f"",
            f"Avg hold: {stats.avg_hold_minutes:.0f} min",
            f"Streak: {stats.streak_current:+d} "
            f"(max win: {stats.streak_max_win}, max loss: {stats.streak_max_loss})",
            f"Recent accuracy (20): {stats.recent_accuracy_pct:.1f}%",
        ]

        if stats.performance_by_symbol:
            lines.append("\nBy Symbol:")
            for sym, data in stats.performance_by_symbol.items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total * 100 if total > 0 else 0
                lines.append(
                    f"  {sym}: {data['wins']}W/{data['losses']}L "
                    f"({wr:.0f}%) R={data['total_r']:+.2f}"
                )

        return "\n".join(lines)


if __name__ == "__main__":
    import tempfile

    print("Signal Performance Tracker Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = SignalPerformanceTracker(data_dir=tmpdir)

        # Register signals
        test_signals = [
            ("s1", "BTCUSDT", "LONG", 45000, 44000, 46000, 47000, 48000),
            ("s2", "ETHUSDT", "SHORT", 3000, 3100, 2900, 2800, 2700),
            ("s3", "BTCUSDT", "LONG", 46000, 45500, 46500, 47000, 47500),
            ("s4", "EURUSD", "LONG", 1.0800, 1.0750, 1.0850, 1.0900, 1.0950),
            ("s5", "BTCUSDT", "SHORT", 47000, 47500, 46500, 46000, 45500),
        ]

        for sid, sym, d, ep, sl, tp1, tp2, tp3 in test_signals:
            tracker.register_signal(sid, sym, d, "A", ep, sl, tp1, tp2, tp3)

        # Simulate outcomes
        tracker.update_outcome("s1", 47000, "TP2")
        tracker.update_outcome("s2", 2900, "TP1")
        tracker.update_outcome("s3", 45500, "SL")
        tracker.update_outcome("s4", 1.0850, "TP1")

        # s5 stays open

        stats = tracker.get_stats()
        print(tracker.format_stats(stats))

    print("\nPerformance tracker test complete.")
