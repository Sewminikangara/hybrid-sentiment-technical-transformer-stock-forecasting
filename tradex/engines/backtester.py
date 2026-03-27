"""
Signal Backtesting Framework
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    direction: str        # LONG or SHORT
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    pnl: float
    pnl_pct: float
    r_multiple: float     # Profit / Risk
    outcome: str          # WIN, LOSS, or OPEN
    hold_bars: int
    signal_grade: str


@dataclass
class BacktestResult:
    """Aggregated backtesting results."""
    symbol: str
    period_start: datetime
    period_end: datetime
    total_bars: int
    total_signals: int
    total_trades: int
    trades: List[BacktestTrade]

    # Performance metrics
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_r_multiple: float
    avg_hold_bars: float
    expectancy: float

    # Equity curve
    equity_curve: List[float]
    drawdown_curve: List[float]

    # Comparison
    buy_hold_return_pct: float


class SignalBacktester:
    """
    Replays historical price data through the signal evaluation
    pipeline and measures how the generated A-grade signals would
    have performed.
    """

    def __init__(self, initial_capital: float = 10000.0,
                 risk_per_trade_pct: float = 2.0,
                 commission_pct: float = 0.1,
                 slippage_pct: float = 0.05):
        """
        Args:
            initial_capital: Starting portfolio value in USD.
            risk_per_trade_pct: Percentage of capital risked per trade.
            commission_pct: Round-trip commission as percentage.
            slippage_pct: Estimated slippage as percentage.
        """
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def _simulate_trade(self, entry_bar: int, signal_direction: str,
                        entry_price: float, sl: float,
                        tp1: float, tp2: float, tp3: float,
                        df: pd.DataFrame,
                        signal_grade: str) -> BacktestTrade:
        """
        Simulate a single trade from entry to exit.

        Uses a multi-target exit strategy:
            - TP1 hit: close 50% of position
            - TP2 hit: close 30%
            - TP3 hit: close remaining 20%
            - SL hit at any point: close entire remaining position
        """
        is_long = signal_direction == "LONG"
        remaining = 1.0
        realised_pnl = 0.0
        exit_price = entry_price
        exit_date = None
        hold_bars = 0

        for i in range(entry_bar + 1, len(df)):
            hold_bars += 1
            bar = df.iloc[i]
            high = bar["High"]
            low = bar["Low"]

            # Check stop loss first
            if is_long and low <= sl:
                exit_price = sl * (1 - self.slippage_pct / 100)
                realised_pnl += remaining * (exit_price - entry_price) / entry_price
                exit_date = bar.name if hasattr(bar, 'name') else None
                remaining = 0.0
                break
            elif not is_long and high >= sl:
                exit_price = sl * (1 + self.slippage_pct / 100)
                realised_pnl += remaining * (entry_price - exit_price) / entry_price
                exit_date = bar.name if hasattr(bar, 'name') else None
                remaining = 0.0
                break

            # Check take profits
            if is_long:
                if remaining > 0.5 and high >= tp1:
                    realised_pnl += 0.5 * (tp1 - entry_price) / entry_price
                    remaining -= 0.5
                if remaining > 0.2 and high >= tp2:
                    realised_pnl += 0.3 * (tp2 - entry_price) / entry_price
                    remaining -= 0.3
                if remaining > 0 and high >= tp3:
                    realised_pnl += remaining * (tp3 - entry_price) / entry_price
                    exit_price = tp3
                    exit_date = bar.name if hasattr(bar, 'name') else None
                    remaining = 0.0
                    break
            else:
                if remaining > 0.5 and low <= tp1:
                    realised_pnl += 0.5 * (entry_price - tp1) / entry_price
                    remaining -= 0.5
                if remaining > 0.2 and low <= tp2:
                    realised_pnl += 0.3 * (entry_price - tp2) / entry_price
                    remaining -= 0.3
                if remaining > 0 and low <= tp3:
                    realised_pnl += remaining * (entry_price - tp3) / entry_price
                    exit_price = tp3
                    exit_date = bar.name if hasattr(bar, 'name') else None
                    remaining = 0.0
                    break

        # If still open after all bars
        if remaining > 0:
            exit_price = df.iloc[-1]["Close"]
            if is_long:
                realised_pnl += remaining * (exit_price - entry_price) / entry_price
            else:
                realised_pnl += remaining * (entry_price - exit_price) / entry_price

        # Apply commission
        realised_pnl -= self.commission_pct / 100

        # Calculate R-multiple
        risk = abs(entry_price - sl) / entry_price
        r_multiple = realised_pnl / risk if risk > 0 else 0.0

        outcome = "WIN" if realised_pnl > 0 else "LOSS"
        if remaining > 0:
            outcome = "OPEN"

        entry_dt = df.index[entry_bar] if isinstance(df.index, pd.DatetimeIndex) else None

        return BacktestTrade(
            entry_date=entry_dt,
            exit_date=exit_date,
            symbol="",
            direction=signal_direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            pnl=realised_pnl * self.initial_capital * (self.risk_per_trade_pct / 100),
            pnl_pct=realised_pnl * 100,
            r_multiple=r_multiple,
            outcome=outcome,
            hold_bars=hold_bars,
            signal_grade=signal_grade,
        )

    def backtest(self, df: pd.DataFrame, symbol: str,
                 signal_interval: int = 20) -> BacktestResult:
        """
        Run full backtest on historical data.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            symbol: Asset symbol.
            signal_interval: Minimum bars between signal evaluations.

        Returns:
            BacktestResult with full performance metrics.
        """
        from tradex.engines.signal_engine import SignalEngine
        from tradex.engines.news_risk_filter import NewsRiskFilter
        from tradex.config import DEFAULT_CONFIG

        engine = SignalEngine(config=DEFAULT_CONFIG)
        nrf = NewsRiskFilter(config=DEFAULT_CONFIG.news)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)

        trades = []
        equity = self.initial_capital
        equity_curve = [equity]

        i = max(200, 0)  # Start after enough lookback
        while i < len(df) - signal_interval:
            # Extract lookback window for signal evaluation
            window = df.iloc[max(0, i - 200):i + 1].copy()

            if len(window) < 50:
                i += signal_interval
                continue

            try:
                signal = engine.evaluate(
                    symbol=symbol,
                    trend_df=window,
                    entry_df=window,
                    news_filter=nrf,
                )
            except Exception:
                i += signal_interval
                continue

            if signal and signal.grade.value == "A":
                entry_price = df.iloc[i]["Close"]
                trade = self._simulate_trade(
                    entry_bar=i,
                    signal_direction=signal.direction.value,
                    entry_price=entry_price,
                    sl=signal.stop_loss,
                    tp1=signal.take_profit_1,
                    tp2=signal.take_profit_2,
                    tp3=signal.take_profit_3,
                    df=df,
                    signal_grade=signal.grade.value,
                )
                trade.symbol = symbol
                trades.append(trade)

                # Update equity
                risk_amount = equity * (self.risk_per_trade_pct / 100)
                equity += trade.r_multiple * risk_amount
                equity_curve.append(equity)

                # Skip bars where the trade is active
                i += max(trade.hold_bars, signal_interval)
            else:
                i += signal_interval
                equity_curve.append(equity)

        # Calculate metrics
        result = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            df=df,
            symbol=symbol,
        )
        return result

    def _calculate_metrics(self, trades: List[BacktestTrade],
                           equity_curve: List[float],
                           df: pd.DataFrame,
                           symbol: str) -> BacktestResult:
        """Calculate all performance metrics from trade list."""
        equity_arr = np.array(equity_curve)

        # Returns
        total_return = (equity_arr[-1] - self.initial_capital) / self.initial_capital
        days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 1
        years = max(days / 365.25, 0.01)
        ann_return = (1 + total_return) ** (1 / years) - 1

        # Daily returns for Sharpe/Sortino
        if len(equity_arr) > 1:
            returns = np.diff(equity_arr) / equity_arr[:-1]
        else:
            returns = np.array([0.0])

        # Sharpe ratio (annualised, risk-free = 2%)
        vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 1e-6
        sharpe = (ann_return - 0.02) / vol if vol > 0 else 0.0

        # Sortino ratio (downside deviation only)
        neg_returns = returns[returns < 0]
        downside_vol = np.std(neg_returns) * np.sqrt(252) if len(neg_returns) > 0 else 1e-6
        sortino = (ann_return - 0.02) / downside_vol if downside_vol > 0 else 0.0

        # Maximum drawdown
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Win rate
        closed = [t for t in trades if t.outcome != "OPEN"]
        wins = [t for t in closed if t.outcome == "WIN"]
        win_rate = len(wins) / len(closed) * 100 if closed else 0.0

        # Profit factor
        gross_profit = sum(t.pnl for t in closed if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average R-multiple
        r_multiples = [t.r_multiple for t in closed]
        avg_r = np.mean(r_multiples) if r_multiples else 0.0

        # Average hold time
        hold_bars = [t.hold_bars for t in trades]
        avg_hold = np.mean(hold_bars) if hold_bars else 0.0

        # Expectancy
        expectancy = avg_r * (win_rate / 100) if win_rate > 0 else 0.0

        # Buy and hold
        bh_return = (df.iloc[-1]["Close"] - df.iloc[0]["Close"]) / df.iloc[0]["Close"]

        start = df.index[0] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        end = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()

        return BacktestResult(
            symbol=symbol,
            period_start=start,
            period_end=end,
            total_bars=len(df),
            total_signals=len(trades),
            total_trades=len(closed),
            trades=trades,
            total_return_pct=total_return * 100,
            annualised_return_pct=ann_return * 100,
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=max_dd * 100,
            win_rate_pct=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            avg_r_multiple=round(avg_r, 2),
            avg_hold_bars=round(avg_hold, 1),
            expectancy=round(expectancy, 3),
            equity_curve=equity_curve,
            drawdown_curve=drawdown.tolist(),
            buy_hold_return_pct=bh_return * 100,
        )

    def summary(self, result: BacktestResult) -> str:
        """Format results as a human-readable summary."""
        lines = [
            f"Backtest Results: {result.symbol}",
            f"{'=' * 50}",
            f"Period: {result.period_start} to {result.period_end}",
            f"Total bars: {result.total_bars}",
            f"",
            f"Signals generated:   {result.total_signals}",
            f"Trades closed:       {result.total_trades}",
            f"",
            f"Total return:        {result.total_return_pct:+.2f}%",
            f"Annualised return:   {result.annualised_return_pct:+.2f}%",
            f"Buy & hold return:   {result.buy_hold_return_pct:+.2f}%",
            f"",
            f"Sharpe ratio:        {result.sharpe_ratio:.3f}",
            f"Sortino ratio:       {result.sortino_ratio:.3f}",
            f"Max drawdown:        {result.max_drawdown_pct:.2f}%",
            f"",
            f"Win rate:            {result.win_rate_pct:.1f}%",
            f"Profit factor:       {result.profit_factor:.2f}",
            f"Avg R-multiple:      {result.avg_r_multiple:.2f}R",
            f"Avg hold bars:       {result.avg_hold_bars:.1f}",
            f"Expectancy:          {result.expectancy:.3f}R",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    logger.info("Signal Backtesting Framework Test")
    logger.info("=")

    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.maximum(close, 10)
    df = pd.DataFrame({
        "Open": close - np.random.rand(n) * 0.5,
        "High": close + np.random.rand(n) * 1.0,
        "Low": close - np.random.rand(n) * 1.0,
        "Close": close,
        "Volume": np.random.randint(1000, 50000, n),
    }, index=dates)

    backtester = SignalBacktester(
        initial_capital=10000,
        risk_per_trade_pct=2.0,
    )

    result = backtester.backtest(df, symbol="TEST")
    logger.info(backtester.summary(result))

    logger.info("\nBacktesting framework test complete.")
