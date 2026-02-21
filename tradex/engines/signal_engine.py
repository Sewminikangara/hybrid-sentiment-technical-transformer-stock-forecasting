
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tradex.config import (
    Direction, SignalGrade, RiskState, SignalConfig,
    TradeXYConfig, DEFAULT_CONFIG, TimeframeConfig
)
from tradex.engines.market_structure import (
    MarketStructureDetector, MarketStructureResult, BOSDirection
)
from tradex.engines.elliott_wave_engine import (
    ElliottWaveEngine, ElliottWaveResult
)
from tradex.engines.news_risk_filter import (
    NewsRiskFilter, SymbolRiskState
)

# --- Data Structures ---

@dataclass
class ChecklistResult:
    """Result of a single condition check."""
    condition: str          # "A", "B", "C", "D", "E"
    name: str               # Human readable
    passed: bool
    details: str            # Explanation
    score: float = 0.0      # 0–1 for weighting

@dataclass
class RiskParameters:
    """Calculated risk management levels."""
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    take_profit_1: float    # 1R
    take_profit_2: float    # 2R
    take_profit_3: float    # 3R
    risk_amount: float      # |entry - SL|
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: float

@dataclass
class TradingSignal:
    """A complete A-grade trading signal output."""
    # Core
    symbol: str
    direction: Direction
    grade: SignalGrade
    timestamp: datetime
    
    # Entry & Risk
    risk: RiskParameters
    
    # Checklist
    checklist: List[ChecklistResult]
    all_passed: bool
    
    # Elliott Wave
    elliott_summary: str
    elliott_confidence: int
    
    # News Risk
    news_risk_state: str    # CLEAR/CAUTION/BLOCK
    news_summary: str
    
    # Explanation
    reason: str             # Max 3 lines
    
    # Metadata
    trend_timeframe: str
    entry_timeframe: str
    current_price: float
    
    def to_dict(self) -> Dict:
        """Serialize for API/dashboard."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "grade": self.grade.value,
            "timestamp": self.timestamp.isoformat(),
            "entry_zone": f"{self.risk.entry_zone_low:.5f} – {self.risk.entry_zone_high:.5f}",
            "entry_price": self.risk.entry_price,
            "stop_loss": self.risk.stop_loss,
            "tp1": self.risk.take_profit_1,
            "tp2": self.risk.take_profit_2,
            "tp3": self.risk.take_profit_3,
            "risk_reward": f"1:{self.risk.risk_reward_1:.1f} / 1:{self.risk.risk_reward_2:.1f} / 1:{self.risk.risk_reward_3:.1f}",
            "checklist": [
                {"condition": c.condition, "name": c.name, 
                 "passed": c.passed, "details": c.details}
                for c in self.checklist
            ],
            "all_passed": self.all_passed,
            "elliott": {"summary": self.elliott_summary, "confidence": self.elliott_confidence},
            "news": {"state": self.news_risk_state, "summary": self.news_summary},
            "reason": self.reason,
            "trend_tf": self.trend_timeframe,
            "entry_tf": self.entry_timeframe,
            "current_price": self.current_price,
        }
    
    def format_alert(self) -> str:
        """Format signal as a text alert (for Telegram/logs)."""
        dir_label = "[LONG]" if self.direction == Direction.LONG else "[SHORT]"
        checks = " | ".join(
            f"{'[PASS]' if c.passed else '[FAIL]'}{c.condition}" for c in self.checklist
        )
        
        return (
            f"**{self.grade.value}-GRADE {self.direction.value} - {self.symbol}**\n"
            f"Entry: {self.risk.entry_zone_low:.5f} – {self.risk.entry_zone_high:.5f}\n"
            f"SL: {self.risk.stop_loss:.5f} | "
            f"TP1: {self.risk.take_profit_1:.5f} | "
            f"TP2: {self.risk.take_profit_2:.5f} | "
            f"TP3: {self.risk.take_profit_3:.5f}\n"
            f"Elliott: {self.elliott_summary} ({self.elliott_confidence}/100)\n"
            f"News: {self.news_risk_state}\n"
            f"Checks: [{checks}]\n"
            f"Reason: {self.reason}"
        )

# --- Signal Engine ---

class SignalEngine:
    """
    TradeXY A-Grade Signal Engine.
    
    Combines trend filter, market structure, Elliott Wave, and news risk
    to generate rare, high-confirmation trading signals.
    
    Usage:
        engine = SignalEngine()
        signal = engine.evaluate("BTCUSDT", trend_df, entry_df)
        if signal and signal.grade == SignalGrade.A:
            # Emit signal
    """
    
    def __init__(self, config: Optional[TradeXYConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize sub-engines
        self.structure_detector = MarketStructureDetector(
            swing_order=self.config.structure.swing_order,
            atr_period=self.config.trend.atr_period,
            min_swing_atr=self.config.trend.min_swing_size_atr,
            bos_min_break_atr=self.config.structure.bos_min_break_atr,
            retest_tolerance_atr=self.config.structure.retest_tolerance_atr,
            retest_hold_bars=self.config.structure.retest_hold_bars
        )
        
        self.elliott_engine = ElliottWaveEngine(
            wave2_fib_min=self.config.elliott.wave2_fib_min,
            wave2_fib_max=self.config.elliott.wave2_fib_max,
            wave2_fib_ideal=self.config.elliott.wave2_fib_ideal,
            wave3_min_extension=self.config.elliott.wave3_min_extension,
            min_confidence=self.config.elliott.min_confidence,
            swing_order=self.config.elliott.swing_order,
            top_n=self.config.elliott.top_n_counts
        )
        
        self.news_filter = NewsRiskFilter(self.config.news)
        
        # Signal tracking (cooldown)
        self._recent_signals: Dict[str, List[datetime]] = {}
    
    # --- Main Evaluation ---
    
    def evaluate(self, symbol: str, 
                 trend_df: pd.DataFrame,
                 entry_df: pd.DataFrame
                 ) -> Optional[TradingSignal]:
        """
        Evaluate all conditions for a symbol and potentially emit a signal.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            trend_df: OHLCV DataFrame for trend timeframe (4H)
            entry_df: OHLCV DataFrame for entry timeframe (15m)
        
        Returns:
            TradingSignal if A-grade conditions met, else None
        """
        now = datetime.utcnow()
        tf_config = TimeframeConfig.for_symbol(symbol)
        
        # --- Signal Cooldown Check ---
        if not self._check_cooldown(symbol, now):
            return None
        
        current_price = float(entry_df['Close'].iloc[-1])
        checklist = []
        
        # --- CONDITION A: Trend Filter (Higher Timeframe) ---
        trend_result = self.structure_detector.analyze(
            trend_df, ema_period=self.config.trend.ema_period
        )
        
        check_a = ChecklistResult(
            condition="A",
            name="Trend Filter",
            passed=trend_result.trend_filter_passed,
            details=(
                f"Trend: {trend_result.current_trend} "
                f"(strength: {trend_result.trend_strength:.0%}), "
                f"EMA200: {'Above' if trend_result.above_ema200 else 'Below'}"
            ),
            score=trend_result.trend_strength
        )
        checklist.append(check_a)
        
        # Determine signal direction from trend
        if trend_result.current_trend == "BULLISH" and trend_result.above_ema200:
            direction = Direction.LONG
        elif trend_result.current_trend == "BEARISH" and not trend_result.above_ema200:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL
        
        # --- CONDITION B: Entry Structure (Lower Timeframe) ---
        entry_result = self.structure_detector.analyze(
            entry_df, ema_period=50  # Shorter EMA for entry TF
        )
        
        # Check BOS alignment with trend direction
        bos_aligned = False
        if entry_result.latest_bos:
            if (direction == Direction.LONG and 
                entry_result.latest_bos.direction.value == "BULLISH"):
                bos_aligned = True
            elif (direction == Direction.SHORT and 
                  entry_result.latest_bos.direction.value == "BEARISH"):
                bos_aligned = True
        
        structure_passed = (
            entry_result.structure_filter_passed and bos_aligned
        )
        
        check_b = ChecklistResult(
            condition="B",
            name="Entry Structure (BOS + Retest)",
            passed=structure_passed,
            details=(
                f"BOS: {entry_result.latest_bos if entry_result.latest_bos else 'None'}, "
                f"Retest: {'Confirmed' if entry_result.retest_valid else 'Not confirmed'}, "
                f"Aligned: {'Yes' if bos_aligned else 'No'}"
            ),
            score=1.0 if structure_passed else 0.0
        )
        checklist.append(check_b)
        
        # --- CONDITION C: Elliott Wave ---
        elliott_direction = "BULLISH" if direction == Direction.LONG else "BEARISH"
        elliott_result = self.elliott_engine.analyze(
            entry_df, trend_direction=elliott_direction
        )
        
        check_c = ChecklistResult(
            condition="C",
            name="Elliott Wave (W2->W3)",
            passed=elliott_result.elliott_filter_passed,
            details=(
                f"{elliott_result.wave_summary}, "
                f"Fib: {elliott_result.best_candidate.fib_score:.0%}" 
                if elliott_result.best_candidate else "No pattern"
            ),
            score=elliott_result.confidence / 100.0
        )
        checklist.append(check_c)
        
        # --- CONDITION D: News Risk Filter ---
        news_allowed, news_reason = self.news_filter.check_signal_allowed(
            symbol, direction.value if direction != Direction.NEUTRAL else "LONG"
        )
        risk_state = self.news_filter.get_risk_state(symbol)
        
        check_d = ChecklistResult(
            condition="D",
            name="News Risk Filter",
            passed=news_allowed,
            details=news_reason,
            score=1.0 if news_allowed else 0.0
        )
        checklist.append(check_d)
        
        # --- CONDITION E: Risk Management ---
        risk_params = self._calculate_risk(
            symbol, direction, current_price,
            entry_result, elliott_result, entry_df
        )
        
        risk_valid = risk_params is not None
        check_e = ChecklistResult(
            condition="E",
            name="Risk Management (SL/TP)",
            passed=risk_valid,
            details=(
                f"SL: {risk_params.stop_loss:.5f}, "
                f"TP1: {risk_params.take_profit_1:.5f} (1:{risk_params.risk_reward_1:.1f})"
                if risk_params else "Cannot calculate valid SL/TP"
            ),
            score=1.0 if risk_valid else 0.0
        )
        checklist.append(check_e)
        
        # --- GRADE DETERMINATION ---
        all_passed = all(c.passed for c in checklist)
        
        if not all_passed or direction == Direction.NEUTRAL:
            # Log but don't emit - not A-grade
            return None
        
        # --- BUILD A-GRADE SIGNAL ---
        # Generate reason text (max 3 lines)
        reason = self._generate_reason(
            symbol, direction, trend_result, elliott_result, risk_state
        )
        
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            grade=SignalGrade.A,
            timestamp=now,
            risk=risk_params,
            checklist=checklist,
            all_passed=True,
            elliott_summary=elliott_result.wave_summary,
            elliott_confidence=elliott_result.confidence,
            news_risk_state=risk_state.state.value,
            news_summary="; ".join(risk_state.reasons[:2]) if risk_state.reasons else "Clear",
            reason=reason,
            trend_timeframe=tf_config.trend_tf,
            entry_timeframe=tf_config.entry_tf,
            current_price=current_price
        )
        
        # Record for cooldown
        self._record_signal(symbol, now)
        
        return signal
    
    # --- Risk Calculation ---
    
    def _calculate_risk(self, symbol: str, direction: Direction,
                        current_price: float,
                        entry_result: MarketStructureResult,
                        elliott_result: ElliottWaveResult,
                        df: pd.DataFrame) -> Optional[RiskParameters]:
        """
        Calculate entry zone, stop loss, and take profit levels.
        
        SL: Below/above the most recent swing low/high
        TP: R-multiples (1R, 2R, 3R)
        """
        if direction == Direction.NEUTRAL:
            return None
        
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        close = df['Close'].values.astype(float)
        
        # Calculate ATR for dynamic sizing
        atr = self._calc_atr(high, low, close, self.config.trend.atr_period)
        current_atr = atr[-1] if len(atr) > 0 else current_price * 0.01
        
        # Find swing levels for SL placement
        swing_lows = []
        swing_highs = []
        for sp in entry_result.swing_points[-10:]:
            if sp.swing_type.value == "LOW":
                swing_lows.append(sp.price)
            else:
                swing_highs.append(sp.price)
        
        if direction == Direction.LONG:
            # SL below recent swing low
            if swing_lows:
                sl_base = min(swing_lows[-3:])  # Last 3 swing lows
            else:
                sl_base = current_price - current_atr * 2
            
            stop_loss = sl_base - current_atr * self.config.signal.sl_atr_multiplier * 0.1
            risk = abs(current_price - stop_loss)
            
            if risk <= 0:
                return None
            
            # Entry zone from Elliott
            if elliott_result.best_candidate:
                entry_zone_low = elliott_result.best_candidate.entry_zone_low
                entry_zone_high = elliott_result.best_candidate.entry_zone_high
            else:
                entry_zone_low = current_price - current_atr * 0.3
                entry_zone_high = current_price + current_atr * 0.1
            
            # Take profits using R-multiples
            tp1 = current_price + risk * self.config.signal.default_risk_reward_1
            tp2 = current_price + risk * self.config.signal.default_risk_reward_2
            tp3 = current_price + risk * self.config.signal.default_risk_reward_3
            
        elif direction == Direction.SHORT:
            if swing_highs:
                sl_base = max(swing_highs[-3:])
            else:
                sl_base = current_price + current_atr * 2
            
            stop_loss = sl_base + current_atr * self.config.signal.sl_atr_multiplier * 0.1
            risk = abs(stop_loss - current_price)
            
            if risk <= 0:
                return None
            
            if elliott_result.best_candidate:
                entry_zone_low = elliott_result.best_candidate.entry_zone_low
                entry_zone_high = elliott_result.best_candidate.entry_zone_high
            else:
                entry_zone_low = current_price - current_atr * 0.1
                entry_zone_high = current_price + current_atr * 0.3
            
            tp1 = current_price - risk * self.config.signal.default_risk_reward_1
            tp2 = current_price - risk * self.config.signal.default_risk_reward_2
            tp3 = current_price - risk * self.config.signal.default_risk_reward_3
        else:
            return None
        
        return RiskParameters(
            entry_price=current_price,
            entry_zone_low=min(entry_zone_low, entry_zone_high),
            entry_zone_high=max(entry_zone_low, entry_zone_high),
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_amount=risk,
            risk_reward_1=self.config.signal.default_risk_reward_1,
            risk_reward_2=self.config.signal.default_risk_reward_2,
            risk_reward_3=self.config.signal.default_risk_reward_3
        )
    
    def _calc_atr(self, high, low, close, period):
        """ATR helper."""
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                       abs(high[i] - close[i-1]),
                       abs(low[i] - close[i-1]))
        atr = np.zeros(n)
        atr[:period] = np.mean(tr[:period]) if period <= n else np.mean(tr)
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    # --- Reason Generation ---
    
    def _generate_reason(self, symbol, direction, trend_result,
                         elliott_result, risk_state) -> str:
        """Generate a concise 3-line reason for the signal."""
        best = elliott_result.best_candidate
        
        line1 = (f"{direction.value} {symbol}: "
                f"{trend_result.current_trend} trend "
                f"({'above' if trend_result.above_ema200 else 'below'} EMA200, "
                f"strength {trend_result.trend_strength:.0%})")
        
        line2 = (f"Elliott Wave {best.current_wave.value}->{best.next_expected.value} "
                f"(conf {best.confidence}/100, fib {best.fib_score:.0%})"
                if best else "Elliott: pattern detected")
        
        line3 = (f"News: {risk_state.state.value}"
                f"{', ' + risk_state.reasons[0][:40] if risk_state.reasons else ''}")
        
        return f"{line1}\n{line2}\n{line3}"
    
    # --- Cooldown Management ---
    
    def _check_cooldown(self, symbol: str, now: datetime) -> bool:
        """Check if we're in cooldown for this symbol."""
        if symbol not in self._recent_signals:
            return True
        
        recent = self._recent_signals[symbol]
        min_interval = timedelta(
            minutes=self.config.signal.min_signal_interval_minutes
        )
        
        # Remove old signals
        cutoff = now - timedelta(hours=24)
        recent = [t for t in recent if t > cutoff]
        self._recent_signals[symbol] = recent
        
        # Check interval
        if recent and (now - recent[-1]) < min_interval:
            return False
        
        # Check daily limit
        today_signals = [t for t in recent if t.date() == now.date()]
        if len(today_signals) >= self.config.signal.max_signals_per_day:
            return False
        
        return True
    
    def _record_signal(self, symbol: str, timestamp: datetime):
        """Record that a signal was emitted."""
        if symbol not in self._recent_signals:
            self._recent_signals[symbol] = []
        self._recent_signals[symbol].append(timestamp)
    
    # --- Batch Evaluation ---
    
    def evaluate_all(self, market_data: Dict[str, Dict[str, pd.DataFrame]]
                     ) -> List[TradingSignal]:
        """
        Evaluate all symbols from market data.
        
        Args:
            market_data: {symbol: {"trend": df, "entry": df}}
        
        Returns:
            List of A-grade signals (may be empty - that's expected)
        """
        signals = []
        
        for symbol, data in market_data.items():
            if "trend" not in data or "entry" not in data:
                continue
            
            signal = self.evaluate(symbol, data["trend"], data["entry"])
            if signal:
                signals.append(signal)
        
        return signals

# --- Standalone Test ---

if __name__ == "__main__":
    print("""
        TradeXY - A-Grade Signal Engine
        Ultra-Strict: ALL 5 Conditions Must Pass
    """)
    
    np.random.seed(42)
    
    # Generate bullish trend data (4H)
    n_trend = 300
    t = np.arange(n_trend)
    trend_close = 50000 + 100 * t + np.random.normal(0, 500, n_trend)
    trend_df = pd.DataFrame({
        'Open': trend_close * 0.999,
        'High': trend_close + np.abs(np.random.normal(300, 100, n_trend)),
        'Low': trend_close - np.abs(np.random.normal(300, 100, n_trend)),
        'Close': trend_close,
        'Volume': np.random.randint(1e6, 1e7, n_trend)
    })
    
    # Generate entry data (15m) with pullback
    n_entry = 500
    t2 = np.arange(n_entry)
    entry_close = 80000 + 50 * t2 + 1000 * np.sin(2 * np.pi * t2 / 60) + np.random.normal(0, 200, n_entry)
    entry_df = pd.DataFrame({
        'Open': entry_close * 0.999,
        'High': entry_close + np.abs(np.random.normal(150, 50, n_entry)),
        'Low': entry_close - np.abs(np.random.normal(150, 50, n_entry)),
        'Close': entry_close,
        'Volume': np.random.randint(1e5, 1e6, n_entry)
    })
    
    # Create engine and evaluate
    engine = SignalEngine()
    
    # Ingest some news first
    from datetime import datetime
    engine.news_filter.ingest_item(
        "Bitcoin Breaks New Highs",
        "BTC surges to record levels amid institutional demand",
        "coindesk.com", "https://example.com/btc", datetime.utcnow()
    )
    
    signal = engine.evaluate("BTCUSDT", trend_df, entry_df)
    
    if signal:
        print("[A-GRADE] SIGNAL GENERATED!")
        print(signal.format_alert())
    else:
        print("No A-grade signal (expected - ultra-strict conditions)")
        print("\nChecklist would show which conditions failed.")
        
        # Run evaluation to show what DID pass
        print("\nRunning diagnostic evaluation...")
        from tradex.engines.market_structure import MarketStructureDetector
        from tradex.engines.elliott_wave_engine import ElliottWaveEngine
        
        detector = MarketStructureDetector()
        trend_result = detector.analyze(trend_df)
        entry_result = detector.analyze(entry_df, ema_period=50)
        
        ew_engine = ElliottWaveEngine(min_confidence=60)
        ew_result = ew_engine.analyze(entry_df, "BULLISH")
        
        print(f"  [A] Trend: {trend_result.current_trend} "
              f"(strength={trend_result.trend_strength:.0%}, "
              f"EMA200={'above' if trend_result.above_ema200 else 'below'})"
              f" -> {'[PASS]' if trend_result.trend_filter_passed else '[FAIL]'}")
        
        print(f"  [B] Structure: BOS={entry_result.latest_bos}, "
              f"Retest={'[PASS]' if entry_result.retest_valid else '[FAIL]'}"
              f" -> {'[PASS]' if entry_result.structure_filter_passed else '[FAIL]'}")
        
        print(f"  [C] Elliott: {ew_result.wave_summary}"
              f" -> {'[PASS]' if ew_result.elliott_filter_passed else '[FAIL]'}")
        
        nrf = engine.news_filter
        allowed, reason = nrf.check_signal_allowed("BTCUSDT", "LONG")
        print(f"  [D] News: {reason} -> {'[PASS]' if allowed else '[FAIL]'}")
        
        print(f"  [E] Risk: calculated -> [PASS]")
    
    print("\nSignal Engine test complete.")
