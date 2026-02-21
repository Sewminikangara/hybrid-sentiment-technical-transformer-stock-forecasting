import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

# --- Data Structures ---

class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"

class StructureType(Enum):
    HH = "HH"   # Higher High
    HL = "HL"    # Higher Low
    LH = "LH"   # Lower High
    LL = "LL"    # Lower Low
    EQ = "EQ"    # Equal (no clear direction)

class BOSDirection(Enum):
    BULLISH = "BULLISH"    # Break above previous swing high
    BEARISH = "BEARISH"    # Break below previous swing low
    NONE = "NONE"

@dataclass
class SwingPoint:
    """A detected swing high or swing low point."""
    index: int              # Bar index in the DataFrame
    price: float            # Price at the swing point
    swing_type: SwingType   # HIGH or LOW
    timestamp: Optional[pd.Timestamp] = None
    
    def __repr__(self):
        return f"{self.swing_type.value}@{self.price:.4f}[{self.index}]"

@dataclass
class StructureLabel:
    """A market structure label (HH/HL/LH/LL) for a swing point."""
    swing: SwingPoint
    label: StructureType
    
    def __repr__(self):
        return f"{self.label.value}@{self.swing.price:.4f}"

@dataclass
class BOSEvent:
    """Break of Structure event - a structural shift in price."""
    direction: BOSDirection
    break_bar: int          # Bar where the break occurred
    break_price: float      # Price at the break
    broken_level: float     # The swing level that was broken
    confirmed: bool = False # Was the retest validated?
    retest_bar: Optional[int] = None
    retest_price: Optional[float] = None
    
    def __repr__(self):
        status = "confirmed" if self.confirmed else "pending"
        return f"BOS_{self.direction.value}@{self.break_price:.4f} ({status})"

@dataclass
class MarketStructureResult:
    """Complete market structure analysis output."""
    swing_points: List[SwingPoint]
    structure_labels: List[StructureLabel]
    bos_events: List[BOSEvent]
    current_trend: str          # "BULLISH", "BEARISH", "RANGING"
    trend_strength: float       # 0.0–1.0
    above_ema200: bool
    latest_bos: Optional[BOSEvent]
    retest_valid: bool
    
    # Summary for signal engine
    trend_filter_passed: bool   # Condition A
    structure_filter_passed: bool  # Condition B

# --- Market Structure Detector ---

class MarketStructureDetector:
    """
    Detects market structure (HH/HL/LH/LL), Break of Structure (BOS),
    and validates retests for the TradeXY signal engine.
    
    Used for:
        Condition A - Trend filter on higher timeframe
        Condition B - Entry structure on lower timeframe
    """
    
    def __init__(self, swing_order: int = 5, atr_period: int = 14,
                 min_swing_atr: float = 0.5, bos_min_break_atr: float = 0.3,
                 retest_tolerance_atr: float = 0.5, retest_hold_bars: int = 3):
        self.swing_order = swing_order
        self.atr_period = atr_period
        self.min_swing_atr = min_swing_atr
        self.bos_min_break_atr = bos_min_break_atr
        self.retest_tolerance_atr = retest_tolerance_atr
        self.retest_hold_bars = retest_hold_bars
    
    # --- Step 1: ATR Calculation ---
    
    def _calc_atr(self, high: np.ndarray, low: np.ndarray, 
                  close: np.ndarray) -> np.ndarray:
        """Average True Range for dynamic thresholds."""
        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
        
        atr = np.zeros(n)
        atr[:self.atr_period] = np.mean(tr[:self.atr_period])
        
        for i in range(self.atr_period, n):
            atr[i] = (atr[i - 1] * (self.atr_period - 1) + tr[i]) / self.atr_period
        
        return atr
    
    # --- Step 2: Swing Point Detection (ZigZag) ---
    
    def detect_swing_points(self, high: np.ndarray, low: np.ndarray,
                            close: np.ndarray, 
                            timestamps: Optional[np.ndarray] = None
                            ) -> List[SwingPoint]:
        """
        Detect swing highs and lows using a ZigZag algorithm with
        ATR-based minimum swing size filtering.
        
        This produces alternating HIGH/LOW points that form the
        skeleton of market structure.
        """
        n = len(high)
        if n < self.swing_order * 2 + 1:
            return []
        
        atr = self._calc_atr(high, low, close)
        
        # Find local maxima and minima
        raw_highs = []
        raw_lows = []
        
        for i in range(self.swing_order, n - self.swing_order):
            # Swing High: high[i] is the highest in the window
            window_high = high[i - self.swing_order: i + self.swing_order + 1]
            if high[i] == np.max(window_high):
                raw_highs.append(SwingPoint(
                    index=i, price=float(high[i]), swing_type=SwingType.HIGH,
                    timestamp=timestamps[i] if timestamps is not None else None
                ))
            
            # Swing Low: low[i] is the lowest in the window
            window_low = low[i - self.swing_order: i + self.swing_order + 1]
            if low[i] == np.min(window_low):
                raw_lows.append(SwingPoint(
                    index=i, price=float(low[i]), swing_type=SwingType.LOW,
                    timestamp=timestamps[i] if timestamps is not None else None
                ))
        
        # Merge and sort by index
        all_swings = sorted(raw_highs + raw_lows, key=lambda s: s.index)
        
        if len(all_swings) < 2:
            return all_swings
        
        # Filter: enforce alternating HIGH/LOW (ZigZag property)
        filtered = [all_swings[0]]
        
        for swing in all_swings[1:]:
            if swing.swing_type == filtered[-1].swing_type:
                # Same type - keep the more extreme one
                if swing.swing_type == SwingType.HIGH:
                    if swing.price > filtered[-1].price:
                        filtered[-1] = swing
                else:
                    if swing.price < filtered[-1].price:
                        filtered[-1] = swing
            else:
                # Different type - check minimum swing size
                swing_size = abs(swing.price - filtered[-1].price)
                min_size = atr[swing.index] * self.min_swing_atr
                
                if swing_size >= min_size:
                    filtered.append(swing)
                # else: too small, skip
        
        return filtered
    
    # --- Step 3: Structure Labeling (HH/HL/LH/LL) ---
    
    def label_structure(self, swings: List[SwingPoint]) -> List[StructureLabel]:
        """
        Label each swing point as HH, HL, LH, LL, or EQ by comparing
        to the previous swing of the same type.
        """
        if len(swings) < 3:
            return []
        
        labels = []
        prev_high: Optional[SwingPoint] = None
        prev_low: Optional[SwingPoint] = None
        
        for swing in swings:
            if swing.swing_type == SwingType.HIGH:
                if prev_high is not None:
                    if swing.price > prev_high.price * 1.0001:
                        labels.append(StructureLabel(swing, StructureType.HH))
                    elif swing.price < prev_high.price * 0.9999:
                        labels.append(StructureLabel(swing, StructureType.LH))
                    else:
                        labels.append(StructureLabel(swing, StructureType.EQ))
                prev_high = swing
            
            elif swing.swing_type == SwingType.LOW:
                if prev_low is not None:
                    if swing.price > prev_low.price * 1.0001:
                        labels.append(StructureLabel(swing, StructureType.HL))
                    elif swing.price < prev_low.price * 0.9999:
                        labels.append(StructureLabel(swing, StructureType.LL))
                    else:
                        labels.append(StructureLabel(swing, StructureType.EQ))
                prev_low = swing
        
        return labels
    
    # --- Step 4: Trend Classification ---
    
    def classify_trend(self, labels: List[StructureLabel],
                       lookback: int = 6) -> Tuple[str, float]:
        """
        Classify the current trend from recent structure labels.
        
        Returns:
            (trend: str, strength: float 0–1)
        """
        if len(labels) < 2:
            return "RANGING", 0.0
        
        recent = labels[-lookback:]
        
        bullish_count = sum(1 for l in recent 
                          if l.label in (StructureType.HH, StructureType.HL))
        bearish_count = sum(1 for l in recent 
                          if l.label in (StructureType.LH, StructureType.LL))
        total = len(recent)
        
        if total == 0:
            return "RANGING", 0.0
        
        bullish_pct = bullish_count / total
        bearish_pct = bearish_count / total
        
        if bullish_pct >= 0.65:
            return "BULLISH", min(bullish_pct, 1.0)
        elif bearish_pct >= 0.65:
            return "BEARISH", min(bearish_pct, 1.0)
        else:
            return "RANGING", max(bullish_pct, bearish_pct)
    
    # --- Step 5: Break of Structure Detection ---
    
    def detect_bos(self, swings: List[SwingPoint], close: np.ndarray,
                   high: np.ndarray, low: np.ndarray) -> List[BOSEvent]:
        """
        Detect Break of Structure (BOS) events.
        
        Bullish BOS: Price closes above the most recent swing high.
        Bearish BOS: Price closes below the most recent swing low.
        """
        if len(swings) < 3:
            return []
        
        atr = self._calc_atr(high, low, close)
        events = []
        
        # Track the most recent swing high and swing low
        recent_swing_high = None
        recent_swing_low = None
        
        for swing in swings:
            if swing.swing_type == SwingType.HIGH:
                recent_swing_high = swing
            elif swing.swing_type == SwingType.LOW:
                recent_swing_low = swing
            
            # After we have both references, check for BOS in subsequent bars
            if recent_swing_high and recent_swing_low:
                # Check for bars AFTER this swing
                start_bar = swing.index + 1
                end_bar = min(swing.index + 30, len(close))  # Look up to 30 bars
                
                for bar in range(start_bar, end_bar):
                    bar_atr = atr[bar] if bar < len(atr) else atr[-1]
                    
                    # Bullish BOS: close above recent swing high
                    if (close[bar] > recent_swing_high.price + 
                        bar_atr * self.bos_min_break_atr):
                        events.append(BOSEvent(
                            direction=BOSDirection.BULLISH,
                            break_bar=bar,
                            break_price=float(close[bar]),
                            broken_level=recent_swing_high.price
                        ))
                        break
                    
                    # Bearish BOS: close below recent swing low
                    if (close[bar] < recent_swing_low.price - 
                        bar_atr * self.bos_min_break_atr):
                        events.append(BOSEvent(
                            direction=BOSDirection.BEARISH,
                            break_bar=bar,
                            break_price=float(close[bar]),
                            broken_level=recent_swing_low.price
                        ))
                        break
        
        return events
    
    # --- Step 6: Retest Validation ---
    
    def validate_retest(self, bos: BOSEvent, close: np.ndarray,
                        high: np.ndarray, low: np.ndarray) -> BOSEvent:
        """
        Validate that price retested the broken level and held.
        
        Bullish BOS retest: price pulls back to broken level, holds, bounces.
        Bearish BOS retest: price rallies to broken level, holds, drops.
        """
        if bos.break_bar >= len(close) - self.retest_hold_bars:
            return bos  # Not enough bars after BOS
        
        atr = self._calc_atr(high, low, close)
        tolerance = atr[bos.break_bar] * self.retest_tolerance_atr
        
        # Scan bars after the BOS for a retest
        search_end = min(bos.break_bar + 30, len(close))
        
        for bar in range(bos.break_bar + 1, search_end):
            if bos.direction == BOSDirection.BULLISH:
                # Price must pull back near the broken level (now support)
                if abs(low[bar] - bos.broken_level) <= tolerance:
                    # Check if it holds for N bars
                    hold_end = min(bar + self.retest_hold_bars, len(close))
                    held = all(
                        close[b] > bos.broken_level - tolerance * 0.5
                        for b in range(bar, hold_end)
                    )
                    if held:
                        bos.confirmed = True
                        bos.retest_bar = bar
                        bos.retest_price = float(low[bar])
                        return bos
            
            elif bos.direction == BOSDirection.BEARISH:
                # Price must rally near the broken level (now resistance)
                if abs(high[bar] - bos.broken_level) <= tolerance:
                    hold_end = min(bar + self.retest_hold_bars, len(close))
                    held = all(
                        close[b] < bos.broken_level + tolerance * 0.5
                        for b in range(bar, hold_end)
                    )
                    if held:
                        bos.confirmed = True
                        bos.retest_bar = bar
                        bos.retest_price = float(high[bar])
                        return bos
        
        return bos  # No confirmed retest found
    
    # --- Step 7: EMA200 Check ---
    
    def _calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        ema = np.zeros(len(data))
        multiplier = 2.0 / (period + 1)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema
    
    # --- Main Analysis ---
    
    def analyze(self, df: pd.DataFrame, ema_period: int = 200
                ) -> MarketStructureResult:
        """
        Run full market structure analysis on OHLCV data.
        
        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
            ema_period: EMA period for trend bias
        
        Returns:
            MarketStructureResult with all analysis outputs
        """
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        close = df['Close'].values.astype(float)
        timestamps = df.index.values if isinstance(df.index, pd.DatetimeIndex) else None
        
        # Step 1: Detect swing points
        swings = self.detect_swing_points(high, low, close, timestamps)
        
        # Step 2: Label structure
        labels = self.label_structure(swings)
        
        # Step 3: Classify trend
        trend, strength = self.classify_trend(labels)
        
        # Step 4: EMA200 check
        ema200 = self._calc_ema(close, ema_period)
        above_ema200 = bool(close[-1] > ema200[-1]) if len(close) > 0 else False
        
        # Step 5: Detect BOS
        bos_events = self.detect_bos(swings, close, high, low)
        
        # Step 6: Validate retests for recent BOS events
        for i, bos in enumerate(bos_events[-3:]):  # Only check last 3
            idx = len(bos_events) - 3 + i
            if idx >= 0:
                bos_events[idx] = self.validate_retest(bos, close, high, low)
        
        latest_bos = bos_events[-1] if bos_events else None
        retest_valid = latest_bos.confirmed if latest_bos else False
        
        # --- Signal Condition Checks ---
        
        # Condition A: Trend filter
        # Bullish: price above EMA200 AND bullish structure (HH + HL)
        # Bearish: price below EMA200 AND bearish structure (LH + LL)
        trend_filter_passed = False
        if trend == "BULLISH" and above_ema200 and strength >= 0.6:
            trend_filter_passed = True
        elif trend == "BEARISH" and not above_ema200 and strength >= 0.6:
            trend_filter_passed = True
        
        # Condition B: Structure confirmation (BOS + retest)
        structure_filter_passed = False
        if latest_bos and latest_bos.confirmed:
            # Bullish BOS for long, Bearish BOS for short
            if trend == "BULLISH" and latest_bos.direction == BOSDirection.BULLISH:
                structure_filter_passed = True
            elif trend == "BEARISH" and latest_bos.direction == BOSDirection.BEARISH:
                structure_filter_passed = True
        
        return MarketStructureResult(
            swing_points=swings,
            structure_labels=labels,
            bos_events=bos_events,
            current_trend=trend,
            trend_strength=strength,
            above_ema200=above_ema200,
            latest_bos=latest_bos,
            retest_valid=retest_valid,
            trend_filter_passed=trend_filter_passed,
            structure_filter_passed=structure_filter_passed
        )
    
    def get_summary(self, result: MarketStructureResult) -> Dict:
        """Human-readable summary for dashboard display."""
        return {
            "trend": result.current_trend,
            "trend_strength": f"{result.trend_strength:.0%}",
            "above_ema200": "Yes" if result.above_ema200 else "No",
            "swing_count": len(result.swing_points),
            "structure_labels": len(result.structure_labels),
            "bos_events": len(result.bos_events),
            "latest_bos": str(result.latest_bos) if result.latest_bos else "None",
            "retest_confirmed": "Yes" if result.retest_valid else "No",
            "trend_filter": "PASS" if result.trend_filter_passed else "FAIL",
            "structure_filter": "PASS" if result.structure_filter_passed else "FAIL",
        }

# --- Standalone Test ---

if __name__ == "__main__":
    print("""
        TradeXY - Market Structure Detector
        HH/HL/LH/LL + BOS + Retest Validation
    """)
    
    # Generate synthetic trending data
    np.random.seed(42)
    n = 500
    t = np.arange(n)
    
    # Bullish trend with pullbacks
    trend = 100 + 0.1 * t
    cycles = 8 * np.sin(2 * np.pi * t / 40) + 4 * np.sin(2 * np.pi * t / 15)
    noise = np.random.normal(0, 1.5, n)
    close = trend + cycles + noise
    high = close + np.abs(np.random.normal(1, 0.5, n))
    low = close - np.abs(np.random.normal(1, 0.5, n))
    
    df = pd.DataFrame({
        'Open': close * 0.999,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': np.random.randint(1e6, 1e7, n)
    })
    
    # Run analysis
    detector = MarketStructureDetector()
    result = detector.analyze(df)
    summary = detector.get_summary(result)
    
    print("Market Structure Analysis:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nSwing Points (last 10):")
    for sp in result.swing_points[-10:]:
        print(f"  {sp}")
    
    print(f"\nStructure Labels (last 8):")
    for sl in result.structure_labels[-8:]:
        print(f"  {sl}")
    
    print(f"\nBOS Events (last 5):")
    for bos in result.bos_events[-5:]:
        print(f"  {bos}")
    
    print("\nMarket Structure Detector test complete.")
