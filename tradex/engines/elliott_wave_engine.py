import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tradex.engines.market_structure import MarketStructureDetector, SwingPoint, SwingType

# --- Data Structures ---

class WavePhase(Enum):
    IMPULSE = "IMPULSE"         # Waves 1–5
    CORRECTIVE = "CORRECTIVE"   # Waves A–C

class WaveLabel(Enum):
    W1 = "1"
    W2 = "2"
    W3 = "3"
    W4 = "4"
    W5 = "5"
    WA = "A"
    WB = "B"
    WC = "C"
    UNKNOWN = "?"

@dataclass
class WaveCandidate:
    """A candidate wave count with confidence score."""
    waves: List[Dict]           # List of wave segments [{label, start_idx, end_idx, start_price, end_price}]
    phase: WavePhase
    current_wave: WaveLabel     # Which wave we think we're in now
    next_expected: WaveLabel    # What comes next
    confidence: int             # 0–100
    fib_score: float            # 0–1 Fibonacci alignment score
    momentum_score: float       # 0–1 Momentum confirmation score
    rule_violations: List[str]  # Any Elliott rules violated
    direction: str              # "BULLISH" or "BEARISH"
    
    # Entry relevance
    is_wave3_entry: bool = False    # "Wave 2 complete -> Wave 3 start"
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    
    def __repr__(self):
        return (f"Elliott({self.direction} {self.phase.value}: "
                f"Wave {self.current_wave.value} -> {self.next_expected.value}, "
                f"conf={self.confidence}/100)")

@dataclass
class ElliottWaveResult:
    """Complete Elliott Wave analysis output."""
    candidates: List[WaveCandidate]     # Top N wave count candidates
    best_candidate: Optional[WaveCandidate]
    wave_summary: str                   # Human readable label
    confidence: int                     # Best candidate confidence
    is_actionable: bool                 # confidence >= min_confidence
    elliott_filter_passed: bool         # Condition C for signal engine
    
    # Fibonacci levels for display
    fib_levels: Dict[str, float]        # {"0.236": price, "0.382": price, ...}
    
    # Feature vector for ML models
    features: Dict[str, float]

# --- Elliott Wave Engine ---

class ElliottWaveEngine:
    """
    Enhanced Elliott Wave engine for the TradeXY signal system.
    
    Pipeline:
        1. Build swing points using ZigZag (ATR-based)
        2. Generate candidate impulse counts for Wave1/Wave2
        3. Fibonacci validation scoring
        4. Momentum validation scoring (RSI/MACD)
        5. Return top 2 counts with confidence
    """
    
    def __init__(self, 
                 wave2_fib_min: float = 0.500,
                 wave2_fib_max: float = 0.786,
                 wave2_fib_ideal: float = 0.618,
                 wave3_min_extension: float = 1.618,
                 wave4_fib_min: float = 0.236,
                 wave4_fib_max: float = 0.500,
                 min_confidence: int = 80,
                 swing_order: int = 5,
                 top_n: int = 2):
        
        self.wave2_fib_min = wave2_fib_min
        self.wave2_fib_max = wave2_fib_max
        self.wave2_fib_ideal = wave2_fib_ideal
        self.wave3_min_extension = wave3_min_extension
        self.wave4_fib_min = wave4_fib_min
        self.wave4_fib_max = wave4_fib_max
        self.min_confidence = min_confidence
        self.top_n = top_n
        
        self.structure_detector = MarketStructureDetector(swing_order=swing_order)
    
    # --- RSI Calculation ---
    
    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI for momentum confirmation."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
        
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    # --- MACD Calculation ---
    
    def _calc_macd(self, close: np.ndarray, fast: int = 12, 
                   slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD line and histogram."""
        def ema(data, period):
            result = np.zeros(len(data))
            mult = 2.0 / (period + 1)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = (data[i] - result[i - 1]) * mult + result[i - 1]
            return result
        
        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, histogram
    
    # --- Fibonacci Levels ---
    
    def _calc_fib_levels(self, start_price: float, 
                         end_price: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement and extension levels.
        For an upward Wave 1 (start < end):
            retracement = end - (end - start) * ratio
        """
        diff = end_price - start_price
        
        return {
            "0.236": end_price - diff * 0.236,
            "0.382": end_price - diff * 0.382,
            "0.500": end_price - diff * 0.500,
            "0.618": end_price - diff * 0.618,
            "0.786": end_price - diff * 0.786,
            "1.000": start_price,  # Full retracement
            "1.618_ext": end_price + diff * 0.618,  # Wave 3 target 1
            "2.618_ext": end_price + diff * 1.618,  # Wave 3 target 2
        }
    
    # --- Wave 1/2 Candidate Detection ---
    
    def _find_wave12_candidates(self, swings: List[SwingPoint], 
                                close: np.ndarray,
                                direction: str = "BULLISH"
                                ) -> List[WaveCandidate]:
        """
        Scan swing points for potential Wave 1 -> Wave 2 completions.
        
        For BULLISH:
            Wave 1 = swing low -> swing high (upward impulse)
            Wave 2 = swing high -> swing low (corrective pullback)
            Validate: Wave 2 retraces 50%–78.6% of Wave 1, doesn't break below Wave 1 start
        
        For BEARISH: symmetric opposite.
        """
        candidates = []
        
        if len(swings) < 4:
            return candidates
        
        # Scan potential triplets: W1_start, W1_end/W2_start, W2_end
        for i in range(len(swings) - 2):
            s0 = swings[i]      # Wave 1 start
            s1 = swings[i + 1]  # Wave 1 end / Wave 2 start
            s2 = swings[i + 2]  # Wave 2 end
            
            if direction == "BULLISH":
                # Wave 1: LOW -> HIGH (upward)
                if s0.swing_type != SwingType.LOW or s1.swing_type != SwingType.HIGH:
                    continue
                # Wave 2: HIGH -> LOW (corrective pullback)
                if s2.swing_type != SwingType.LOW:
                    continue
                
                w1_start = s0.price
                w1_end = s1.price
                w2_end = s2.price
                
                # Basic validation
                if w1_end <= w1_start:
                    continue  # Not an upward impulse
                
                # Rule 1: Wave 2 must NOT break below Wave 1 start
                if w2_end < w1_start:
                    continue  # Violated - skip
                
                # Fibonacci validation
                w1_range = w1_end - w1_start
                w2_retrace = (w1_end - w2_end) / w1_range if w1_range > 0 else 0
                
            elif direction == "BEARISH":
                # Wave 1: HIGH -> LOW (downward)
                if s0.swing_type != SwingType.HIGH or s1.swing_type != SwingType.LOW:
                    continue
                if s2.swing_type != SwingType.HIGH:
                    continue
                
                w1_start = s0.price
                w1_end = s1.price
                w2_end = s2.price
                
                if w1_end >= w1_start:
                    continue
                
                if w2_end > w1_start:
                    continue
                
                w1_range = abs(w1_start - w1_end)
                w2_retrace = abs(w2_end - w1_end) / w1_range if w1_range > 0 else 0
            else:
                continue
            
            # --- Fibonacci Score ---
            fib_score = 0.0
            
            # Check if retrace is in the golden zone (0.5–0.786)
            if self.wave2_fib_min <= w2_retrace <= self.wave2_fib_max:
                # How close to ideal 0.618?
                distance_to_ideal = abs(w2_retrace - self.wave2_fib_ideal)
                fib_score = max(0, 1.0 - distance_to_ideal / 0.3)
            elif 0.382 <= w2_retrace < self.wave2_fib_min:
                fib_score = 0.3  # Shallow retrace - less ideal
            elif w2_retrace > self.wave2_fib_max:
                fib_score = max(0, 0.5 - (w2_retrace - self.wave2_fib_max))
            
            # --- Momentum Score ---
            rsi = self._calc_rsi(close)
            _, macd_hist = self._calc_macd(close)
            
            momentum_score = 0.0
            w2_end_idx = s2.index
            
            if w2_end_idx < len(rsi) and w2_end_idx < len(macd_hist):
                if direction == "BULLISH":
                    # RSI should be recovering (not oversold at W2 end = good)
                    if 30 <= rsi[w2_end_idx] <= 55:
                        momentum_score += 0.5  # RSI at launch zone
                    elif rsi[w2_end_idx] < 30:
                        momentum_score += 0.3  # Oversold - potential
                    
                    # MACD histogram should be turning positive
                    if w2_end_idx >= 2:
                        if macd_hist[w2_end_idx] > macd_hist[w2_end_idx - 1]:
                            momentum_score += 0.3  # Histogram expanding
                        if macd_hist[w2_end_idx] > 0:
                            momentum_score += 0.2  # Already positive
                
                elif direction == "BEARISH":
                    if 45 <= rsi[w2_end_idx] <= 70:
                        momentum_score += 0.5
                    elif rsi[w2_end_idx] > 70:
                        momentum_score += 0.3
                    
                    if w2_end_idx >= 2:
                        if macd_hist[w2_end_idx] < macd_hist[w2_end_idx - 1]:
                            momentum_score += 0.3
                        if macd_hist[w2_end_idx] < 0:
                            momentum_score += 0.2
            
            momentum_score = min(momentum_score, 1.0)
            
            # --- Confidence Calculation ---
            rule_violations = []
            rule_score = 1.0
            
            # Rule: Wave 2 should not retrace more than 100%
            if w2_retrace > 1.0:
                rule_violations.append("Wave 2 retraces > 100% of Wave 1")
                rule_score -= 0.5
            
            # Guideline: Ideal retrace in 0.5–0.786 zone
            if not (self.wave2_fib_min <= w2_retrace <= self.wave2_fib_max):
                rule_violations.append(f"Wave 2 retrace {w2_retrace:.1%} outside ideal zone")
                rule_score -= 0.15
            
            # Guideline: Wave 1 should have meaningful size
            if w1_range < close[s0.index] * 0.005:  # Less than 0.5% move
                rule_violations.append("Wave 1 too small")
                rule_score -= 0.2
            
            rule_score = max(rule_score, 0.0)
            
            # Final confidence: weighted combination
            confidence = int(
                (fib_score * 40 + momentum_score * 30 + rule_score * 30)
            )
            confidence = min(max(confidence, 0), 100)
            
            # --- Fib Levels ---
            fib_levels = self._calc_fib_levels(w1_start, w1_end)
            
            # --- Entry Zone ---
            if direction == "BULLISH":
                entry_zone_low = fib_levels["0.618"]
                entry_zone_high = fib_levels["0.500"]
            else:
                entry_zone_low = fib_levels["0.500"]
                entry_zone_high = fib_levels["0.618"]
            
            # --- Build Candidate ---
            is_w3_entry = (fib_score >= 0.5 and confidence >= self.min_confidence)
            
            wave_segments = [
                {"label": "W1", "start_idx": s0.index, "end_idx": s1.index,
                 "start_price": w1_start, "end_price": w1_end},
                {"label": "W2", "start_idx": s1.index, "end_idx": s2.index,
                 "start_price": w1_end, "end_price": w2_end},
            ]
            
            candidates.append(WaveCandidate(
                waves=wave_segments,
                phase=WavePhase.IMPULSE,
                current_wave=WaveLabel.W2,
                next_expected=WaveLabel.W3,
                confidence=confidence,
                fib_score=fib_score,
                momentum_score=momentum_score,
                rule_violations=rule_violations,
                direction=direction,
                is_wave3_entry=is_w3_entry,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high
            ))
        
        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates[:self.top_n * 2]  # Return extra for filtering
    
    # --- Main Analysis ---
    
    def analyze(self, df: pd.DataFrame, 
                trend_direction: str = "BULLISH") -> ElliottWaveResult:
        """
        Run the full Elliott Wave analysis pipeline.
        
        Args:
            df: OHLCV DataFrame  
            trend_direction: "BULLISH" or "BEARISH" from trend filter
        
        Returns:
            ElliottWaveResult with candidates, confidence, and actionability
        """
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        close = df['Close'].values.astype(float)
        timestamps = df.index.values if isinstance(df.index, pd.DatetimeIndex) else None
        
        # Step 1: Detect swing points
        swings = self.structure_detector.detect_swing_points(
            high, low, close, timestamps
        )
        
        # Step 2: Find Wave 1/2 candidates
        candidates = self._find_wave12_candidates(swings, close, trend_direction)
        
        # Also check opposite direction as secondary count
        opposite = "BEARISH" if trend_direction == "BULLISH" else "BULLISH"
        alt_candidates = self._find_wave12_candidates(swings, close, opposite)
        
        # Combine and sort
        all_candidates = candidates + alt_candidates
        all_candidates.sort(key=lambda c: c.confidence, reverse=True)
        top_candidates = all_candidates[:self.top_n]
        
        # Best candidate
        best = top_candidates[0] if top_candidates else None
        
        # Summary
        if best:
            wave_summary = (f"{best.direction} {best.phase.value}: "
                          f"Wave {best.current_wave.value} -> {best.next_expected.value} "
                          f"(conf: {best.confidence}/100)")
        else:
            wave_summary = "No clear wave pattern detected"
        
        # Confidence
        confidence = best.confidence if best else 0
        is_actionable = confidence >= self.min_confidence
        
        # Condition C: Elliott filter
        elliott_filter_passed = (
            best is not None 
            and best.is_wave3_entry 
            and best.confidence >= self.min_confidence
            and best.direction == trend_direction
        )
        
        # Fibonacci levels from best candidate
        fib_levels = {}
        if best and len(best.waves) >= 1:
            w1 = best.waves[0]
            fib_levels = self._calc_fib_levels(w1["start_price"], w1["end_price"])
        
        # Feature vector for ML models
        features = {
            "ew_confidence": confidence / 100.0,
            "ew_fib_score": best.fib_score if best else 0.0,
            "ew_momentum_score": best.momentum_score if best else 0.0,
            "ew_is_wave3_entry": 1.0 if (best and best.is_wave3_entry) else 0.0,
            "ew_direction": 1.0 if (best and best.direction == "BULLISH") else -1.0,
            "ew_wave_number": self._wave_label_to_number(
                best.current_wave if best else WaveLabel.UNKNOWN
            ),
            "ew_violations": len(best.rule_violations) if best else 0,
        }
        
        return ElliottWaveResult(
            candidates=top_candidates,
            best_candidate=best,
            wave_summary=wave_summary,
            confidence=confidence,
            is_actionable=is_actionable,
            elliott_filter_passed=elliott_filter_passed,
            fib_levels=fib_levels,
            features=features
        )
    
    def _wave_label_to_number(self, label: WaveLabel) -> float:
        """Convert wave label to numeric for features."""
        mapping = {
            WaveLabel.W1: 1, WaveLabel.W2: 2, WaveLabel.W3: 3,
            WaveLabel.W4: 4, WaveLabel.W5: 5,
            WaveLabel.WA: -1, WaveLabel.WB: -2, WaveLabel.WC: -3,
            WaveLabel.UNKNOWN: 0
        }
        return float(mapping.get(label, 0))
    
    def get_summary(self, result: ElliottWaveResult) -> Dict:
        """Human-readable summary for the dashboard."""
        best = result.best_candidate
        return {
            "wave_summary": result.wave_summary,
            "confidence": f"{result.confidence}/100",
            "actionable": "Yes" if result.is_actionable else "No",
            "elliott_filter": "PASS" if result.elliott_filter_passed else "FAIL",
            "candidates": len(result.candidates),
            "direction": best.direction if best else "N/A",
            "current_wave": best.current_wave.value if best else "?",
            "next_expected": best.next_expected.value if best else "?",
            "fib_score": f"{best.fib_score:.0%}" if best else "N/A",
            "momentum_score": f"{best.momentum_score:.0%}" if best else "N/A",
            "is_wave3_entry": "Yes" if (best and best.is_wave3_entry) else "No",
            "violations": best.rule_violations if best else [],
            "entry_zone": (
                f"{best.entry_zone_low:.4f} – {best.entry_zone_high:.4f}" 
                if best else "N/A"
            ),
        }

# --- Standalone Test ---

if __name__ == "__main__":
    print("""
        TradeXY - Enhanced Elliott Wave Engine
        ZigZag + Fibonacci + Momentum Validation
    """)
    
    np.random.seed(42)
    n = 500
    t = np.arange(n)
    
    # Create realistic impulse pattern: W1 up, W2 pullback, W3 start
    trend = 100 + 0.15 * t
    impulse = 15 * np.sin(2 * np.pi * t / 60)
    correction = 8 * np.sin(2 * np.pi * t / 25)
    noise = np.random.normal(0, 2, n)
    close = trend + impulse + correction + noise
    
    df = pd.DataFrame({
        'Open': close * 0.999,
        'High': close + np.abs(np.random.normal(1.5, 0.5, n)),
        'Low': close - np.abs(np.random.normal(1.5, 0.5, n)),
        'Close': close,
        'Volume': np.random.randint(1e6, 1e7, n)
    })
    
    engine = ElliottWaveEngine(min_confidence=60)  # Lower for test
    result = engine.analyze(df, trend_direction="BULLISH")
    summary = engine.get_summary(result)
    
    print("Elliott Wave Analysis:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    if result.fib_levels:
        print(f"\nFibonacci Levels:")
        for level, price in result.fib_levels.items():
            print(f"  {level}: {price:.4f}")
    
    print(f"\nML Features:")
    for key, value in result.features.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nElliott Wave Engine test complete.")
