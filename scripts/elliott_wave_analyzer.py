import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from scipy.signal import argrelextrema
FIB_RATIOS = {
    'fib_236': 0.236,
    'fib_382': 0.382,
    'fib_500': 0.500,
    'fib_618': 0.618,
    'fib_786': 0.786,
    'fib_1000': 1.000,
    'fib_1618': 1.618,   # Wave 3 extension target
    'fib_2618': 2.618,   # Wave 5 extension target
}



#  Swing Point Detection

def detect_swing_points(prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local swing highs and swing lows in price data.
    
    Uses scipy.signal.argrelextrema to find local extrema.
    The 'order' parameter controls how many neighbouring points
    to compare — higher order = fewer but more significant swings.

    Args:
        prices: 1D array of closing prices
        order: Number of points on each side to compare (default 5)

    Returns:
        swing_highs: Indices of local maxima
        swing_lows:  Indices of local minima
    """
    swing_highs = argrelextrema(prices, np.greater_equal, order=order)[0]
    swing_lows = argrelextrema(prices, np.less_equal, order=order)[0]
    
    return swing_highs, swing_lows


def merge_swing_points(swing_highs: np.ndarray, swing_lows: np.ndarray,
                       prices: np.ndarray) -> List[Dict]:
    """
    Merge swing highs and lows into a chronologically ordered list
    of alternating pivot points.
    
    Returns:
        List of dicts: [{'index': i, 'price': p, 'type': 'high'|'low'}, ...]
    """
    pivots = []
    
    for idx in swing_highs:
        pivots.append({'index': int(idx), 'price': float(prices[idx]), 'type': 'high'})
    for idx in swing_lows:
        pivots.append({'index': int(idx), 'price': float(prices[idx]), 'type': 'low'})
    
    # Sort chronologically
    pivots.sort(key=lambda x: x['index'])
    
    # Remove consecutive same-type pivots (keep the more extreme one)
    cleaned = []
    for p in pivots:
        if not cleaned or cleaned[-1]['type'] != p['type']:
            cleaned.append(p)
        else:
            # Same type — keep the more extreme
            if p['type'] == 'high' and p['price'] > cleaned[-1]['price']:
                cleaned[-1] = p
            elif p['type'] == 'low' and p['price'] < cleaned[-1]['price']:
                cleaned[-1] = p
    
    return cleaned



#  Elliott Wave Validation Rules (Frost & Prechter, Ch. 2)

def validate_impulse_wave(waves: List[Dict]) -> Dict:
    """
    Validate a potential 5-wave impulse pattern against
    Elliott's three inviolable rules:

    Rule 1: Wave 2 never retraces more than 100% of Wave 1.
    Rule 2: Wave 3 is never the shortest of the three impulse waves (1, 3, 5).
    Rule 3: Wave 4 does not overlap the price territory of Wave 1.

    Also checks guideline-based Fibonacci relationships:
    - Wave 2 typically retraces 50-61.8% of Wave 1
    - Wave 3 is often 1.618× the length of Wave 1
    - Wave 4 typically retraces 38.2% of Wave 3
    - Wave 5 is often 1.0× or 0.618× the length of Wave 1

    Args:
        waves: List of 6 pivot points defining waves 1-2-3-4-5
               [start, end_w1, end_w2, end_w3, end_w4, end_w5]

    Returns:
        Dict with 'valid', 'confidence', 'direction', 'details'
    """
    if len(waves) < 6:
        return {'valid': False, 'confidence': 0.0, 'direction': 0, 'details': 'Insufficient pivot points'}
    
    p0 = waves[0]['price']  # Start
    p1 = waves[1]['price']  # End of Wave 1
    p2 = waves[2]['price']  # End of Wave 2
    p3 = waves[3]['price']  # End of Wave 3
    p4 = waves[4]['price']  # End of Wave 4
    p5 = waves[5]['price']  # End of Wave 5
    
    # Determine direction (bullish if Wave 1 goes up)
    direction = 1 if p1 > p0 else -1
    
    # Wave lengths (absolute moves)
    w1_len = abs(p1 - p0)
    w2_len = abs(p2 - p1)
    w3_len = abs(p3 - p2)
    w4_len = abs(p4 - p3)
    w5_len = abs(p5 - p4)
    
    if w1_len == 0:
        return {'valid': False, 'confidence': 0.0, 'direction': 0, 'details': 'Zero-length Wave 1'}
    
    confidence = 0.0
    violations = []
    
    # ── RULE 1: Wave 2 never retraces > 100% of Wave 1 ──
    w2_retracement = w2_len / w1_len
    if direction == 1:
        rule1_valid = p2 > p0  # In bullish: Wave 2 low must stay above Wave 1 start
    else:
        rule1_valid = p2 < p0  # In bearish: Wave 2 high must stay below Wave 1 start
    
    if rule1_valid:
        confidence += 0.25
    else:
        violations.append('Rule 1 violated: Wave 2 retraces beyond Wave 1 start')
    
    # ── RULE 2: Wave 3 is never the shortest impulse wave ──
    impulse_lengths = [w1_len, w3_len, w5_len]
    if w3_len >= min(impulse_lengths):
        # Wave 3 is not the shortest
        if w3_len == max(impulse_lengths):
            confidence += 0.25  # Best case: Wave 3 is longest
        else:
            confidence += 0.15  # Acceptable: Wave 3 is not shortest
    else:
        violations.append('Rule 2 violated: Wave 3 is shortest impulse wave')
    
    # ── RULE 3: Wave 4 does not overlap Wave 1 price territory ──
    if direction == 1:
        rule3_valid = p4 > p1  # In bullish: Wave 4 low stays above Wave 1 high
    else:
        rule3_valid = p4 < p1  # In bearish: Wave 4 high stays below Wave 1 low
    
    if rule3_valid:
        confidence += 0.20
    else:
        # Allow some tolerance (diagonal triangles violate this)
        overlap = abs(p4 - p1) / w1_len if w1_len > 0 else 1.0
        if overlap < 0.1:
            confidence += 0.10  # Minor violation — possible diagonal
            violations.append('Rule 3 minor violation: slight Wave 4/Wave 1 overlap')
        else:
            violations.append('Rule 3 violated: Wave 4 overlaps Wave 1')
    
    # FIBONACCI GUIDELINES (bonus confidence) 
    # Wave 2 retracement: ideally 50-61.8% of Wave 1
    if 0.382 <= w2_retracement <= 0.786:
        confidence += 0.10
    
    # Wave 3 extension: ideally 1.618× Wave 1
    w3_extension = w3_len / w1_len if w1_len > 0 else 0
    if 1.0 <= w3_extension <= 2.618:
        confidence += 0.10
    
    # Wave 4 retracement: ideally 38.2% of Wave 3
    w4_retracement = w4_len / w3_len if w3_len > 0 else 0
    if 0.236 <= w4_retracement <= 0.618:
        confidence += 0.10
    
    # Cap confidence at 1.0
    confidence = min(confidence, 1.0)
    
    is_valid = len([v for v in violations if 'minor' not in v.lower()]) == 0
    
    return {
        'valid': is_valid,
        'confidence': confidence,
        'direction': direction,
        'details': '; '.join(violations) if violations else 'All rules satisfied',
        'w2_retracement': w2_retracement,
        'w3_extension': w3_extension,
        'w4_retracement': w4_retracement,
        'wave_lengths': [w1_len, w2_len, w3_len, w4_len, w5_len]
    }


def validate_corrective_wave(waves: List[Dict], impulse_direction: int) -> Dict:
    """
    Validate a 3-wave corrective pattern (A-B-C) against Elliott rules.
    
    Corrective patterns retrace against the primary impulse direction.
    
    Types of corrections (Frost & Prechter, Ch. 2):
    - Zigzag: Sharp A-B-C where A and C are impulse waves
    - Flat: B retraces ~100% of A; C ≈ A in length
    - Triangle: Converging A-B-C-D-E (simplified to A-B-C here)

    Args:
        waves: List of 4 pivot points [start, end_A, end_B, end_C]
        impulse_direction: Direction of preceding impulse (1=bull, -1=bear)

    Returns:
        Dict with validation results
    """
    if len(waves) < 4:
        return {'valid': False, 'confidence': 0.0, 'type': 'unknown'}
    
    p0 = waves[0]['price']
    pA = waves[1]['price']
    pB = waves[2]['price']
    pC = waves[3]['price']
    
    confidence = 0.0
    
    a_len = abs(pA - p0)
    b_len = abs(pB - pA)
    c_len = abs(pC - pB)
    
    if a_len == 0:
        return {'valid': False, 'confidence': 0.0, 'type': 'unknown'}
    
    # Wave A moves against the impulse direction
    a_direction = 1 if pA > p0 else -1
    if a_direction != impulse_direction:
        confidence += 0.20  # Correct direction
    
    # Wave B retraces Wave A partially
    b_retracement = b_len / a_len if a_len > 0 else 0
    
    # Wave C typically equals or exceeds Wave A
    c_ratio = c_len / a_len if a_len > 0 else 0
    
    # Classify correction type
    if b_retracement < 0.618 and c_ratio > 0.618:
        corr_type = 'zigzag'
        confidence += 0.30
    elif 0.618 <= b_retracement <= 1.0 and 0.618 <= c_ratio <= 1.382:
        corr_type = 'flat'
        confidence += 0.25
    elif b_retracement > 1.0:
        corr_type = 'expanded_flat'
        confidence += 0.20
    else:
        corr_type = 'complex'
        confidence += 0.10
    
    # Fibonacci alignment bonus
    if abs(b_retracement - 0.618) < 0.1 or abs(b_retracement - 0.500) < 0.1:
        confidence += 0.15
    if abs(c_ratio - 1.0) < 0.15 or abs(c_ratio - 1.618) < 0.15:
        confidence += 0.15
    
    confidence = min(confidence, 1.0)
    
    return {
        'valid': confidence > 0.30,
        'confidence': confidence,
        'type': corr_type,
        'b_retracement': b_retracement,
        'c_ratio': c_ratio
    }


#  Fibonacci Level Calculator
def calculate_fibonacci_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement and extension levels between two price points.
    
    Retracements: 23.6%, 38.2%, 50.0%, 61.8%, 78.6%
    Extensions:   100%, 161.8%, 261.8%
    
    These ratios are derived from the Fibonacci sequence and are central
    to Elliott Wave price projection (Frost & Prechter, Ch. 3).

    Args:
        swing_low: Lower price point
        swing_high: Upper price point

    Returns:
        Dict of level_name → price_value
    """
    diff = swing_high - swing_low
    
    levels = {}
    for name, ratio in FIB_RATIOS.items():
        # Retracement levels (measured down from swing_high)
        levels[f'{name}_retracement'] = swing_high - diff * ratio
        # Extension levels (measured up from swing_low)
        levels[f'{name}_extension'] = swing_low + diff * ratio
    
    return levels


# 
#  Main Elliott Wave Feature Generator
class ElliottWaveAnalyzer:
    """
    Elliott Wave feature generator for stock price time series.
    
    Implements the wave detection algorithm based on Frost & Prechter's
    Elliott Wave Principle, generating 8 quantitative features per
    trading day that can be used as inputs to the hybrid Transformer models.

    Usage:
        analyzer = ElliottWaveAnalyzer(lookback=120, swing_order=5)
        df = analyzer.calculate_elliott_wave_features(df)
        # df now contains 8 new 'ew_*' columns
    """
    
    def __init__(self, lookback: int = 120, swing_order: int = 5):
        """
        Args:
            lookback: Number of trading days to use for wave detection window
            swing_order: Sensitivity of swing point detection (higher = fewer pivots)
        """
        self.lookback = lookback
        self.swing_order = swing_order
    
    def _detect_waves_in_window(self, prices: np.ndarray) -> Dict:
        """
        Detect Elliott Wave pattern in a price window.
        
        Strategy:
        1. Find swing highs/lows
        2. Merge into alternating pivot sequence
        3. Try to fit 5-wave impulse pattern to most recent pivots
        4. If 5-wave found, check for corrective A-B-C after it
        5. Return wave classification for the current position
        """
        if len(prices) < 20:
            return self._neutral_result()
        
        # Detect swing points
        swing_highs, swing_lows = detect_swing_points(prices, order=self.swing_order)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return self._neutral_result()
        
        # Merge into alternating pivots
        pivots = merge_swing_points(swing_highs, swing_lows, prices)
        
        if len(pivots) < 6:
            return self._neutral_result()
        
        # Try to identify the most recent 5-wave impulse pattern
        best_impulse = None
        best_confidence = 0.0
        
        # Slide window over pivots to find best 5-wave fit
        for i in range(len(pivots) - 5):
            candidate = pivots[i:i+6]
            result = validate_impulse_wave(candidate)
            
            if result['confidence'] > best_confidence:
                best_impulse = {
                    'pivots': candidate,
                    'validation': result,
                    'start_idx': candidate[0]['index'],
                    'end_idx': candidate[5]['index']
                }
                best_confidence = result['confidence']
        
        if best_impulse is None or best_confidence < 0.30:
            return self._neutral_result()
        
        # Determine current wave position
        current_idx = len(prices) - 1
        impulse_end = best_impulse['end_idx']
        impulse_pivots = best_impulse['pivots']
        direction = best_impulse['validation']['direction']
        
        # Check which wave the current position falls in
        wave_number, wave_position = self._classify_position(
            current_idx, impulse_pivots, direction, pivots
        )
        
        # Calculate Fibonacci levels from the impulse swing
        if direction == 1:
            swing_low = min(p['price'] for p in impulse_pivots)
            swing_high = max(p['price'] for p in impulse_pivots)
        else:
            swing_low = min(p['price'] for p in impulse_pivots)
            swing_high = max(p['price'] for p in impulse_pivots)
        
        fib_levels = calculate_fibonacci_levels(swing_low, swing_high)
        current_price = prices[-1]
        
        # Distance to Fibonacci levels (normalized by price range)
        price_range = swing_high - swing_low if swing_high != swing_low else 1.0
        fib_382_dist = (current_price - fib_levels['fib_382_retracement']) / price_range
        fib_618_dist = (current_price - fib_levels['fib_618_retracement']) / price_range
        
        # Calculate impulse strength (ratio of Wave 3 to Wave 1)
        wave_lengths = best_impulse['validation'].get('wave_lengths', [0, 0, 0, 0, 0])
        impulse_strength = wave_lengths[2] / wave_lengths[0] if wave_lengths[0] > 0 else 0.0
        impulse_strength = min(impulse_strength / 2.618, 1.0)  # Normalize to 0-1
        
        # Calculate corrective depth
        corrective_depth = 0.0
        if current_idx > impulse_end and len(pivots) > 6:
            # After impulse — we're in a correction
            post_impulse = [p for p in pivots if p['index'] > impulse_end]
            if post_impulse:
                correction_range = max(p['price'] for p in post_impulse) - min(p['price'] for p in post_impulse)
                corrective_depth = correction_range / price_range if price_range > 0 else 0.0
                corrective_depth = min(corrective_depth, 1.0)
        
        return {
            'ew_wave_number': wave_number,
            'ew_wave_direction': direction,
            'ew_wave_position': wave_position,
            'ew_fib_retracement_382': np.clip(fib_382_dist, -2.0, 2.0),
            'ew_fib_retracement_618': np.clip(fib_618_dist, -2.0, 2.0),
            'ew_wave_confidence': best_confidence,
            'ew_impulse_strength': impulse_strength,
            'ew_corrective_depth': corrective_depth
        }
    
    def _classify_position(self, current_idx: int, impulse_pivots: List[Dict],
                           direction: int, all_pivots: List[Dict]) -> Tuple[int, float]:
        """
        Classify which wave the current index falls within.
        
        Returns:
            wave_number: 1-5 for impulse waves, -1/-2/-3 for A/B/C corrective
            wave_position: 0.0 → 1.0 position within that wave
        """
        # Define wave boundaries from impulse pivots
        wave_boundaries = [p['index'] for p in impulse_pivots]
        
        # Check if we're within the 5-wave impulse
        for w in range(5):
            start = wave_boundaries[w]
            end = wave_boundaries[w + 1]
            
            if start <= current_idx <= end:
                wave_len = end - start
                position = (current_idx - start) / wave_len if wave_len > 0 else 0.5
                return w + 1, min(position, 1.0)
        
        # If after the impulse — we're in the corrective phase
        impulse_end = wave_boundaries[-1]
        if current_idx > impulse_end:
            # Find corrective pivots after impulse
            corr_pivots = [p for p in all_pivots if p['index'] > impulse_end]
            
            if len(corr_pivots) >= 3:
                # We have A-B-C structure
                a_end = corr_pivots[0]['index']
                b_end = corr_pivots[1]['index'] if len(corr_pivots) > 1 else current_idx
                c_end = corr_pivots[2]['index'] if len(corr_pivots) > 2 else current_idx
                
                if current_idx <= a_end:
                    pos = (current_idx - impulse_end) / (a_end - impulse_end) if a_end > impulse_end else 0.5
                    return -1, min(pos, 1.0)  # Wave A
                elif current_idx <= b_end:
                    pos = (current_idx - a_end) / (b_end - a_end) if b_end > a_end else 0.5
                    return -2, min(pos, 1.0)  # Wave B
                else:
                    pos = (current_idx - b_end) / (c_end - b_end) if c_end > b_end else 0.5
                    return -3, min(pos, 1.0)  # Wave C
            elif len(corr_pivots) >= 1:
                return -1, 0.5  # Early corrective wave A
            else:
                return -1, 0.0  # Just started correcting
        
        # Before the impulse
        return 0, 0.0
    
    def _neutral_result(self) -> Dict:
        """Return neutral/zero features when no wave pattern is detected."""
        return {
            'ew_wave_number': 0,
            'ew_wave_direction': 0,
            'ew_wave_position': 0.5,
            'ew_fib_retracement_382': 0.0,
            'ew_fib_retracement_618': 0.0,
            'ew_wave_confidence': 0.0,
            'ew_impulse_strength': 0.0,
            'ew_corrective_depth': 0.0
        }
    
    def calculate_elliott_wave_features(self, df: pd.DataFrame,
                                         price_col: str = 'Close') -> pd.DataFrame:
        """
        Calculate Elliott Wave features for every row in the DataFrame.
        
        Uses a rolling lookback window to detect wave patterns at each point in time.
        This ensures no lookahead bias — each row only uses data available up to that point.

        Args:
            df: DataFrame with at least a 'Close' price column
            price_col: Name of the close price column

        Returns:
            DataFrame with 8 new 'ew_*' columns added
        """
        print("  Calculating Elliott Wave features...")
        
        prices = df[price_col].values
        n = len(prices)
        
        # Initialize output arrays
        features = {
            'ew_wave_number': np.zeros(n),
            'ew_wave_direction': np.zeros(n),
            'ew_wave_position': np.full(n, 0.5),
            'ew_fib_retracement_382': np.zeros(n),
            'ew_fib_retracement_618': np.zeros(n),
            'ew_wave_confidence': np.zeros(n),
            'ew_impulse_strength': np.zeros(n),
            'ew_corrective_depth': np.zeros(n)
        }
        
        # Calculate features using rolling window (no lookahead bias)
        for i in range(self.lookback, n):
            window = prices[max(0, i - self.lookback):i + 1]
            
            try:
                result = self._detect_waves_in_window(window)
                for key in features:
                    features[key][i] = result[key]
            except Exception:
                # On any detection error, keep neutral values
                pass
        
        # Forward-fill the first `lookback` rows with the first computed value
        for key in features:
            first_valid_idx = self.lookback if self.lookback < n else 0
            features[key][:first_valid_idx] = features[key][first_valid_idx]
        
        # Add to DataFrame
        for key, values in features.items():
            df[key] = values
        
        print(f"    ✓ Elliott Wave features computed ({n} rows)")
        print(f"    ✓ Detected waves with avg confidence: {np.mean(features['ew_wave_confidence']):.3f}")
        
        return df
    
    def get_current_wave_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a human-readable summary of the current Elliott Wave state.
        Useful for the Streamlit app display.
        """
        if 'ew_wave_number' not in df.columns:
            return {'status': 'Not computed'}
        
        last = df.iloc[-1]
        wave_num = int(last['ew_wave_number'])
        direction = int(last['ew_wave_direction'])
        confidence = float(last['ew_wave_confidence'])
        position = float(last['ew_wave_position'])
        
        # Wave label
        if wave_num > 0:
            wave_label = f"Impulse Wave {wave_num}"
            phase = "Impulse"
        elif wave_num < 0:
            abc_map = {-1: 'A', -2: 'B', -3: 'C'}
            wave_label = f"Corrective Wave {abc_map.get(wave_num, '?')}"
            phase = "Corrective"
        else:
            wave_label = "No clear wave pattern"
            phase = "Neutral"
        
        # Direction
        dir_label = "Bullish" if direction == 1 else "Bearish" if direction == -1 else "Neutral"
        
        # Position within wave
        if position < 0.33:
            pos_label = "Early"
        elif position < 0.67:
            pos_label = "Middle"
        else:
            pos_label = "Late"
        
        # Fibonacci context
        fib_382 = float(last['ew_fib_retracement_382'])
        fib_618 = float(last['ew_fib_retracement_618'])
        
        if abs(fib_382) < 0.05:
            fib_context = "At 38.2% Fibonacci level"
        elif abs(fib_618) < 0.05:
            fib_context = "At 61.8% Fibonacci level"
        elif fib_382 > 0:
            fib_context = "Above 38.2% Fibonacci level"
        else:
            fib_context = "Below 38.2% Fibonacci level"
        
        return {
            'wave_label': wave_label,
            'phase': phase,
            'direction': dir_label,
            'position': pos_label,
            'position_pct': f"{position*100:.0f}%",
            'confidence': f"{confidence*100:.0f}%",
            'fib_context': fib_context,
            'impulse_strength': f"{float(last['ew_impulse_strength'])*100:.0f}%",
            'corrective_depth': f"{float(last['ew_corrective_depth'])*100:.0f}%"
        }


#  Convenience Functions
def add_elliott_wave_features(df: pd.DataFrame, price_col: str = 'Close',
                               lookback: int = 120, swing_order: int = 5) -> pd.DataFrame:
    """
    Convenience function to add Elliott Wave features to a DataFrame.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the close price column
        lookback: Lookback window for wave detection
        swing_order: Swing point sensitivity
    
    Returns:
        DataFrame with 8 new 'ew_*' columns
    """
    analyzer = ElliottWaveAnalyzer(lookback=lookback, swing_order=swing_order)
    return analyzer.calculate_elliott_wave_features(df, price_col=price_col)
