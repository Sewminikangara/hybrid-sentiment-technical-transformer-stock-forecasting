"""
Adaptive Thresholds (Volatility-Based)

"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class AdaptiveParams:
    """Adjusted parameters for the current volatility regime."""
    regime: VolatilityRegime
    regime_score: float           # 0.0 (calm) to 1.0 (extreme)

    # Adjusted signal engine parameters
    sl_atr_multiplier: float      # Stop loss ATR multiplier
    min_confidence: int           # Elliott Wave min confidence
    signal_cooldown_min: int      # Minutes between signals
    max_signals_per_day: int
    bos_min_break_atr: float      # BOS detection sensitivity
    zigzag_atr_mult: float        # ZigZag reversal threshold

    # Volatility data
    current_atr: float
    atr_percentile: float         # 0-100
    bb_width: float
    hist_vol_annualised: float


class AdaptiveThresholdEngine:
    """
    Monitors market volatility and adjusts trading parameters to
    maintain consistent risk exposure across different conditions.

    Regime detection uses a composite score from ATR percentile,
    Bollinger Band width, and historical volatility, each weighted
    equally.
    """

    # Default parameters per regime
    REGIME_PARAMS = {
        VolatilityRegime.LOW: {
            "sl_atr_multiplier": 1.2,
            "min_confidence": 75,
            "signal_cooldown_min": 45,
            "max_signals_per_day": 4,
            "bos_min_break_atr": 0.25,
            "zigzag_atr_mult": 1.2,
        },
        VolatilityRegime.NORMAL: {
            "sl_atr_multiplier": 1.5,
            "min_confidence": 80,
            "signal_cooldown_min": 60,
            "max_signals_per_day": 3,
            "bos_min_break_atr": 0.30,
            "zigzag_atr_mult": 1.5,
        },
        VolatilityRegime.HIGH: {
            "sl_atr_multiplier": 2.0,
            "min_confidence": 85,
            "signal_cooldown_min": 90,
            "max_signals_per_day": 2,
            "bos_min_break_atr": 0.40,
            "zigzag_atr_mult": 2.0,
        },
        VolatilityRegime.EXTREME: {
            "sl_atr_multiplier": 2.5,
            "min_confidence": 90,
            "signal_cooldown_min": 120,
            "max_signals_per_day": 1,
            "bos_min_break_atr": 0.50,
            "zigzag_atr_mult": 2.5,
        },
    }

    # Regime thresholds (based on composite volatility score)
    REGIME_THRESHOLDS = {
        VolatilityRegime.LOW: (0.0, 0.25),
        VolatilityRegime.NORMAL: (0.25, 0.55),
        VolatilityRegime.HIGH: (0.55, 0.80),
        VolatilityRegime.EXTREME: (0.80, 1.01),
    }

    def __init__(self, atr_period: int = 14,
                 bb_period: int = 20,
                 hist_vol_window: int = 30,
                 lookback_percentile: int = 100):
        """
        Args:
            atr_period: ATR calculation period.
            bb_period: Bollinger Band SMA period.
            hist_vol_window: Window for historical volatility.
            lookback_percentile: Bars for percentile ranking of ATR.
        """
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.hist_vol_window = hist_vol_window
        self.lookback_percentile = lookback_percentile

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """True Range-based ATR."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)

        return tr.rolling(self.atr_period).mean()

    def _calculate_bb_width(self, df: pd.DataFrame) -> float:
        """
        Bollinger Band width as a percentage of the middle band.
        BBW = (Upper - Lower) / Middle * 100
        """
        sma = df["Close"].rolling(self.bb_period).mean()
        std = df["Close"].rolling(self.bb_period).std()

        upper = sma + 2 * std
        lower = sma - 2 * std

        bbw = ((upper - lower) / sma * 100).iloc[-1]
        return float(bbw) if not np.isnan(bbw) else 0.0

    def _calculate_hist_vol(self, df: pd.DataFrame) -> float:
        """Annualised historical volatility from log returns."""
        log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        if len(log_returns) < self.hist_vol_window:
            return 0.0

        recent = log_returns.iloc[-self.hist_vol_window:]
        return float(recent.std() * np.sqrt(252) * 100)

    def _score_to_regime(self, score: float) -> VolatilityRegime:
        """Map composite score to regime."""
        for regime, (low, high) in self.REGIME_THRESHOLDS.items():
            if low <= score < high:
                return regime
        return VolatilityRegime.NORMAL

    def analyze(self, df: pd.DataFrame) -> AdaptiveParams:
        """
        Analyze current volatility and return adapted parameters.

        Args:
            df: OHLCV DataFrame with at least 100 bars.

        Returns:
            AdaptiveParams with adjusted values for the current regime.
        """
        if len(df) < 50:
            params = self.REGIME_PARAMS[VolatilityRegime.NORMAL]
            return AdaptiveParams(
                regime=VolatilityRegime.NORMAL,
                regime_score=0.5,
                current_atr=0.0,
                atr_percentile=50.0,
                bb_width=0.0,
                hist_vol_annualised=0.0,
                **params,
            )

        # ATR and percentile
        atr = self._calculate_atr(df)
        current_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0
        lookback_atr = atr.iloc[-self.lookback_percentile:]
        atr_pct = float(
            (lookback_atr < current_atr).sum() / len(lookback_atr) * 100
        ) if len(lookback_atr) > 0 else 50.0

        # Bollinger Band width
        bb_width = self._calculate_bb_width(df)

        # Historical volatility
        hist_vol = self._calculate_hist_vol(df)

        # Normalize each component to 0-1 range
        # ATR percentile: already 0-100, divide by 100
        atr_norm = atr_pct / 100.0

        # BB width: typical range 1-10%, normalize using sigmoid-like
        bb_norm = min(1.0, bb_width / 8.0)

        # Hist vol: typical range 10-60%, normalize
        hv_norm = min(1.0, hist_vol / 50.0)

        # Composite score (equal weights)
        composite = (atr_norm + bb_norm + hv_norm) / 3.0

        regime = self._score_to_regime(composite)
        params = self.REGIME_PARAMS[regime]

        return AdaptiveParams(
            regime=regime,
            regime_score=round(composite, 3),
            current_atr=round(current_atr, 6),
            atr_percentile=round(atr_pct, 1),
            bb_width=round(bb_width, 2),
            hist_vol_annualised=round(hist_vol, 2),
            **params,
        )

    def format_summary(self, params: AdaptiveParams) -> str:
        """Human-readable summary of adaptive parameters."""
        lines = [
            f"Volatility Regime: {params.regime.value}",
            f"  Composite score: {params.regime_score:.3f}",
            f"  ATR: {params.current_atr:.4f} (P{params.atr_percentile:.0f})",
            f"  BB Width: {params.bb_width:.2f}%",
            f"  Hist Vol: {params.hist_vol_annualised:.1f}%",
            f"",
            f"Adapted Parameters:",
            f"  SL ATR mult: {params.sl_atr_multiplier}x",
            f"  Min confidence: {params.min_confidence}",
            f"  Signal cooldown: {params.signal_cooldown_min} min",
            f"  Max signals/day: {params.max_signals_per_day}",
            f"  BOS threshold: {params.bos_min_break_atr} ATR",
            f"  ZigZag mult: {params.zigzag_atr_mult}x",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("Adaptive Thresholds Test")
    print("=" * 50)

    np.random.seed(42)

    engine = AdaptiveThresholdEngine()

    # Low volatility
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    df_low = pd.DataFrame({
        "Open": close - 0.1, "High": close + 0.2,
        "Low": close - 0.2, "Close": close,
        "Volume": np.random.randint(1000, 5000, n),
    }, index=dates)

    result = engine.analyze(df_low)
    print(f"\nLow Vol Scenario:")
    print(engine.format_summary(result))

    # High volatility
    close_high = 100 + np.cumsum(np.random.randn(n) * 3.0)
    close_high = np.maximum(close_high, 10)
    df_high = pd.DataFrame({
        "Open": close_high - 1.0, "High": close_high + 2.5,
        "Low": close_high - 2.5, "Close": close_high,
        "Volume": np.random.randint(10000, 100000, n),
    }, index=dates)

    result = engine.analyze(df_high)
    print(f"\nHigh Vol Scenario:")
    print(engine.format_summary(result))

    print("\nAdaptive thresholds test complete.")
