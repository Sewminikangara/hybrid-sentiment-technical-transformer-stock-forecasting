"""
Multi-Timeframe Elliott Wave Correlation

"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimeframeWaveResult:
    """Wave analysis result for a single timeframe."""
    timeframe: str
    current_wave: str
    direction: str
    confidence: int
    fib_score: float
    is_wave3_entry: bool
    wave_summary: str


@dataclass
class MultiTimeframeResult:
    """Combined result across timeframes."""
    symbol: str
    timeframes_analysed: List[str]
    results: Dict[str, TimeframeWaveResult]
    alignment_score: float         # 0.0 to 1.0
    composite_confidence: int      # 0 to 100
    direction_consensus: str       # LONG, SHORT, or MIXED
    recommendation: str
    is_aligned: bool


class MultiTimeframeWaveAnalyzer:
    

    DEFAULT_TIMEFRAMES = ["1d", "4h", "1h"]

    # Timeframe ranking (higher = longer term)
    TF_RANK = {
        "1w": 5, "1d": 4, "4h": 3, "1h": 2, "15m": 1, "5m": 0
    }

    def __init__(self, timeframes: Optional[List[str]] = None,
                 min_alignment: float = 0.6):
        """
        Args:
            timeframes: List of timeframes to analyse (descending).
            min_alignment: Minimum alignment score for a favourable recommendation.
        """
        self.timeframes = timeframes or self.DEFAULT_TIMEFRAMES
        self.min_alignment = min_alignment

    def _run_single_timeframe(self, df: pd.DataFrame,
                              timeframe: str) -> TimeframeWaveResult:
        """
        Run Elliott Wave engine on a single timeframe dataframe.

        Args:
            df: OHLCV data for the given timeframe.
            timeframe: Timeframe label (e.g. '4h').

        Returns:
            TimeframeWaveResult for the given timeframe.
        """
        from tradex.engines.elliott_wave_engine import ElliottWaveEngine

        engine = ElliottWaveEngine()
        result = engine.analyze(df)

        best = result.best_candidate
        if best:
            return TimeframeWaveResult(
                timeframe=timeframe,
                current_wave=best.current_wave.value,
                direction=best.direction,
                confidence=best.confidence,
                fib_score=best.fib_score,
                is_wave3_entry=best.is_wave3_entry,
                wave_summary=result.wave_summary,
            )
        else:
            return TimeframeWaveResult(
                timeframe=timeframe,
                current_wave="NONE",
                direction="NEUTRAL",
                confidence=0,
                fib_score=0.0,
                is_wave3_entry=False,
                wave_summary="No wave pattern detected",
            )

    def _resample_ohlcv(self, df: pd.DataFrame,
                        target_tf: str) -> pd.DataFrame:
        """
        Resample intraday OHLCV data to a coarser timeframe.

        Args:
            df: OHLCV data with DatetimeIndex.
            target_tf: Target timeframe ('4h', '1d', '1w').

        Returns:
            Resampled DataFrame.
        """
        tf_map = {
            "5m": "5min", "15m": "15min", "1h": "1h",
            "4h": "4h", "1d": "1D", "1w": "1W"
        }
        rule = tf_map.get(target_tf, "1D")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
            elif "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)

        resampled = df.resample(rule).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

        return resampled

    def _calculate_alignment(
        self, results: Dict[str, TimeframeWaveResult]
    ) -> Tuple[float, str]:
        """
        Calculate alignment score across timeframes.

        Returns:
            (alignment_score, direction_consensus)
        """
        if not results:
            return 0.0, "NEUTRAL"

        valid = {k: v for k, v in results.items()
                 if v.current_wave != "NONE"}

        if not valid:
            return 0.0, "NEUTRAL"

        # Direction alignment (0.4 weight)
        directions = [r.direction for r in valid.values()]
        bull_count = sum(1 for d in directions if d == "BULLISH")
        bear_count = sum(1 for d in directions if d == "BEARISH")
        total = len(directions)

        if bull_count == total:
            dir_score = 1.0
            consensus = "LONG"
        elif bear_count == total:
            dir_score = 1.0
            consensus = "SHORT"
        elif bull_count > bear_count:
            dir_score = bull_count / total
            consensus = "LONG"
        elif bear_count > bull_count:
            dir_score = bear_count / total
            consensus = "SHORT"
        else:
            dir_score = 0.0
            consensus = "MIXED"

        # Wave position consistency (0.3 weight)
        wave_entries = [r.is_wave3_entry for r in valid.values()]
        entry_agreement = sum(wave_entries) / len(wave_entries) if wave_entries else 0.0

        # Fibonacci score consistency (0.3 weight)
        fib_scores = [r.fib_score for r in valid.values() if r.fib_score > 0]
        if fib_scores:
            fib_mean = np.mean(fib_scores)
            fib_std = np.std(fib_scores)
            # Low variance means good agreement
            fib_agreement = max(0.0, 1.0 - fib_std * 2)
        else:
            fib_agreement = 0.0

        alignment = (
            0.4 * dir_score
            + 0.3 * entry_agreement
            + 0.3 * fib_agreement
        )

        return round(alignment, 3), consensus

    def analyze(self, dataframes: Dict[str, pd.DataFrame],
                symbol: str = "UNKNOWN") -> MultiTimeframeResult:
        
        # If a single dataframe is provided, resample it
        if len(dataframes) == 1:
            base_tf = list(dataframes.keys())[0]
            base_df = dataframes[base_tf]
            for tf in self.timeframes:
                if tf not in dataframes:
                    try:
                        dataframes[tf] = self._resample_ohlcv(base_df, tf)
                    except Exception as e:
                        logger.debug(f"Cannot resample to {tf}: {e}")

        # Run analysis on each timeframe
        results = {}
        for tf in self.timeframes:
            if tf in dataframes and len(dataframes[tf]) >= 30:
                try:
                    results[tf] = self._run_single_timeframe(
                        dataframes[tf], tf
                    )
                except Exception as e:
                    logger.debug(f"Wave analysis failed for {tf}: {e}")

        alignment, consensus = self._calculate_alignment(results)

        # Composite confidence = weighted average of per-TF confidence
        # weighted by timeframe rank
        confidences = []
        weights = []
        for tf, r in results.items():
            rank = self.TF_RANK.get(tf, 1)
            confidences.append(r.confidence)
            weights.append(rank)

        if confidences:
            composite = int(np.average(confidences, weights=weights))
        else:
            composite = 0

        # Boost composite by alignment
        composite = min(100, int(composite * (0.7 + 0.3 * alignment)))

        # Recommendation
        if alignment >= self.min_alignment and composite >= 70:
            recommendation = f"ALIGNED {consensus} - high composite confidence"
        elif alignment >= self.min_alignment:
            recommendation = f"ALIGNED {consensus} - moderate confidence"
        elif composite >= 70:
            recommendation = f"MISALIGNED - single timeframe signal only"
        else:
            recommendation = "NO CLEAR SETUP - wait for alignment"

        return MultiTimeframeResult(
            symbol=symbol,
            timeframes_analysed=list(results.keys()),
            results=results,
            alignment_score=alignment,
            composite_confidence=composite,
            direction_consensus=consensus,
            recommendation=recommendation,
            is_aligned=bool(alignment >= self.min_alignment),
        )


if __name__ == "__main__":
    print("Multi-Timeframe Elliott Wave Correlation Test")
    print("=" * 50)

    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open": close - np.random.rand(n) * 0.3,
        "High": close + np.random.rand(n) * 0.5,
        "Low": close - np.random.rand(n) * 0.5,
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    }, index=dates)

    analyzer = MultiTimeframeWaveAnalyzer()
    result = analyzer.analyze({"1h": df}, symbol="TEST")

    print(f"  Symbol: {result.symbol}")
    print(f"  Timeframes analysed: {result.timeframes_analysed}")
    print(f"  Alignment score: {result.alignment_score:.3f}")
    print(f"  Composite confidence: {result.composite_confidence}/100")
    print(f"  Direction consensus: {result.direction_consensus}")
    print(f"  Is aligned: {result.is_aligned}")
    print(f"  Recommendation: {result.recommendation}")

    for tf, r in result.results.items():
        print(f"\n  [{tf}] Wave: {r.current_wave}, "
              f"Dir: {r.direction}, Conf: {r.confidence}, "
              f"Fib: {r.fib_score:.3f}")

    print("\nMulti-Timeframe Wave test complete.")
