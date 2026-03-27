"""
Volume Profile and Order Flow Analysis

"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    """Volume profile analysis for a given period."""
    poc_price: float           # Point of Control (highest volume price)
    value_area_high: float     # Upper bound of 70% value area
    value_area_low: float      # Lower bound of 70% value area
    total_volume: float
    price_bins: int
    price_in_value_area: bool  # Whether current price is in value area


@dataclass
class VolumeAnalysisResult:
    """Complete volume analysis output."""
    symbol: str
    obv_trend: str              # BULLISH, BEARISH, or NEUTRAL
    obv_divergence: str         # BULL_DIV, BEAR_DIV, or NONE
    vwap: float
    price_vs_vwap: str          # ABOVE or BELOW
    ad_line_trend: str          # ACCUMULATION or DISTRIBUTION
    volume_climax: bool         # Whether recent bar is volume climax
    volume_profile: VolumeProfileResult
    confirmation_score: float   # 0.0 to 1.0 (how well volume confirms trend)
    summary: str


class VolumeAnalyzer:
    """
    Multi-indicator volume analysis engine. Produces a composite
    confirmation score that measures how strongly volume supports
    the current price trend.
    """

    def __init__(self, profile_bins: int = 50,
                 value_area_pct: float = 0.70,
                 obv_lookback: int = 20,
                 climax_threshold: float = 2.5):
        """
        Args:
            profile_bins: Number of price bins for volume profile.
            value_area_pct: Fraction of volume that defines value area (0.70 = 70%).
            obv_lookback: Lookback for OBV trend calculation.
            climax_threshold: Volume must exceed mean by this factor for climax.
        """
        self.profile_bins = profile_bins
        self.value_area_pct = value_area_pct
        self.obv_lookback = obv_lookback
        self.climax_threshold = climax_threshold

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.

        OBV = cumulative sum of volume on up-days minus volume on down-days.
        """
        close_diff = df["Close"].diff()
        obv = pd.Series(0.0, index=df.index, dtype=float)

        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + df["Volume"].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - df["Volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def detect_obv_divergence(self, df: pd.DataFrame,
                               obv: pd.Series,
                               lookback: int = 20) -> str:
        """
        Detect divergence between price and OBV.

        Bullish divergence: price makes lower low, OBV makes higher low.
        Bearish divergence: price makes higher high, OBV makes lower high.
        """
        if len(df) < lookback * 2:
            return "NONE"

        recent = slice(-lookback, None)
        prior = slice(-lookback * 2, -lookback)

        price_recent_low = df["Low"].iloc[recent].min()
        price_prior_low = df["Low"].iloc[prior].min()
        obv_recent_low = obv.iloc[recent].min()
        obv_prior_low = obv.iloc[prior].min()

        price_recent_high = df["High"].iloc[recent].max()
        price_prior_high = df["High"].iloc[prior].max()
        obv_recent_high = obv.iloc[recent].max()
        obv_prior_high = obv.iloc[prior].max()

        # Bullish divergence
        if price_recent_low < price_prior_low and obv_recent_low > obv_prior_low:
            return "BULL_DIV"

        # Bearish divergence
        if price_recent_high > price_prior_high and obv_recent_high < obv_prior_high:
            return "BEAR_DIV"

        return "NONE"

    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        Calculate Volume-Weighted Average Price for the session.

        VWAP = sum(Typical_Price * Volume) / sum(Volume)
        """
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (typical * df["Volume"]).sum() / df["Volume"].sum()
        return float(vwap)

    def calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        AD = cumsum(CLV * Volume)
        """
        high_low = df["High"] - df["Low"]
        high_low = high_low.replace(0, 1e-10)

        clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low
        ad = (clv * df["Volume"]).cumsum()
        return ad

    def calculate_volume_profile(self, df: pd.DataFrame,
                                  lookback: int = 50) -> VolumeProfileResult:
        """
        Build volume profile (volume at price histogram).

        Returns POC, VAH, VAL for the lookback period.
        """
        window = df.tail(lookback)
        prices = window["Close"].values
        volumes = window["Volume"].values.astype(float)

        price_min = prices.min()
        price_max = prices.max()

        if price_max == price_min:
            return VolumeProfileResult(
                poc_price=price_min,
                value_area_high=price_max,
                value_area_low=price_min,
                total_volume=volumes.sum(),
                price_bins=1,
                price_in_value_area=True,
            )

        bins = np.linspace(price_min, price_max, self.profile_bins + 1)
        bin_volumes = np.zeros(self.profile_bins)

        for p, v in zip(prices, volumes):
            idx = min(int((p - price_min) / (price_max - price_min) * self.profile_bins),
                      self.profile_bins - 1)
            bin_volumes[idx] += v

        # POC = bin with highest volume
        poc_idx = np.argmax(bin_volumes)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Value area = 70% of total volume centred on POC
        total_vol = bin_volumes.sum()
        target_vol = total_vol * self.value_area_pct

        va_vol = bin_volumes[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while va_vol < target_vol:
            add_low = bin_volumes[low_idx - 1] if low_idx > 0 else 0
            add_high = bin_volumes[high_idx + 1] if high_idx < self.profile_bins - 1 else 0

            if add_high >= add_low and high_idx < self.profile_bins - 1:
                high_idx += 1
                va_vol += add_high
            elif low_idx > 0:
                low_idx -= 1
                va_vol += add_low
            else:
                break

        vah = bins[high_idx + 1] if high_idx < self.profile_bins else price_max
        val = bins[low_idx]
        current_price = prices[-1]

        return VolumeProfileResult(
            poc_price=poc_price,
            value_area_high=float(vah),
            value_area_low=float(val),
            total_volume=float(total_vol),
            price_bins=self.profile_bins,
            price_in_value_area=val <= current_price <= vah,
        )

    def detect_volume_climax(self, df: pd.DataFrame,
                              lookback: int = 20) -> bool:
        """
        Detect if the latest bar is a volume climax (extreme volume
        spike that often signals exhaustion or capitulation).
        """
        if len(df) < lookback:
            return False

        recent_vol = df["Volume"].iloc[-1]
        mean_vol = df["Volume"].iloc[-lookback:].mean()

        return bool(recent_vol > mean_vol * self.climax_threshold)

    def analyze(self, df: pd.DataFrame,
                symbol: str = "UNKNOWN") -> VolumeAnalysisResult:
        """
        Run full volume analysis.

        Args:
            df: OHLCV DataFrame.
            symbol: Asset symbol.

        Returns:
            VolumeAnalysisResult with all indicators and confirmation score.
        """
        if len(df) < 30:
            return VolumeAnalysisResult(
                symbol=symbol, obv_trend="NEUTRAL", obv_divergence="NONE",
                vwap=0.0, price_vs_vwap="NEUTRAL", ad_line_trend="NEUTRAL",
                volume_climax=False,
                volume_profile=VolumeProfileResult(0, 0, 0, 0, 0, False),
                confirmation_score=0.0, summary="Insufficient data",
            )

        # OBV
        obv = self.calculate_obv(df)
        obv_slope = obv.iloc[-self.obv_lookback:].diff().mean()
        obv_trend = "BULLISH" if obv_slope > 0 else "BEARISH" if obv_slope < 0 else "NEUTRAL"
        obv_div = self.detect_obv_divergence(df, obv)

        # VWAP
        vwap = self.calculate_vwap(df.tail(50))
        current_price = df["Close"].iloc[-1]
        price_vs_vwap = "ABOVE" if current_price > vwap else "BELOW"

        # A/D line
        ad = self.calculate_ad_line(df)
        ad_slope = ad.iloc[-20:].diff().mean()
        ad_trend = "ACCUMULATION" if ad_slope > 0 else "DISTRIBUTION"

        # Volume profile
        profile = self.calculate_volume_profile(df)

        # Volume climax
        climax = self.detect_volume_climax(df)

        # Confirmation score
        # Measures how well volume supports a bullish trend
        score = 0.0
        checks = 0

        # OBV confirms trend (+0.25)
        price_trend = "BULLISH" if df["Close"].iloc[-1] > df["Close"].iloc[-20] else "BEARISH"
        if obv_trend == price_trend:
            score += 0.25
        checks += 1

        # No bearish divergence (+0.25)
        if obv_div != "BEAR_DIV":
            score += 0.15
        if obv_div == "BULL_DIV":
            score += 0.10
        checks += 1

        # Price above VWAP (+0.20)
        if price_vs_vwap == "ABOVE" and price_trend == "BULLISH":
            score += 0.20
        elif price_vs_vwap == "BELOW" and price_trend == "BEARISH":
            score += 0.20
        checks += 1

        # Accumulation confirms bullish (+0.15)
        if (ad_trend == "ACCUMULATION" and price_trend == "BULLISH") or \
           (ad_trend == "DISTRIBUTION" and price_trend == "BEARISH"):
            score += 0.15
        checks += 1

        # Price in value area (+0.15)
        if profile.price_in_value_area:
            score += 0.15
        checks += 1

        # Summary
        parts = []
        parts.append(f"OBV: {obv_trend}")
        if obv_div != "NONE":
            parts.append(f"Divergence: {obv_div}")
        parts.append(f"VWAP: {price_vs_vwap}")
        parts.append(f"A/D: {ad_trend}")
        parts.append(f"Confirmation: {score:.0%}")
        summary = " | ".join(parts)

        return VolumeAnalysisResult(
            symbol=symbol,
            obv_trend=obv_trend,
            obv_divergence=obv_div,
            vwap=round(vwap, 4),
            price_vs_vwap=price_vs_vwap,
            ad_line_trend=ad_trend,
            volume_climax=climax,
            volume_profile=profile,
            confirmation_score=round(score, 3),
            summary=summary,
        )


if __name__ == "__main__":
    logger.info("Volume Profile and Order Flow Analysis Test")
    logger.info("=")

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 100 + np.cumsum(np.random.randn(n) * 1.0)
    close = np.maximum(close, 20)
    df = pd.DataFrame({
        "Open": close - np.random.rand(n) * 0.3,
        "High": close + np.random.rand(n) * 0.8,
        "Low": close - np.random.rand(n) * 0.8,
        "Close": close,
        "Volume": np.random.randint(5000, 50000, n),
    }, index=dates)

    analyzer = VolumeAnalyzer()
    result = analyzer.analyze(df, symbol="TEST")

    logger.info("  Symbol: {result.symbol}")
    logger.info("  OBV trend: {result.obv_trend}")
    logger.info("  OBV divergence: {result.obv_divergence}")
    logger.info("  VWAP: {result.vwap:.2f}")
    logger.info("  Price vs VWAP: {result.price_vs_vwap}")
    logger.info("  A/D trend: {result.ad_line_trend}")
    logger.info("  Volume climax: {result.volume_climax}")
    logger.info("  Confirmation score: {result.confirmation_score:.1%}")
    logger.info("\n  Volume Profile:")
    logger.info("    POC: {result.volume_profile.poc_price:.2f}")
    logger.info("    VAH: {result.volume_profile.value_area_high:.2f}")
    logger.info("    VAL: {result.volume_profile.value_area_low:.2f}")
    logger.info("    Price in VA: {result.volume_profile.price_in_value_area}")
    logger.info("\n  Summary: {result.summary}")

    logger.info("\nVolume analysis test complete.")
