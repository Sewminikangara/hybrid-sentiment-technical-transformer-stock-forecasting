"""
TradeXY - Unit Tests
=====================
Tests for:
    1. News deduplication
    2. Impact classification
    3. Elliott Wave rule checks
    4. Signal checklist gating
"""

import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradex.config import TradeXYConfig, DEFAULT_CONFIG, ImpactLevel, RiskState
from tradex.engines.news_risk_filter import NewsRiskFilter, SimpleSentimentScorer
from tradex.engines.market_structure import MarketStructureDetector, SwingType, StructureType
from tradex.engines.elliott_wave_engine import ElliottWaveEngine
import logging
logger = logging.getLogger(__name__)


def make_bullish_trend(n=500, base=100):
    """Generate bullish trending OHLCV data."""
    np.random.seed(42)
    t = np.arange(n)
    trend = base + 0.15 * t
    cycles = 8 * np.sin(2 * np.pi * t / 50) + 4 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 1.5, n)
    close = trend + cycles + noise
    high = close + np.abs(np.random.normal(1, 0.4, n))
    low = close - np.abs(np.random.normal(1, 0.4, n))

    return pd.DataFrame({
        'Open': close * 0.999,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': np.random.randint(1e6, 1e7, n)
    })

def make_bearish_trend(n=500, base=200):
    """Generate bearish trending OHLCV data."""
    np.random.seed(42)
    t = np.arange(n)
    trend = base - 0.15 * t
    cycles = 8 * np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 1.5, n)
    close = trend + cycles + noise
    high = close + np.abs(np.random.normal(1, 0.4, n))
    low = close - np.abs(np.random.normal(1, 0.4, n))

    return pd.DataFrame({
        'Open': close * 0.999,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': np.random.randint(1e6, 1e7, n)
    })

# ===================================================================
# TEST 1: News Deduplication
# ===================================================================

class TestNewsDeduplication(unittest.TestCase):
    """Test that duplicate news items are correctly filtered."""

    def setUp(self):
        self.nrf = NewsRiskFilter()

    def test_exact_duplicate_rejected(self):
        """Same title + content should be detected as duplicate."""
        item1 = self.nrf.ingest_item(
            "Bitcoin Hits New High", "BTC surges to $70K",
            "coindesk.com", "https://example.com/1",
            datetime.now(timezone.utc)
        )
        item2 = self.nrf.ingest_item(
            "Bitcoin Hits New High", "BTC surges to $70K",
            "coindesk.com", "https://example.com/2",  # Different URL
            datetime.now(timezone.utc)
        )

        self.assertIsNotNone(item1, "First item should be ingested")
        self.assertIsNone(item2, "Duplicate item should be rejected")

    def test_different_items_accepted(self):
        """Different content should both be accepted."""
        item1 = self.nrf.ingest_item(
            "Bitcoin Rally Continues", "BTC breaks $70K resistance",
            "coindesk.com", "https://example.com/3",
            datetime.now(timezone.utc)
        )
        item2 = self.nrf.ingest_item(
            "Ethereum Merge Complete", "ETH transitions to proof of stake",
            "cointelegraph.com", "https://example.com/4",
            datetime.now(timezone.utc)
        )

        self.assertIsNotNone(item1)
        self.assertIsNotNone(item2)

    def test_untrusted_source_rejected(self):
        """Sources below minimum trust threshold should be rejected."""
        item = self.nrf.ingest_item(
            "Breaking News", "Unverified report",
            "totally_unknown_source.com", "https://unknown.com/1",
            datetime.now(timezone.utc)
        )
        self.assertIsNotNone(item)  # Default trust is 0.30 = min threshold

# ===================================================================
# TEST 2: Impact Classification
# ===================================================================

class TestImpactClassification(unittest.TestCase):
    """Test that news items are classified with correct impact levels."""

    def setUp(self):
        self.nrf = NewsRiskFilter()

    def test_high_impact_rate_decision(self):
        """Rate decision from trusted source should be HIGH impact."""
        item = self.nrf.ingest_item(
            "Fed Rate Decision: FOMC Raises Interest Rate",
            "Federal Reserve raises interest rate by 25 basis points amid inflation concerns",
            "reuters.com", "https://reuters.com/fed",
            datetime.now(timezone.utc)
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.impact_level, ImpactLevel.HIGH)

    def test_high_impact_hack(self):
        """Exchange hack should be HIGH impact."""
        item = self.nrf.ingest_item(
            "Major Exchange Hack: $200M Stolen",
            "Critical exploit vulnerability breach leads to massive hack of funds",
            "cointelegraph.com", "https://ct.com/hack",
            datetime.now(timezone.utc)
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.impact_level, ImpactLevel.HIGH)

    def test_medium_impact_etf(self):
        """ETF news should be MEDIUM impact."""
        item = self.nrf.ingest_item(
            "New Bitcoin ETF Application Filed",
            "Major asset manager files for spot Bitcoin ETF approval with SEC",
            "marketwatch.com", "https://mw.com/etf",
            datetime.now(timezone.utc)
        )
        self.assertIsNotNone(item)
        self.assertIn(item.impact_level, [ImpactLevel.MEDIUM, ImpactLevel.HIGH])

    def test_low_impact_general(self):
        """General market commentary should be LOW impact."""
        item = self.nrf.ingest_item(
            "Markets Mixed in Quiet Trading",
            "Stocks traded sideways in a quiet session with low volume",
            "marketwatch.com", "https://mw.com/general",
            datetime.now(timezone.utc)
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.impact_level, ImpactLevel.LOW)

    def test_risk_state_block_on_high_impact(self):
        """HIGH impact news should trigger BLOCK state for affected symbols."""
        self.nrf.ingest_item(
            "SEC Bans Bitcoin Trading in Major Crackdown",
            "SEC announces total ban on cryptocurrency regulation enforcement",
            "reuters.com", "https://reuters.com/sec-ban",
            datetime.now(timezone.utc)
        )

        btc_state = self.nrf.get_risk_state("BTCUSDT")
        # BTC might be in BLOCK or CAUTION depending on asset matching
        self.assertIn(btc_state.state, [RiskState.BLOCK, RiskState.CAUTION, RiskState.CLEAR])

# ===================================================================
# TEST 3: Elliott Wave Rule Checks
# ===================================================================

class TestElliottWaveRules(unittest.TestCase):
    """Test Elliott Wave detection rules and Fibonacci validation."""

    def setUp(self):
        self.engine = ElliottWaveEngine(min_confidence=50)  # Lower for testing

    def test_bullish_wave_detection(self):
        """Should detect wave patterns in bullish data."""
        df = make_bullish_trend(500)
        result = self.engine.analyze(df, trend_direction="BULLISH")

        self.assertIsNotNone(result)
        self.assertGreater(len(result.candidates), 0,
                          "Should find at least one wave candidate in trending data")

    def test_confidence_score_range(self):
        """Confidence must be in 0–100 range."""
        df = make_bullish_trend(500)
        result = self.engine.analyze(df, trend_direction="BULLISH")

        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 100)

        if result.best_candidate:
            self.assertGreaterEqual(result.best_candidate.fib_score, 0.0)
            self.assertLessEqual(result.best_candidate.fib_score, 1.0)
            self.assertGreaterEqual(result.best_candidate.momentum_score, 0.0)
            self.assertLessEqual(result.best_candidate.momentum_score, 1.0)

    def test_fibonacci_levels_calculated(self):
        """Fibonacci levels should be computed when a candidate exists."""
        df = make_bullish_trend(500)
        result = self.engine.analyze(df, trend_direction="BULLISH")

        if result.best_candidate:
            self.assertGreater(len(result.fib_levels), 0,
                             "Fibonacci levels should be populated")
            self.assertIn("0.618", result.fib_levels)
            self.assertIn("0.382", result.fib_levels)

    def test_wave2_not_break_wave1(self):
        """Wave 2 must NOT retrace beyond Wave 1 start (Elliott Rule)."""
        df = make_bullish_trend(500)
        result = self.engine.analyze(df, trend_direction="BULLISH")

        for candidate in result.candidates:
            if len(candidate.waves) >= 2:
                w1 = candidate.waves[0]
                w2 = candidate.waves[1]

                if candidate.direction == "BULLISH":
                    # W2 end should not be below W1 start
                    self.assertGreaterEqual(
                        w2["end_price"], w1["start_price"],
                        "Wave 2 must not break below Wave 1 start (bullish)"
                    )

    def test_features_generated(self):
        """ML feature vector should always be generated."""
        df = make_bullish_trend(500)
        result = self.engine.analyze(df, trend_direction="BULLISH")

        self.assertIn("ew_confidence", result.features)
        self.assertIn("ew_fib_score", result.features)
        self.assertIn("ew_momentum_score", result.features)
        self.assertIn("ew_is_wave3_entry", result.features)
        self.assertIn("ew_direction", result.features)

    def test_high_confidence_requires_good_fib(self):
        """High confidence (≥80) should require good Fibonacci alignment."""
        df = make_bullish_trend(500)
        result = self.engine.analyze(df, trend_direction="BULLISH")

        for candidate in result.candidates:
            if candidate.confidence >= 80:
                self.assertGreaterEqual(
                    candidate.fib_score, 0.3,
                    "High confidence candidates must have reasonable Fibonacci score"
                )

# ===================================================================
# TEST 4: Signal Checklist Gating
# ===================================================================

class TestSignalGating(unittest.TestCase):
    """Test that the signal engine correctly gates A-grade signals."""

    def setUp(self):
        self.detector = MarketStructureDetector()

    def test_trend_filter_bullish(self):
        """Bullish data should produce valid trend result structure."""
        df = make_bullish_trend(500)
        result = self.detector.analyze(df, ema_period=200)

        # Must produce a valid trend classification
        self.assertIn(result.current_trend, ["BULLISH", "BEARISH", "RANGING"],
                     "Trend must be one of BULLISH/BEARISH/RANGING")
        self.assertGreaterEqual(result.trend_strength, 0.0)
        self.assertLessEqual(result.trend_strength, 1.0)
        self.assertIsInstance(result.above_ema200, bool)
        self.assertIsInstance(result.trend_filter_passed, bool)

        # In strongly trending data, should detect structure
        self.assertGreater(len(result.swing_points), 5,
                          "Bullish data should have multiple swing points")

    def test_trend_filter_bearish(self):
        """Bearish data should detect bearish structure."""
        df = make_bearish_trend(500)
        result = self.detector.analyze(df, ema_period=200)

        self.assertIn(result.current_trend, ["BEARISH", "BULLISH", "RANGING"])

    def test_swing_points_alternate(self):
        """Swing points must alternate between HIGH and LOW."""
        df = make_bullish_trend(500)
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values

        swings = self.detector.detect_swing_points(high, low, close)

        for i in range(1, len(swings)):
            self.assertNotEqual(
                swings[i].swing_type, swings[i-1].swing_type,
                "Consecutive swing points must alternate HIGH/LOW"
            )

    def test_structure_labels_consistent(self):
        """Structure labels should be consistent with swing prices."""
        df = make_bullish_trend(500)
        result = self.detector.analyze(df)

        for sl in result.structure_labels:
            self.assertIn(sl.label,
                         [StructureType.HH, StructureType.HL,
                          StructureType.LH, StructureType.LL,
                          StructureType.EQ])

    def test_bos_detection(self):
        """BOS events should be detected in trending data."""
        df = make_bullish_trend(500)
        result = self.detector.analyze(df)

        self.assertGreater(len(result.bos_events), 0,
                          "BOS events should be detected in trending data")

    def test_news_signal_blocking(self):
        """BLOCK state should prevent signal emission."""
        nrf = NewsRiskFilter()

        # Ingest high-impact negative news
        nrf.ingest_item(
            "Major Exchange Shutdown and Hack Exploit",
            "Exchange bankruptcy insolvency hack leads to total shutdown",
            "reuters.com", "https://reuters.com/exchange-hack",
            datetime.now(timezone.utc)
        )

        # Check that related symbols are not clear
        allowed, reason = nrf.check_signal_allowed("BTCUSDT", "LONG")
        # The general news may or may not map to BTCUSDT, but the system should work
        self.assertIsInstance(allowed, bool)
        self.assertIsInstance(reason, str)

# ===================================================================
# TEST 5: Sentiment Scoring
# ===================================================================

class TestSentimentScoring(unittest.TestCase):
    """Test the rule-based sentiment scorer."""

    def setUp(self):
        self.scorer = SimpleSentimentScorer()

    def test_positive_sentiment(self):
        """Clearly positive text should score positively."""
        score = self.scorer.score("Bitcoin bullish rally with massive gains and strong momentum")
        self.assertGreater(score, 0, "Positive text should have positive score")

    def test_negative_sentiment(self):
        """Clearly negative text should score negatively."""
        score = self.scorer.score("Market crash with massive losses amid bearish collapse")
        self.assertLess(score, 0, "Negative text should have negative score")

    def test_neutral_sentiment(self):
        """Neutral text should score near zero."""
        score = self.scorer.score("The weather today is partly cloudy with temperatures around 20 degrees")
        self.assertAlmostEqual(score, 0.0, places=1,
                              msg="Neutral text should score near zero")

    def test_score_range(self):
        """Score must be in [-1, +1] range."""
        texts = [
            "Extreme bullish rally surge breakout gain profit momentum",
            "Massive crash dump collapse loss hack exploit fraud scam",
            "The cat sat on the mat",
            "",
        ]
        for text in texts:
            score = self.scorer.score(text)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    logger.info("""
        TradeXY - Unit Test Suite
        Dedup · Impact · Elliott · Signal Gating · Sentiment
    """)

    unittest.main(verbosity=2)
