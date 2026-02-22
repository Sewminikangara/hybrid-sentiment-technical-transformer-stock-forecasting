"""
TradeXY Unit Tests

Additional tests covering 
    FinBERT sentiment integration (hybrid scorer)
    Multi-timeframe Elliott Wave correlation
    Backtesting framework
    Volume profile / order flow analysis
    Adaptive thresholds (volatility-based)
    Economic calendar integration
    Position sizing (Kelly Criterion)
    Signal performance tracking
    Alert notification system
    Cross-module integration
"""

import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Helper: Generate Test OHLCV Data ---

def make_ohlcv(n=200, base=100, trend=0.1, volatility=1.0, seed=42):
    """Generate OHLCV data with configurable trend and volatility."""
    np.random.seed(seed)
    t = np.arange(n)
    close = base + trend * t + np.cumsum(np.random.randn(n) * volatility)
    close = np.maximum(close, 10)
    high = close + np.abs(np.random.normal(0.5, 0.3, n))
    low = close - np.abs(np.random.normal(0.5, 0.3, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(5000, 50000, n),
    }, index=dates)


# ===================================================================
# TEST 1: FinBERT Sentiment Integration
# ===================================================================

class TestFinBERTIntegration(unittest.TestCase):
    """Test FinBERT and hybrid sentiment scoring."""

    def test_finbert_analyzer_init(self):
        """FinBERT analyser should initialise without error."""
        from tradex.engines.finbert_sentiment import FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()
        # is_available depends on whether transformers is installed
        self.assertIsInstance(analyzer.is_available, bool)

    def test_finbert_score_range(self):
        """Score should be in [-1, +1] regardless of availability."""
        from tradex.engines.finbert_sentiment import FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()
        score = analyzer.score("Bitcoin surges to new all-time high")
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_finbert_empty_text(self):
        """Empty text should return 0.0."""
        from tradex.engines.finbert_sentiment import FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()
        self.assertEqual(analyzer.score(""), 0.0)
        self.assertEqual(analyzer.score(None), 0.0)

    def test_finbert_analyze_returns_dict(self):
        """analyze() should return dict with label, confidence, score."""
        from tradex.engines.finbert_sentiment import FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()
        result = analyzer.analyze("Market rally continues")
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("score", result)
        self.assertIn(result["label"], ["positive", "negative", "neutral"])

    def test_hybrid_scorer_init(self):
        """Hybrid scorer should combine FinBERT and rule-based."""
        from tradex.engines.finbert_sentiment import HybridSentimentScorer
        scorer = HybridSentimentScorer(finbert_weight=0.7)
        score = scorer.score("Major crash in cryptocurrency markets")
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_hybrid_detailed_breakdown(self):
        """score_detailed should return complete breakdown."""
        from tradex.engines.finbert_sentiment import HybridSentimentScorer
        scorer = HybridSentimentScorer()
        detail = scorer.score_detailed("Bitcoin bullish momentum surge")
        self.assertIn("rule_score", detail)
        self.assertIn("finbert_score", detail)
        self.assertIn("hybrid_score", detail)
        self.assertIn("finbert_available", detail)
        self.assertIsInstance(detail["finbert_available"], bool)

    def test_batch_scoring(self):
        """Batch scoring should return one score per text."""
        from tradex.engines.finbert_sentiment import FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer()
        texts = ["good earnings", "market crash", "neutral day"]
        scores = analyzer.score_batch(texts)
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)


# ===================================================================
# TEST 2: Multi-Timeframe Elliott Wave
# ===================================================================

class TestMultiTimeframeWave(unittest.TestCase):
    """Test multi-timeframe wave correlation."""

    def test_analyzer_init(self):
        """Analyser initialises with default timeframes."""
        from tradex.engines.multi_timeframe_wave import MultiTimeframeWaveAnalyzer
        analyzer = MultiTimeframeWaveAnalyzer()
        self.assertEqual(len(analyzer.timeframes), 3)

    def test_single_dataframe_resampling(self):
        """When given a single DF, it should attempt to resample."""
        from tradex.engines.multi_timeframe_wave import MultiTimeframeWaveAnalyzer
        analyzer = MultiTimeframeWaveAnalyzer(timeframes=["1d", "4h"])
        df = make_ohlcv(500, base=100, trend=0.1)
        result = analyzer.analyze({"1d": df}, symbol="TEST")
        self.assertEqual(result.symbol, "TEST")
        self.assertGreaterEqual(len(result.timeframes_analysed), 1)

    def test_alignment_score_range(self):
        """Alignment score should be 0.0 to 1.0."""
        from tradex.engines.multi_timeframe_wave import MultiTimeframeWaveAnalyzer
        analyzer = MultiTimeframeWaveAnalyzer()
        df = make_ohlcv(500)
        result = analyzer.analyze({"1d": df}, symbol="TEST")
        self.assertGreaterEqual(result.alignment_score, 0.0)
        self.assertLessEqual(result.alignment_score, 1.0)

    def test_composite_confidence_range(self):
        """Composite confidence should be 0 to 100."""
        from tradex.engines.multi_timeframe_wave import MultiTimeframeWaveAnalyzer
        analyzer = MultiTimeframeWaveAnalyzer()
        df = make_ohlcv(500)
        result = analyzer.analyze({"1d": df})
        self.assertGreaterEqual(result.composite_confidence, 0)
        self.assertLessEqual(result.composite_confidence, 100)

    def test_direction_consensus_values(self):
        """Direction consensus should be LONG, SHORT, or MIXED."""
        from tradex.engines.multi_timeframe_wave import MultiTimeframeWaveAnalyzer
        analyzer = MultiTimeframeWaveAnalyzer()
        df = make_ohlcv(500)
        result = analyzer.analyze({"1d": df})
        self.assertIn(result.direction_consensus,
                      ["LONG", "SHORT", "MIXED", "NEUTRAL"])

    def test_is_aligned_flag(self):
        """is_aligned should be boolean."""
        from tradex.engines.multi_timeframe_wave import MultiTimeframeWaveAnalyzer
        analyzer = MultiTimeframeWaveAnalyzer()
        df = make_ohlcv(500)
        result = analyzer.analyze({"1d": df})
        self.assertIsInstance(result.is_aligned, bool)


# ===================================================================
# TEST 3: Backtesting Framework
# ===================================================================

class TestBacktester(unittest.TestCase):
    """Test backtesting engine."""

    def test_backtester_init(self):
        """Should initialise with configurable parameters."""
        from tradex.engines.backtester import SignalBacktester
        bt = SignalBacktester(initial_capital=50000, risk_per_trade_pct=1.0)
        self.assertEqual(bt.initial_capital, 50000)
        self.assertEqual(bt.risk_per_trade_pct, 1.0)

    def test_backtest_result_structure(self):
        """Result should have all required metrics."""
        from tradex.engines.backtester import BacktestResult
        # Verify dataclass fields exist
        fields = BacktestResult.__dataclass_fields__
        required = [
            "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
            "win_rate_pct", "profit_factor", "equity_curve",
        ]
        for f in required:
            self.assertIn(f, fields,
                         f"BacktestResult must have '{f}' field")

    def test_summary_format(self):
        """Summary should produce readable text."""
        from tradex.engines.backtester import SignalBacktester, BacktestResult
        bt = SignalBacktester()
        result = BacktestResult(
            symbol="TEST", period_start=datetime.now(),
            period_end=datetime.now(), total_bars=100,
            total_signals=5, total_trades=4,
            trades=[], total_return_pct=10.5,
            annualised_return_pct=21.0, sharpe_ratio=1.5,
            sortino_ratio=2.0, max_drawdown_pct=-5.0,
            win_rate_pct=60.0, profit_factor=1.8,
            avg_r_multiple=1.2, avg_hold_bars=15.0,
            expectancy=0.72, equity_curve=[10000, 11050],
            drawdown_curve=[0.0, 0.0],
            buy_hold_return_pct=8.0,
        )
        summary = bt.summary(result)
        self.assertIn("TEST", summary)
        self.assertIn("10.50%", summary)
        self.assertIn("Sharpe", summary)


# ===================================================================
# TEST 4: Volume Profile / Order Flow
# ===================================================================

class TestVolumeAnalyzer(unittest.TestCase):
    """Test volume-based analysis indicators."""

    def setUp(self):
        from tradex.engines.volume_analyzer import VolumeAnalyzer
        self.analyzer = VolumeAnalyzer()
        self.df = make_ohlcv(200)

    def test_obv_calculation(self):
        """OBV should return a series of same length as input."""
        obv = self.analyzer.calculate_obv(self.df)
        self.assertEqual(len(obv), len(self.df))
        self.assertEqual(obv.iloc[0], 0.0)

    def test_vwap_positive(self):
        """VWAP should be a positive price level."""
        vwap = self.analyzer.calculate_vwap(self.df.tail(50))
        self.assertGreater(vwap, 0)

    def test_ad_line_length(self):
        """A/D line should match input length."""
        ad = self.analyzer.calculate_ad_line(self.df)
        self.assertEqual(len(ad), len(self.df))

    def test_volume_profile_poc(self):
        """POC should be within the price range."""
        profile = self.analyzer.calculate_volume_profile(self.df)
        self.assertGreaterEqual(profile.poc_price, self.df["Low"].min())
        self.assertLessEqual(profile.poc_price, self.df["High"].max())

    def test_value_area_bounds(self):
        """Value area high should exceed value area low."""
        profile = self.analyzer.calculate_volume_profile(self.df)
        self.assertGreaterEqual(profile.value_area_high, profile.value_area_low)

    def test_climax_detection(self):
        """Volume climax should return boolean."""
        result = self.analyzer.detect_volume_climax(self.df)
        self.assertIsInstance(result, bool)

    def test_full_analysis(self):
        """Full analysis should produce valid result."""
        result = self.analyzer.analyze(self.df, symbol="TEST")
        self.assertEqual(result.symbol, "TEST")
        self.assertIn(result.obv_trend, ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertIn(result.obv_divergence, ["BULL_DIV", "BEAR_DIV", "NONE"])
        self.assertIn(result.price_vs_vwap, ["ABOVE", "BELOW"])
        self.assertIn(result.ad_line_trend, ["ACCUMULATION", "DISTRIBUTION"])
        self.assertGreaterEqual(result.confirmation_score, 0.0)
        self.assertLessEqual(result.confirmation_score, 1.0)

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        tiny_df = self.df.head(5)
        result = self.analyzer.analyze(tiny_df)
        self.assertEqual(result.summary, "Insufficient data")


# ===================================================================
# TEST 5: Adaptive Thresholds
# ===================================================================

class TestAdaptiveThresholds(unittest.TestCase):
    """Test volatility-regime detection and parameter adaptation."""

    def setUp(self):
        from tradex.engines.adaptive_thresholds import AdaptiveThresholdEngine
        self.engine = AdaptiveThresholdEngine()

    def test_regime_detection(self):
        """Should detect a volatility regime."""
        from tradex.engines.adaptive_thresholds import VolatilityRegime
        df = make_ohlcv(200)
        result = self.engine.analyze(df)
        self.assertIn(result.regime, list(VolatilityRegime))

    def test_regime_score_range(self):
        """Regime score should be 0 to 1."""
        df = make_ohlcv(200)
        result = self.engine.analyze(df)
        self.assertGreaterEqual(result.regime_score, 0.0)
        self.assertLessEqual(result.regime_score, 1.0)

    def test_low_vol_narrower_stops(self):
        """Low volatility should have smaller SL multiplier than high."""
        from tradex.engines.adaptive_thresholds import AdaptiveThresholdEngine
        params = AdaptiveThresholdEngine.REGIME_PARAMS
        from tradex.engines.adaptive_thresholds import VolatilityRegime
        low = params[VolatilityRegime.LOW]["sl_atr_multiplier"]
        high = params[VolatilityRegime.HIGH]["sl_atr_multiplier"]
        self.assertLess(low, high)

    def test_high_vol_stricter_confidence(self):
        """Higher volatility should require higher confidence."""
        from tradex.engines.adaptive_thresholds import (
            AdaptiveThresholdEngine, VolatilityRegime
        )
        params = AdaptiveThresholdEngine.REGIME_PARAMS
        low_conf = params[VolatilityRegime.LOW]["min_confidence"]
        high_conf = params[VolatilityRegime.HIGH]["min_confidence"]
        self.assertLess(low_conf, high_conf)

    def test_insufficient_data(self):
        """Small dataset should default to NORMAL regime."""
        from tradex.engines.adaptive_thresholds import VolatilityRegime
        tiny = make_ohlcv(20)
        result = self.engine.analyze(tiny)
        self.assertEqual(result.regime, VolatilityRegime.NORMAL)

    def test_adapted_params_populated(self):
        """All adapted parameters should be set."""
        df = make_ohlcv(200)
        result = self.engine.analyze(df)
        self.assertGreater(result.sl_atr_multiplier, 0)
        self.assertGreater(result.min_confidence, 0)
        self.assertGreater(result.signal_cooldown_min, 0)
        self.assertGreater(result.max_signals_per_day, 0)


# ===================================================================
# TEST 6: Economic Calendar
# ===================================================================

class TestEconomicCalendar(unittest.TestCase):
    """Test economic event scheduling and blocking."""

    def setUp(self):
        from tradex.engines.economic_calendar import EconomicCalendar
        self.calendar = EconomicCalendar()

    def test_add_event(self):
        """Should accept and store events."""
        from tradex.engines.economic_calendar import (
            EconomicEvent, EventType, EventImpact
        )
        ev = EconomicEvent(
            name="Test Event",
            event_type=EventType.INTEREST_RATE,
            impact=EventImpact.HIGH,
            scheduled_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            currency_affected=["USD"],
        )
        self.calendar.add_event(ev)
        self.assertEqual(len(self.calendar._events), 1)

    def test_block_before_high_impact(self):
        """Signals should be blocked close to HIGH events."""
        from tradex.engines.economic_calendar import (
            EconomicEvent, EventType, EventImpact
        )
        now = datetime.now(timezone.utc)
        ev = EconomicEvent(
            name="FOMC Decision",
            event_type=EventType.INTEREST_RATE,
            impact=EventImpact.HIGH,
            scheduled_time=now + timedelta(minutes=30),
            currency_affected=["USD", "EURUSD"],
        )
        self.calendar.add_event(ev)
        result = self.calendar.check("EURUSD", current_time=now)
        self.assertTrue(result.should_block_signals)

    def test_clear_when_no_events(self):
        """No events should mean CLEAR."""
        result = self.calendar.check("BTCUSDT")
        self.assertFalse(result.should_block_signals)
        self.assertFalse(result.should_reduce_confidence)

    def test_caution_for_medium_impact(self):
        """MEDIUM impact should trigger caution, not block."""
        from tradex.engines.economic_calendar import (
            EconomicEvent, EventType, EventImpact
        )
        now = datetime.now(timezone.utc)
        ev = EconomicEvent(
            name="GDP Report",
            event_type=EventType.GDP,
            impact=EventImpact.MEDIUM,
            scheduled_time=now + timedelta(minutes=30),
            currency_affected=["USD"],
        )
        self.calendar.add_event(ev)
        result = self.calendar.check("AAPL", current_time=now)
        self.assertTrue(result.should_reduce_confidence)

    def test_affected_symbols_populated(self):
        """Affected symbols should list related pairs."""
        from tradex.engines.economic_calendar import (
            EconomicEvent, EventType, EventImpact
        )
        ev = EconomicEvent(
            name="Fed Decision",
            event_type=EventType.INTEREST_RATE,
            impact=EventImpact.HIGH,
            scheduled_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            currency_affected=["USD", "EURUSD", "GBPUSD"],
        )
        self.calendar.add_event(ev)
        result = self.calendar.check("EURUSD")
        self.assertIn("EURUSD", result.affected_symbols)


# ===================================================================
# TEST 7: Position Sizing (Kelly Criterion)
# ===================================================================

class TestPositionSizer(unittest.TestCase):
    """Test Kelly Criterion position sizing."""

    def setUp(self):
        from tradex.engines.position_sizer import PositionSizer
        self.sizer = PositionSizer(kelly_fraction=0.25, max_risk_pct=5.0)

    def test_kelly_formula_positive_edge(self):
        """Positive edge should produce positive Kelly fraction."""
        kelly = self.sizer.calculate_kelly(
            win_rate=0.55, avg_win=2.0, avg_loss=1.0
        )
        self.assertGreater(kelly, 0)

    def test_kelly_formula_no_edge(self):
        """No edge (break-even) should produce zero or negative Kelly."""
        kelly = self.sizer.calculate_kelly(
            win_rate=0.33, avg_win=1.0, avg_loss=1.0
        )
        self.assertLessEqual(kelly, 0.01)

    def test_risk_cap_enforced(self):
        """Position size should not exceed max risk cap."""
        result = self.sizer.calculate_simple(
            capital=10000,
            entry_price=100.0,
            stop_loss=99.0,
            win_rate=0.80,
            reward_risk_ratio=5.0,
        )
        self.assertLessEqual(result.position_size_pct, 5.0)

    def test_units_calculation(self):
        """Units should be consistent with risk and SL distance."""
        result = self.sizer.calculate_simple(
            capital=10000,
            entry_price=150.0,
            stop_loss=147.0,
            win_rate=0.55,
            reward_risk_ratio=2.0,
        )
        expected_units = result.position_size_usd / 3.0
        self.assertAlmostEqual(result.shares_or_units, expected_units, places=2)

    def test_from_trades_insufficient_history(self):
        """With < 10 trades, should use fixed fractional."""
        result = self.sizer.calculate_from_trades(
            trades=[{"pnl": 100, "pnl_pct": 1.0}],
            capital=10000,
            entry_price=100,
            stop_loss=98,
        )
        self.assertEqual(result.method, "FIXED_FRACTIONAL")

    def test_from_trades_sufficient_history(self):
        """With 10+ trades, should use Kelly or capped."""
        np.random.seed(42)
        trades = []
        for _ in range(20):
            win = np.random.random() < 0.55
            pnl = 200 if win else -100
            trades.append({
                "pnl": pnl,
                "pnl_pct": abs(pnl) / 10000 * 100,
            })
        result = self.sizer.calculate_from_trades(
            trades=trades, capital=10000,
            entry_price=45000, stop_loss=44000,
        )
        self.assertIn(result.method, ["KELLY", "CAPPED"])


# ===================================================================
# TEST 8: Signal Performance Tracker
# ===================================================================

class TestPerformanceTracker(unittest.TestCase):
    """Test closed-loop signal performance tracking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from tradex.engines.performance_tracker import SignalPerformanceTracker
        self.tracker = SignalPerformanceTracker(data_dir=self.tmpdir)

    def test_register_signal(self):
        """Should register and store a new signal."""
        self.tracker.register_signal(
            "test1", "BTCUSDT", "LONG", "A",
            45000, 44000, 46000, 47000, 48000,
        )
        stats = self.tracker.get_stats()
        self.assertEqual(stats.total_signals, 1)
        self.assertEqual(stats.open_signals, 1)

    def test_update_outcome(self):
        """Should update outcome and calculate R-multiple."""
        self.tracker.register_signal(
            "test2", "ETHUSDT", "LONG", "A",
            3000, 2900, 3100, 3200, 3300,
        )
        self.tracker.update_outcome("test2", 3100, "TP1")
        stats = self.tracker.get_stats()
        self.assertEqual(stats.closed_signals, 1)
        self.assertEqual(stats.wins, 1)

    def test_win_rate_calculation(self):
        """Win rate should be calculated correctly."""
        for i in range(10):
            self.tracker.register_signal(
                f"s{i}", "TEST", "LONG", "A",
                100, 95, 105, 110, 115,
            )
            if i < 6:
                self.tracker.update_outcome(f"s{i}", 105, "TP1")
            else:
                self.tracker.update_outcome(f"s{i}", 95, "SL")

        stats = self.tracker.get_stats()
        self.assertAlmostEqual(stats.win_rate_pct, 60.0, places=0)

    def test_persistence(self):
        """Data should persist to disk and reload."""
        self.tracker.register_signal(
            "persist1", "AAPL", "LONG", "A",
            150, 148, 152, 154, 156,
        )
        # Create new tracker loading from same dir
        from tradex.engines.performance_tracker import SignalPerformanceTracker
        tracker2 = SignalPerformanceTracker(data_dir=self.tmpdir)
        stats = tracker2.get_stats()
        self.assertEqual(stats.total_signals, 1)

    def test_per_symbol_breakdown(self):
        """Stats should break down by symbol."""
        self.tracker.register_signal("a1", "AAPL", "LONG", "A", 150, 148, 152, 154, 156)
        self.tracker.register_signal("b1", "GOOGL", "SHORT", "A", 2800, 2850, 2750, 2700, 2650)
        self.tracker.update_outcome("a1", 152, "TP1")
        self.tracker.update_outcome("b1", 2850, "SL")

        stats = self.tracker.get_stats()
        self.assertIn("AAPL", stats.performance_by_symbol)
        self.assertIn("GOOGL", stats.performance_by_symbol)


# ===================================================================
# TEST 9: Alert Notification System
# ===================================================================

class TestAlertNotifier(unittest.TestCase):
    """Test alert messaging and formatting."""

    def test_alert_message_text(self):
        """to_text() should produce readable output."""
        from tradex.engines.alert_notifier import AlertMessage
        alert = AlertMessage(
            title="Test Signal",
            symbol="BTCUSDT",
            direction="LONG",
            grade="A",
            entry_price=45000,
            stop_loss=44000,
            take_profit_1=46500,
            take_profit_2=48000,
            take_profit_3=50000,
            confidence=85,
            checklist=["Trend: PASS", "Structure: PASS"],
            timestamp="2024-01-01T00:00:00",
        )
        text = alert.to_text()
        self.assertIn("BTCUSDT", text)
        self.assertIn("LONG", text)
        self.assertIn("45000", text)

    def test_alert_message_html(self):
        """to_html() should produce valid HTML."""
        from tradex.engines.alert_notifier import AlertMessage
        alert = AlertMessage(
            title="Test", symbol="ETH", direction="SHORT",
            grade="A", entry_price=3000, stop_loss=3100,
            take_profit_1=2900, take_profit_2=2800,
            take_profit_3=2700, confidence=80,
            checklist=["OK"], timestamp="now",
        )
        html = alert.to_html()
        self.assertIn("<div", html)
        self.assertIn("ETH", html)

    def test_telegram_not_configured(self):
        """Telegram should report not enabled without credentials."""
        from tradex.engines.alert_notifier import TelegramNotifier
        notifier = TelegramNotifier()
        self.assertFalse(notifier.is_enabled)

    def test_email_not_configured(self):
        """Email should report not enabled without credentials."""
        from tradex.engines.alert_notifier import EmailNotifier
        notifier = EmailNotifier()
        self.assertFalse(notifier.is_enabled)

    def test_alert_manager_history(self):
        """AlertManager should track dispatch history."""
        from tradex.engines.alert_notifier import AlertManager, AlertMessage
        manager = AlertManager()
        alert = AlertMessage(
            title="T", symbol="X", direction="LONG",
            grade="A", entry_price=100, stop_loss=95,
            take_profit_1=105, take_profit_2=110,
            take_profit_3=115, confidence=90,
            checklist=[], timestamp="now",
        )
        # Will produce empty results (no channels configured)
        results = manager.send_signal_alert(alert)
        self.assertIsInstance(results, dict)


# ===================================================================
# TEST 10: Cross-Module Integration
# ===================================================================

class TestCrossModuleIntegration(unittest.TestCase):
    """Test that all enhancements work together."""

    def test_volume_with_adaptive_thresholds(self):
        """Volume analyser and adaptive thresholds on same data."""
        from tradex.engines.volume_analyzer import VolumeAnalyzer
        from tradex.engines.adaptive_thresholds import AdaptiveThresholdEngine

        df = make_ohlcv(200, trend=0.2, volatility=2.0)

        vol = VolumeAnalyzer().analyze(df, "TEST")
        adapt = AdaptiveThresholdEngine().analyze(df)

        # Both should produce valid results on the same data
        self.assertGreaterEqual(vol.confirmation_score, 0.0)
        self.assertGreater(adapt.current_atr, 0.0)

    def test_calendar_with_position_sizer(self):
        """Economic calendar and position sizer interoperability."""
        from tradex.engines.economic_calendar import (
            EconomicCalendar, EconomicEvent, EventType, EventImpact
        )
        from tradex.engines.position_sizer import PositionSizer

        cal = EconomicCalendar()
        now = datetime.now(timezone.utc)
        cal.add_event(EconomicEvent(
            name="CPI", event_type=EventType.INFLATION,
            impact=EventImpact.HIGH,
            scheduled_time=now + timedelta(hours=2),
            currency_affected=["USD"],
        ))

        check = cal.check("EURUSD", now)
        sizer = PositionSizer()

        if check.should_block_signals:
            # Position sizer should not be called when blocked
            pass
        else:
            result = sizer.calculate_simple(
                capital=10000, entry_price=1.08,
                stop_loss=1.075, win_rate=0.55,
            )
            self.assertGreater(result.position_size_usd, 0)

    def test_performance_tracker_with_alerts(self):
        """Tracker and alert system work together."""
        from tradex.engines.performance_tracker import SignalPerformanceTracker
        from tradex.engines.alert_notifier import AlertManager, AlertMessage

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = SignalPerformanceTracker(data_dir=tmpdir)
            tracker.register_signal(
                "int1", "TEST", "LONG", "A",
                100, 95, 105, 110, 115,
            )

            alert = AlertMessage(
                title="Signal", symbol="TEST", direction="LONG",
                grade="A", entry_price=100, stop_loss=95,
                take_profit_1=105, take_profit_2=110,
                take_profit_3=115, confidence=85,
                checklist=["PASS"], timestamp="now",
            )

            manager = AlertManager()
            results = manager.send_signal_alert(alert)
            self.assertIsInstance(results, dict)

            stats = tracker.get_stats()
            self.assertEqual(stats.total_signals, 1)


# --- Run Tests ---

if __name__ == "__main__":
    print("""
        TradeXY - Expanded Unit Tests
        Covers all 10 future enhancements
    """)

    unittest.main(verbosity=2)
