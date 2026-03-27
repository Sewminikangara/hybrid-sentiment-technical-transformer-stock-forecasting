"""
TradeXY Configuration - All Thresholds & Settings
===================================================
Central configuration for signal generation, Elliott Wave,
news risk filter, and market structure detection.

All thresholds are configurable per-symbol where noted.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class SignalGrade(Enum):
    A = "A"          # All conditions passed - emit
    B = "B"          # Most conditions - log only, do NOT emit
    BLOCKED = "BLOCKED"  # News risk blocked

class RiskState(Enum):
    CLEAR = "CLEAR"       # No risk - signals allowed
    CAUTION = "CAUTION"   # Proceed with reduced confidence
    BLOCK = "BLOCK"       # No signals - high-impact event window

class ImpactLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class MarketPhase(Enum):
    BULLISH_TREND = "BULLISH_TREND"
    BEARISH_TREND = "BEARISH_TREND"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"


STOCKS = [
    "AAPL", "GOOGL", "TSLA", "AMZN", "MSFT",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "CSEALL"
]

CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSD", "SOLUSD", "XRPUSD", "ADAUSD"
]

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"
]

ALL_SYMBOLS = STOCKS + CRYPTO_PAIRS + FOREX_PAIRS


@dataclass
class TimeframeConfig:
    """Per-symbol timeframe configuration."""
    trend_tf: str = "4h"      # Higher timeframe for trend filter
    entry_tf: str = "15m"     # Lower timeframe for entry confirmation

    # Override per symbol type
    @staticmethod
    def for_symbol(symbol: str) -> "TimeframeConfig":
        if symbol in STOCKS:
            return TimeframeConfig(trend_tf="1d", entry_tf="1h")
        elif symbol in CRYPTO_PAIRS:
            return TimeframeConfig(trend_tf="4h", entry_tf="15m")
        elif symbol in FOREX_PAIRS:
            return TimeframeConfig(trend_tf="4h", entry_tf="15m")
        return TimeframeConfig()


@dataclass
class TrendFilterConfig:
    """Trend identification thresholds."""
    ema_period: int = 200           # EMA period for trend bias
    structure_lookback: int = 50    # Bars to look back for HH/HL/LH/LL
    min_swing_size_atr: float = 0.5 # Minimum swing size in ATR units
    atr_period: int = 14            # ATR calculation period

    # Market structure confirmation
    min_higher_highs: int = 2       # Min HH count for bullish structure
    min_higher_lows: int = 2        # Min HL count for bullish structure
    min_lower_highs: int = 2        # Min LH count for bearish structure
    min_lower_lows: int = 2         # Min LL count for bearish structure


@dataclass
class MarketStructureConfig:
    """Break of Structure (BOS) and retest detection."""
    bos_min_break_atr: float = 0.3   # Min break size in ATR for BOS
    retest_tolerance_atr: float = 0.5 # How close price must return for retest
    retest_hold_bars: int = 3         # Min bars retest must hold
    swing_order: int = 5              # Swing point detection order
    zigzag_pct: float = 3.0           # ZigZag reversal percentage (%)
    zigzag_atr_mult: float = 1.5      # ZigZag reversal in ATR multiples


@dataclass
class ElliottWaveConfig:
    """Elliott Wave engine thresholds (Frost & Prechter based)."""
    # Fibonacci zones for Wave 2 retracement
    wave2_fib_min: float = 0.500     # Min retrace (50.0%)
    wave2_fib_max: float = 0.786     # Max retrace (78.6%)
    wave2_fib_ideal: float = 0.618   # Ideal retrace (61.8%)

    # Wave 3 extension targets
    wave3_min_extension: float = 1.618  # Min Wave3/Wave1 ratio
    wave3_ideal_extension: float = 2.618

    # Wave 4 retracement
    wave4_fib_min: float = 0.236
    wave4_fib_max: float = 0.500
    wave4_fib_ideal: float = 0.382

    # Confidence thresholds
    min_confidence: int = 80          # Minimum confidence (0-100) for A-grade
    lookback_periods: int = 120       # Rolling window for wave detection
    swing_order: int = 5              # Local extrema detection order

    # Momentum confirmation
    rsi_period: int = 14
    rsi_wave3_min: float = 50.0       # RSI must be above this for bullish Wave3
    macd_confirmation: bool = True     # Require MACD histogram expansion

    # Top N candidate counts to return
    top_n_counts: int = 2


@dataclass
class NewsRiskConfig:
    """News intelligence and risk filter thresholds."""
    # Impact classification
    high_impact_keywords: List[str] = field(default_factory=lambda: [
        "rate decision", "interest rate", "fed", "fomc", "ecb", "boj",
        "cpi", "inflation", "nfp", "non-farm", "gdp", "unemployment",
        "hack", "exploit", "rug pull", "sec lawsuit", "ban", "regulation",
        "exchange shutdown", "insolvency", "bankruptcy", "delisting",
        "war", "invasion", "sanctions", "default", "crash", "black swan"
    ])

    medium_impact_keywords: List[str] = field(default_factory=lambda: [
        "etf", "approval", "adoption", "partnership", "upgrade",
        "halving", "merge", "fork", "earnings", "revenue", "guidance",
        "stimulus", "trade deal", "tariff", "oil", "commodity"
    ])

    # Cooldown periods (minutes)
    high_impact_cooldown: int = 120     # Block signals for 2 hours
    medium_impact_cooldown: int = 30    # Caution for 30 minutes

    # Trust scoring
    min_source_trust: float = 0.3       # Ignore sources below this

    # Sentiment thresholds for blocking
    strong_negative_threshold: float = -0.6  # Block longs if sentiment below this
    strong_positive_threshold: float = 0.6   # Block shorts if sentiment above this

    # Deduplication
    dedup_similarity_threshold: float = 0.85  # Cosine similarity for dedup
    dedup_time_window_hours: int = 24          # Dedup within this window


@dataclass
class SignalConfig:
    """A-grade signal generation thresholds."""
    # Risk management (R-multiple based)
    default_risk_reward_1: float = 1.0   # TP1 = 1R
    default_risk_reward_2: float = 2.0   # TP2 = 2R
    default_risk_reward_3: float = 3.0   # TP3 = 3R

    # Stop loss
    sl_atr_multiplier: float = 1.5       # SL = swing low/high ± ATR * mult
    sl_buffer_pips: float = 5.0          # Extra buffer for SL

    # Signal cooldown
    min_signal_interval_minutes: int = 60  # No repeat signals within 1 hour
    max_signals_per_day: int = 3           # Max A-grade signals per symbol/day

    # Confirmation requirements (ALL must pass for A-grade)
    require_trend_filter: bool = True
    require_structure_bos: bool = True
    require_elliott_confidence: bool = True
    require_news_clear: bool = True
    require_momentum_confirm: bool = True


SOURCE_TRUST_SCORES: Dict[str, float] = {
    # Tier 1 - Highly trusted (0.9–1.0)
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "wsj.com": 0.93,
    "ft.com": 0.93,
    "cnbc.com": 0.90,
    "bbc.com/news/business": 0.90,

    # Tier 2 - Trusted (0.7–0.89)
    "coindesk.com": 0.85,
    "cointelegraph.com": 0.80,
    "theblock.co": 0.82,
    "decrypt.co": 0.78,
    "forexfactory.com": 0.80,
    "investing.com": 0.78,
    "marketwatch.com": 0.82,
    "seekingalpha.com": 0.75,
    "yahoo.com/finance": 0.75,
    "moneycontrol.com": 0.72,

    # Tier 3 - Moderate (0.4–0.69)
    "reddit.com/r/cryptocurrency": 0.50,
    "reddit.com/r/bitcoin": 0.48,
    "reddit.com/r/forex": 0.45,
    "reddit.com/r/wallstreetbets": 0.35,
    "reddit.com/r/stocks": 0.50,
    "medium.com": 0.40,

    # Default for unknown
    "_default": 0.30,
}


RSS_FEEDS: List[Dict] = [
    # Finance - General
    {"url": "https://feeds.reuters.com/reuters/businessNews", "source": "reuters.com", "category": "macro"},
    {"url": "https://feeds.bbci.co.uk/news/business/rss.xml", "source": "bbc.com/news/business", "category": "macro"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "source": "cnbc.com", "category": "macro"},
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories/", "source": "marketwatch.com", "category": "macro"},

    # Crypto
    {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "source": "coindesk.com", "category": "crypto"},
    {"url": "https://cointelegraph.com/rss", "source": "cointelegraph.com", "category": "crypto"},
    {"url": "https://decrypt.co/feed", "source": "decrypt.co", "category": "crypto"},

    # Forex
    {"url": "https://www.forexfactory.com/rss", "source": "forexfactory.com", "category": "forex"},
    {"url": "https://www.investing.com/rss/news.rss", "source": "investing.com", "category": "forex"},
]


ASSET_KEYWORDS: Dict[str, List[str]] = {
    # Stocks
    "AAPL": ["apple", "aapl", "iphone", "tim cook", "apple inc"],
    "GOOGL": ["google", "googl", "alphabet", "android", "sundar pichai"],
    "TSLA": ["tesla", "tsla", "elon musk", "ev", "electric vehicle", "cybertruck"],
    "AMZN": ["amazon", "amzn", "aws", "jeff bezos", "andy jassy"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "satya nadella"],
    "RELIANCE.NS": ["reliance", "mukesh ambani", "jio", "reliance industries"],
    "TCS.NS": ["tcs", "tata consultancy", "tata"],
    "INFY.NS": ["infosys", "infy"],
    "CSEALL": ["colombo", "cse", "sri lanka stock"],

    # Crypto
    "BTCUSDT": ["bitcoin", "btc", "satoshi", "btcusd"],
    "ETHUSDT": ["ethereum", "eth", "vitalik", "ethusd"],
    "BNBUSD": ["binance coin", "bnb", "binance"],
    "SOLUSD": ["solana", "sol"],
    "XRPUSD": ["xrp", "ripple"],
    "ADAUSD": ["cardano", "ada"],

    # Forex
    "EURUSD": ["eur/usd", "eurusd", "euro", "ecb", "eurozone"],
    "GBPUSD": ["gbp/usd", "gbpusd", "pound", "sterling", "bank of england", "boe"],
    "USDJPY": ["usd/jpy", "usdjpy", "yen", "boj", "bank of japan"],
    "AUDUSD": ["aud/usd", "audusd", "aussie", "rba", "australia"],
    "USDCAD": ["usd/cad", "usdcad", "loonie", "bank of canada"],
    "USDCHF": ["usd/chf", "usdchf", "swiss franc", "snb"],
}


@dataclass
class TradeXYConfig:
    """Master configuration for the TradeXY system."""
    trend: TrendFilterConfig = field(default_factory=TrendFilterConfig)
    structure: MarketStructureConfig = field(default_factory=MarketStructureConfig)
    elliott: ElliottWaveConfig = field(default_factory=ElliottWaveConfig)
    news: NewsRiskConfig = field(default_factory=NewsRiskConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "tradex"

    # Feature flags
    enable_telegram: bool = False
    enable_twitter: bool = False  # Skip - requires paid API
    enable_reddit: bool = True

    @staticmethod
    def default() -> "TradeXYConfig":
        return TradeXYConfig()

# Singleton default config
DEFAULT_CONFIG = TradeXYConfig.default()
