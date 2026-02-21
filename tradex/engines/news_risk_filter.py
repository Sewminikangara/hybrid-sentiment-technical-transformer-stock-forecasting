import hashlib
import re
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tradex.config import (
    RiskState, ImpactLevel, SOURCE_TRUST_SCORES,
    ASSET_KEYWORDS, NewsRiskConfig, DEFAULT_CONFIG
)

# --- Data Structures ---

@dataclass
class NewsItem:
    """A single news/social media item."""
    id: str                             # Unique hash ID
    title: str
    content: str                        # Full text or summary
    source: str                         # Domain or subreddit
    url: str
    published_at: datetime
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    
    # Computed scores
    source_trust: float = 0.0           # 0–1
    sentiment_score: float = 0.0        # -1 to +1
    topic: str = "general"              # macro, regulation, hack, etc.
    impact_level: ImpactLevel = ImpactLevel.LOW
    asset_mentions: List[str] = field(default_factory=list)  # ["BTCUSDT", "ETHUSDT"]
    
    # Dedup
    content_hash: str = ""              # For similarity dedup
    is_duplicate: bool = False

@dataclass
class SymbolRiskState:
    """Risk state for a single symbol."""
    symbol: str
    state: RiskState = RiskState.CLEAR
    reasons: List[str] = field(default_factory=list)
    cooldown_expiry: Optional[datetime] = None
    active_news_count: int = 0
    avg_sentiment: float = 0.0          # Average sentiment of recent news
    highest_impact: ImpactLevel = ImpactLevel.LOW
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_blocked(self) -> bool:
        return self.state == RiskState.BLOCK
    
    @property
    def is_clear(self) -> bool:
        return self.state == RiskState.CLEAR
    
    def __repr__(self):
        return f"{self.symbol}: {self.state.value} ({', '.join(self.reasons) or 'clear'})"

# --- Sentiment Analyzer (VADER-style rule-based) ---

class SimpleSentimentScorer:
    """
    Lightweight rule-based sentiment scorer for financial text.
    More sophisticated analysis uses the existing FinBERT pipeline.
    """
    
    # Financial sentiment lexicon
    POSITIVE_WORDS = {
        "bullish", "rally", "surge", "soar", "breakout", "gain", "profit",
        "upgrade", "approval", "adoption", "partnership", "innovation",
        "growth", "recovery", "optimistic", "strong", "beat", "exceed",
        "momentum", "record high", "all-time high", "ath", "pump",
        "accumulate", "institutional", "etf approved", "halving",
    }
    
    NEGATIVE_WORDS = {
        "bearish", "crash", "plunge", "dump", "collapse", "loss", "lawsuit",
        "hack", "exploit", "rug pull", "scam", "fraud", "ban", "regulation",
        "recession", "inflation", "default", "bankruptcy", "insolvency",
        "delisting", "warning", "sell-off", "liquidation", "fear",
        "downgrade", "miss", "disappoint", "investigation", "indictment",
        "sanctions", "war", "invasion", "black swan",
    }
    
    INTENSIFIERS = {"very", "extremely", "massive", "huge", "major", "critical"}
    NEGATION = {"not", "no", "never", "without", "hardly", "barely", "isn't", "wasn't"}
    
    def score(self, text: str) -> float:
        """
        Score text sentiment from -1 (very negative) to +1 (very positive).
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        bigrams = set()
        word_list = re.findall(r'\b\w+\b', text_lower)
        for i in range(len(word_list) - 1):
            bigrams.add(f"{word_list[i]} {word_list[i+1]}")
        
        all_tokens = words | bigrams
        
        pos_count = len(all_tokens & self.POSITIVE_WORDS)
        neg_count = len(all_tokens & self.NEGATIVE_WORDS)
        
        # Check for negation modifying sentiment words
        for i in range(len(word_list) - 1):
            if word_list[i] in self.NEGATION:
                next_word = word_list[i + 1]
                if next_word in self.POSITIVE_WORDS:
                    pos_count -= 1
                    neg_count += 0.5
                elif next_word in self.NEGATIVE_WORDS:
                    neg_count -= 1
                    pos_count += 0.5
        
        # Intensifier boost
        intensifier_count = len(words & self.INTENSIFIERS)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        raw_score = (pos_count - neg_count) / total
        
        # Boost with intensifiers
        if intensifier_count > 0 and abs(raw_score) > 0:
            raw_score *= (1.0 + 0.2 * min(intensifier_count, 3))
        
        return max(-1.0, min(1.0, raw_score))

# --- News Risk Filter ---

class NewsRiskFilter:
    """
    Processes news items and computes risk state per trading symbol.
    
    Used for Condition D in the signal engine:
        - No high-impact event window active
        - Sentiment does not contradict signal direction
        - Block signals during major negative news with cooldown
    """
    
    def __init__(self, config: Optional[NewsRiskConfig] = None):
        self.config = config or DEFAULT_CONFIG.news
        self.sentiment_scorer = SimpleSentimentScorer()
        
        # In-memory storage (backed by MongoDB in production)
        self._news_items: List[NewsItem] = []
        self._risk_states: Dict[str, SymbolRiskState] = {}
        self._seen_hashes: Set[str] = set()
    
    # --- Step 1: Ingest & Score ---
    
    def ingest_item(self, title: str, content: str, source: str,
                    url: str, published_at: datetime) -> Optional[NewsItem]:
        """
        Ingest a single news item: score trust, sentiment, topic, impact,
        and map to relevant assets.
        
        Returns None if the item is a duplicate.
        """
        # Generate content hash for dedup
        content_hash = hashlib.md5(
            (title + content[:200]).lower().encode()
        ).hexdigest()
        
        # Check dedup
        if content_hash in self._seen_hashes:
            return None
        
        item_id = hashlib.sha256(
            (url + str(published_at)).encode()
        ).hexdigest()[:16]
        
        # Source trust score
        source_trust = SOURCE_TRUST_SCORES.get(source, 
                       SOURCE_TRUST_SCORES.get("_default", 0.3))
        
        # Skip untrusted sources
        if source_trust < self.config.min_source_trust:
            return None
        
        # Sentiment scoring
        combined_text = f"{title} {content}"
        sentiment_score = self.sentiment_scorer.score(combined_text)
        
        # Topic classification
        topic = self._classify_topic(combined_text)
        
        # Impact level
        impact_level = self._classify_impact(combined_text, source_trust)
        
        # Asset mapping
        asset_mentions = self._map_assets(combined_text)
        
        # Build item
        item = NewsItem(
            id=item_id,
            title=title,
            content=content[:500],  # Truncate for storage
            source=source,
            url=url,
            published_at=published_at,
            source_trust=source_trust,
            sentiment_score=sentiment_score,
            topic=topic,
            impact_level=impact_level,
            asset_mentions=asset_mentions,
            content_hash=content_hash
        )
        
        # Store
        self._news_items.append(item)
        self._seen_hashes.add(content_hash)
        
        # Update risk states for mentioned symbols
        for symbol in asset_mentions:
            self._update_risk_state(symbol, item)
        
        return item
    
    # --- Step 2: Topic Classification ---
    
    def _classify_topic(self, text: str) -> str:
        """Classify news topic using keyword matching."""
        text_lower = text.lower()
        
        topic_keywords = {
            "regulation": ["regulation", "sec", "ban", "lawsuit", "compliance", 
                          "legal", "enforcement", "subpoena"],
            "macro": ["interest rate", "fed", "fomc", "ecb", "cpi", "inflation",
                     "gdp", "unemployment", "nfp", "central bank", "monetary policy"],
            "hack": ["hack", "exploit", "vulnerability", "breach", "stolen",
                    "rug pull", "scam", "fraud"],
            "exchange": ["exchange", "outage", "delisting", "listing", "maintenance",
                        "withdrawal", "deposit"],
            "etf": ["etf", "fund", "institutional", "grayscale", "blackrock"],
            "geopolitics": ["war", "invasion", "sanctions", "geopolitical", 
                           "conflict", "tariff", "trade war"],
            "earnings": ["earnings", "revenue", "profit", "guidance", "forecast",
                        "quarterly", "annual report"],
            "adoption": ["adoption", "partnership", "integration", "launch",
                        "upgrade", "development"],
        }
        
        best_topic = "general"
        best_score = 0
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_topic = topic
        
        return best_topic
    
    # --- Step 3: Impact Classification ---
    
    def _classify_impact(self, text: str, source_trust: float) -> ImpactLevel:
        """
        Classify news impact level using keywords + source trust.
        """
        text_lower = text.lower()
        
        # Check high-impact keywords
        high_matches = sum(1 for kw in self.config.high_impact_keywords 
                          if kw in text_lower)
        med_matches = sum(1 for kw in self.config.medium_impact_keywords 
                         if kw in text_lower)
        
        # High impact: multiple keyword matches OR high-trust source + any match
        if high_matches >= 2 or (high_matches >= 1 and source_trust >= 0.8):
            return ImpactLevel.HIGH
        elif high_matches >= 1 or med_matches >= 2:
            return ImpactLevel.MEDIUM
        elif med_matches >= 1:
            return ImpactLevel.MEDIUM if source_trust >= 0.7 else ImpactLevel.LOW
        
        return ImpactLevel.LOW
    
    # --- Step 4: Asset Mapping ---
    
    def _map_assets(self, text: str) -> List[str]:
        """Map news text to relevant trading symbols using entity matching."""
        text_lower = text.lower()
        matched_assets = []
        
        for symbol, keywords in ASSET_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matched_assets.append(symbol)
                    break  # One match per symbol is enough
        
        return list(set(matched_assets))
    
    # --- Step 5: Risk State Update ---
    
    @staticmethod
    def _to_naive_utc(dt: datetime) -> datetime:
        """Convert a datetime to naive UTC for consistent comparison."""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
    
    def _update_risk_state(self, symbol: str, item: NewsItem):
        """Update the risk state for a symbol based on a new news item."""
        if symbol not in self._risk_states:
            self._risk_states[symbol] = SymbolRiskState(symbol=symbol)
        
        state = self._risk_states[symbol]
        now = datetime.utcnow()
        
        # Check if existing cooldown has expired
        if state.cooldown_expiry and now > state.cooldown_expiry:
            state.state = RiskState.CLEAR
            state.reasons = []
            state.cooldown_expiry = None
        
        # Apply new impact
        if item.impact_level == ImpactLevel.HIGH:
            state.state = RiskState.BLOCK
            state.reasons.append(f"HIGH: {item.title[:80]}")
            state.cooldown_expiry = now + timedelta(
                minutes=self.config.high_impact_cooldown
            )
            state.highest_impact = ImpactLevel.HIGH
        
        elif item.impact_level == ImpactLevel.MEDIUM:
            if state.state != RiskState.BLOCK:  # Don't downgrade BLOCK
                state.state = RiskState.CAUTION
                state.reasons.append(f"MED: {item.title[:80]}")
                if not state.cooldown_expiry:
                    state.cooldown_expiry = now + timedelta(
                        minutes=self.config.medium_impact_cooldown
                    )
                state.highest_impact = max(
                    state.highest_impact, ImpactLevel.MEDIUM,
                    key=lambda x: [ImpactLevel.LOW, ImpactLevel.MEDIUM, 
                                   ImpactLevel.HIGH].index(x)
                )
        
        # Check directional sentiment blocking
        if (item.sentiment_score <= self.config.strong_negative_threshold
            and item.impact_level in (ImpactLevel.HIGH, ImpactLevel.MEDIUM)):
            state.state = RiskState.BLOCK
            state.reasons.append(f"Strong negative sentiment ({item.sentiment_score:.2f})")
            state.cooldown_expiry = now + timedelta(
                minutes=self.config.high_impact_cooldown
            )
        
        # Update aggregates
        recent_items = [
            n for n in self._news_items 
            if symbol in n.asset_mentions 
            and (now - self._to_naive_utc(n.published_at)).total_seconds() < 86400
        ]
        state.active_news_count = len(recent_items)
        if recent_items:
            state.avg_sentiment = sum(n.sentiment_score for n in recent_items) / len(recent_items)
        
        state.last_updated = now
        self._risk_states[symbol] = state
    
    # --- Public API ---
    
    def get_risk_state(self, symbol: str) -> SymbolRiskState:
        """Get current risk state for a symbol."""
        if symbol not in self._risk_states:
            return SymbolRiskState(symbol=symbol)
        
        state = self._risk_states[symbol]
        now = datetime.utcnow()
        
        # Check cooldown expiry
        if state.cooldown_expiry and now > state.cooldown_expiry:
            state.state = RiskState.CLEAR
            state.reasons = []
            state.cooldown_expiry = None
            self._risk_states[symbol] = state
        
        return state
    
    def check_signal_allowed(self, symbol: str, 
                             direction: str) -> Tuple[bool, str]:
        """
        Check if a signal is allowed for a symbol in a given direction.
        
        Condition D for the signal engine:
            1. Risk state is not BLOCK
            2. Sentiment doesn't contradict direction
        
        Returns:
            (allowed: bool, reason: str)
        """
        state = self.get_risk_state(symbol)
        
        # Check 1: BLOCK state
        if state.state == RiskState.BLOCK:
            return False, f"BLOCKED: {'; '.join(state.reasons[:2])}"
        
        # Check 2: Sentiment contradiction
        if direction == "LONG" and state.avg_sentiment < self.config.strong_negative_threshold:
            return False, f"Negative sentiment ({state.avg_sentiment:.2f}) contradicts LONG"
        
        if direction == "SHORT" and state.avg_sentiment > self.config.strong_positive_threshold:
            return False, f"Positive sentiment ({state.avg_sentiment:.2f}) contradicts SHORT"
        
        # Check 3: CAUTION - allow but note it
        if state.state == RiskState.CAUTION:
            return True, f"CAUTION: {'; '.join(state.reasons[:1])}"
        
        return True, "CLEAR"
    
    def get_all_risk_states(self) -> Dict[str, SymbolRiskState]:
        """Get risk states for all tracked symbols."""
        # Refresh expired cooldowns
        for symbol in list(self._risk_states.keys()):
            self.get_risk_state(symbol)
        return dict(self._risk_states)
    
    def get_recent_news(self, symbol: Optional[str] = None,
                        hours: int = 24,
                        limit: int = 50) -> List[NewsItem]:
        """Get recent news items, optionally filtered by symbol."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        items = [
            n for n in self._news_items
            if self._to_naive_utc(n.published_at) >= cutoff and not n.is_duplicate
        ]
        
        if symbol:
            items = [n for n in items if symbol in n.asset_mentions]
        
        # Sort by published date descending
        items.sort(key=lambda n: n.published_at, reverse=True)
        
        return items[:limit]
    
    def get_summary(self) -> Dict:
        """Dashboard summary of all risk states."""
        states = self.get_all_risk_states()
        
        return {
            "total_news_items": len(self._news_items),
            "blocked_symbols": [s for s, rs in states.items() if rs.state == RiskState.BLOCK],
            "caution_symbols": [s for s, rs in states.items() if rs.state == RiskState.CAUTION],
            "clear_symbols": [s for s, rs in states.items() if rs.state == RiskState.CLEAR],
            "risk_states": {s: {
                "state": rs.state.value,
                "reasons": rs.reasons[:3],
                "cooldown": str(rs.cooldown_expiry) if rs.cooldown_expiry else None,
                "sentiment": f"{rs.avg_sentiment:.2f}",
                "news_count": rs.active_news_count
            } for s, rs in states.items()}
        }

# --- Standalone Test ---

if __name__ == "__main__":
    print("""
        TradeXY - News Risk Filter
        CLEAR / CAUTION / BLOCK per Symbol
    """)
    
    nrf = NewsRiskFilter()
    
    # Simulate news intake
    test_news = [
        ("Bitcoin ETF Approved by SEC", "The SEC has approved the first spot Bitcoin ETF",
         "reuters.com", "https://reuters.com/btc-etf", ImpactLevel.HIGH),
        ("Ethereum Network Upgrade Successful", "The merge upgrade completed without issues",
         "coindesk.com", "https://coindesk.com/eth-merge", ImpactLevel.MEDIUM),
        ("Fed Raises Interest Rates by 0.25%", "FOMC decided to raise rates amid inflation concerns",
         "bloomberg.com", "https://bloomberg.com/fed", ImpactLevel.HIGH),
        ("Major Crypto Exchange Hacked", "Exchange loses $200M in exploit vulnerability breach",
         "cointelegraph.com", "https://ct.com/hack", ImpactLevel.HIGH),
        ("Apple Reports Record Revenue", "Apple beats earnings expectations with strong iPhone sales",
         "cnbc.com", "https://cnbc.com/aapl", ImpactLevel.MEDIUM),
        ("Tesla Stock Rallies on AI News", "Tesla announces major AI partnership for self-driving",
         "marketwatch.com", "https://mw.com/tsla", ImpactLevel.MEDIUM),
    ]
    
    for title, content, source, url, expected_impact in test_news:
        item = nrf.ingest_item(title, content, source, url, datetime.utcnow())
        if item:
            print(f"  [OK] {item.impact_level.value:6s} | {item.sentiment_score:+.2f} | "
                  f"{','.join(item.asset_mentions) or 'general':20s} | {title[:50]}")
        else:
            print(f"  [--] Duplicate skipped: {title[:50]}")
    
    print(f"\n  Risk States:")
    for symbol, state in nrf.get_all_risk_states().items():
        print(f"    {state}")
    
    # Test signal check
    print(f"\n  Signal Checks:")
    for symbol, direction in [("BTCUSDT", "LONG"), ("AAPL", "LONG"), 
                               ("ETHUSDT", "SHORT"), ("EURUSD", "LONG")]:
        allowed, reason = nrf.check_signal_allowed(symbol, direction)
        status = "[PASS] ALLOWED" if allowed else "[FAIL] BLOCKED"
        print(f"    {symbol:12s} {direction:5s}: {status} - {reason}")
    
    print("\nNews Risk Filter test complete.")
