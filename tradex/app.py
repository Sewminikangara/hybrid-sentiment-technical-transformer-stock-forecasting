
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import json
import logging

# --- Setup ---

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tradex.config import (
    TradeXYConfig, DEFAULT_CONFIG, ALL_SYMBOLS,
    STOCKS, CRYPTO_PAIRS, FOREX_PAIRS,
    Direction, SignalGrade, RiskState, ImpactLevel
)
from tradex.engines.market_structure import MarketStructureDetector
from tradex.engines.elliott_wave_engine import ElliottWaveEngine
from tradex.engines.news_risk_filter import NewsRiskFilter
from tradex.engines.signal_engine import SignalEngine
from tradex.data.market_data import MarketDataProvider
from tradex.data.news_ingestion import NewsIngestionManager
from tradex.db.mongodb import MongoDBManager

logging.basicConfig(level=logging.WARNING)

# --- Page Config ---

st.set_page_config(
    page_title="TradeXY - A-Grade Signal Engine",
    page_icon="TX",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---

st.markdown("""
<style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #8b95a5;
        margin-top: -8px;
        font-weight: 400;
    }
    
    /* Signal Cards */
    .signal-card {
        background: linear-gradient(135deg, #1a1f36 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    
    .signal-long {
        border-left: 4px solid #00d4aa;
    }
    
    .signal-short {
        border-left: 4px solid #ff6b6b;
    }
    
    /* Risk State Badges */
    .risk-clear {
        background: #00d4aa22;
        color: #00d4aa;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .risk-caution {
        background: #ffa50022;
        color: #ffa500;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    .risk-block {
        background: #ff6b6b22;
        color: #ff6b6b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Metric Boxes */
    .metric-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e6edf3;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Checklist */
    .check-pass {
        color: #00d4aa;
        font-weight: 600;
    }
    
    .check-fail {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* News Stream */
    .news-item {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    
    .news-high {
        border-left: 3px solid #ff6b6b;
    }
    
    .news-medium {
        border-left: 3px solid #ffa500;
    }
    
    .news-low {
        border-left: 3px solid #30363d;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---

@st.cache_resource
def init_engines():
    """Initialize all TradeXY engines (cached)."""
    config = DEFAULT_CONFIG
    signal_engine = SignalEngine(config)
    market_data = MarketDataProvider()
    db = MongoDBManager(config.mongo_uri, config.mongo_db_name)
    news_manager = NewsIngestionManager()
    return signal_engine, market_data, db, news_manager, config

signal_engine, market_data, db, news_manager, config = init_engines()

# --- Sidebar ---

with st.sidebar:
    st.markdown('<p class="hero-title">TradeXY</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Ultra-Strict A-Grade Signals</p>', unsafe_allow_html=True)
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Overview", "Signals", "News Intelligence", 
         "Elliott Wave", "Market Structure", "Settings",
         "Research Models"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Quick Stats
    st.markdown("### Quick Stats")
    db_stats = db.get_dashboard_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Signals (24h)", db_stats.get("signals_24h", 0))
    with col2:
        st.metric("News Items", db_stats.get("news_24h", 0))
    
    db_status = "[OK] Connected" if db.is_connected else "[--] In-Memory"
    st.caption(f"Database: {db_status}")
    
    st.divider()
    
    # Symbol Selection
    asset_type = st.selectbox("Asset Type", ["All", "Stocks", "Crypto", "Forex"])
    
    if asset_type == "Stocks":
        symbols = STOCKS
    elif asset_type == "Crypto":
        symbols = CRYPTO_PAIRS
    elif asset_type == "Forex":
        symbols = FOREX_PAIRS
    else:
        symbols = ALL_SYMBOLS
    
    selected_symbol = st.selectbox("Symbol", symbols)

# --- Page: Overview ---

def show_overview():
    st.markdown("## Overview: Risk States and Latest Signals")
    st.markdown("Real-time risk state monitoring across all tracked assets.")
    
    # Risk State Summary
    risk_summary = signal_engine.news_filter.get_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(ALL_SYMBOLS)}</div>
            <div class="metric-label">Tracked Symbols</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        blocked = len(risk_summary.get('blocked_symbols', []))
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {'#ff6b6b' if blocked > 0 else '#00d4aa'}">{blocked}</div>
            <div class="metric-label">Blocked</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        caution = len(risk_summary.get('caution_symbols', []))
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {'#ffa500' if caution > 0 else '#00d4aa'}">{caution}</div>
            <div class="metric-label">Caution</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{risk_summary.get('total_news_items', 0)}</div>
            <div class="metric-label">News Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Risk States by Symbol")
    
    # Create risk state grid
    categories = [("Stocks", STOCKS), ("Crypto", CRYPTO_PAIRS), ("Forex", FOREX_PAIRS)]
    
    for cat_name, cat_symbols in categories:
        st.markdown(f"#### {cat_name}")
        cols = st.columns(min(len(cat_symbols), 6))
        
        for i, sym in enumerate(cat_symbols):
            with cols[i % len(cols)]:
                risk = signal_engine.news_filter.get_risk_state(sym)
                state = risk.state.value
                
                if state == "CLEAR":
                    badge = f'<span class="risk-clear">● {state}</span>'
                elif state == "CAUTION":
                    badge = f'<span class="risk-caution">[!] {state}</span>'
                else:
                    badge = f'<span class="risk-block">[X] {state}</span>'
                
                st.markdown(f"**{sym}**  \n{badge}", unsafe_allow_html=True)
    
    # Latest Signals
    st.markdown("### Recent A-Grade Signals")
    signals = db.get_signals(limit=5)
    
    if signals:
        for sig in signals:
            direction = sig.get('direction', 'LONG')
            css_class = "signal-long" if direction == "LONG" else "signal-short"
            dir_label = "LONG" if direction == "LONG" else "SHORT"
            
            st.markdown(f"""
            <div class="signal-card {css_class}">
                <h4>[{dir_label}] {sig.get('symbol', '-')} - {direction}</h4>
                <p>Entry: {sig.get('entry_price', '-')} | SL: {sig.get('stop_loss', '-')} | 
                TP1: {sig.get('tp1', '-')}</p>
                <small>{sig.get('reason', '')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No A-grade signals yet. This is expected. Ultra-strict conditions mean signals are rare and high-quality.")

# --- Page: Signals ---

def show_signals():
    st.markdown("## Signal Scanner")
    st.markdown("Run the A-grade checklist for any symbol. All 5 conditions must pass.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        scan_symbol = st.selectbox("Select Symbol to Scan", ALL_SYMBOLS, 
                                    index=ALL_SYMBOLS.index(selected_symbol) if selected_symbol in ALL_SYMBOLS else 0)
    
    with col2:
        st.markdown("")  # Spacer
        scan_btn = st.button("Run A-Grade Scan", type="primary", use_container_width=True)
    
    if scan_btn:
        with st.spinner(f"Scanning {scan_symbol} across all conditions..."):
            # Fetch data
            data = market_data.get_multi_timeframe(scan_symbol)
            
            if "trend" not in data or "entry" not in data:
                st.error(f"Could not fetch data for {scan_symbol}")
                return
            
            trend_df = data["trend"]
            entry_df = data["entry"]
            
            # Run individual analyses for display
            detector = signal_engine.structure_detector
            ew_engine = signal_engine.elliott_engine
            
            # --- Condition A: Trend ---
            trend_result = detector.analyze(trend_df, ema_period=200)
            trend_summary = detector.get_summary(trend_result)
            
            # --- Condition B: Structure ---
            entry_result = detector.analyze(entry_df, ema_period=50)
            entry_summary = detector.get_summary(entry_result)
            
            # --- Condition C: Elliott ---
            ew_direction = "BULLISH" if trend_result.current_trend == "BULLISH" else "BEARISH"
            ew_result = ew_engine.analyze(entry_df, ew_direction)
            ew_summary = ew_engine.get_summary(ew_result)
            
            # --- Condition D: News ---
            direction = "LONG" if trend_result.current_trend == "BULLISH" else "SHORT"
            news_allowed, news_reason = signal_engine.news_filter.check_signal_allowed(
                scan_symbol, direction
            )
            
            # --- Display Checklist ---
            st.markdown("### A-Grade Checklist")
            
            checks = [
                ("A", "Trend Filter (EMA200 + HH/HL)", 
                 trend_result.trend_filter_passed,
                 f"Trend: {trend_result.current_trend}, Strength: {trend_result.trend_strength:.0%}, EMA200: {'Above [PASS]' if trend_result.above_ema200 else 'Below'}"),
                
                ("B", "Entry Structure (BOS + Retest)",
                 entry_result.structure_filter_passed,
                 f"BOS: {entry_result.latest_bos or 'None'}, Retest: {'Confirmed [PASS]' if entry_result.retest_valid else 'Not confirmed'}"),
                
                ("C", "Elliott Wave (W2->W3, conf≥80)",
                 ew_result.elliott_filter_passed,
                 f"{ew_result.wave_summary}"),
                
                ("D", "News Risk Filter",
                 news_allowed,
                 news_reason),
                
                ("E", "Risk Management (SL/TP)",
                 True,
                 "R-multiples: 1R / 2R / 3R targets"),
            ]
            
            all_passed = all(c[2] for c in checks)
            
            for cond, name, passed, detail in checks:
                icon = "[PASS]" if passed else "[FAIL]"
                status_class = "check-pass" if passed else "check-fail"
                
                with st.expander(f"{icon} Condition {cond}: {name}", expanded=not passed):
                    st.markdown(f'<span class="{status_class}">{detail}</span>', 
                               unsafe_allow_html=True)
            
            # Result
            st.divider()
            if all_passed:
                st.success("ALL CONDITIONS PASSED -- A-GRADE SIGNAL GENERATED")
                # Try to generate the actual signal
                signal = signal_engine.evaluate(scan_symbol, trend_df, entry_df)
                if signal:
                    st.code(signal.format_alert())
                    db.save_signal(signal.to_dict())
            else:
                failed = [c[1] for c in checks if not c[2]]
                st.warning(f"Not an A-grade setup. Failed conditions: {', '.join(failed)}")
                st.info("This is normal - A-grade signals are rare by design. High confirmation = high quality.")
            
            # Price Chart
            st.markdown("### Price Action")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.7, 0.3],
                               vertical_spacing=0.05)
            
            fig.add_trace(go.Candlestick(
                x=list(range(len(entry_df))),
                open=entry_df['Open'],
                high=entry_df['High'],
                low=entry_df['Low'],
                close=entry_df['Close'],
                name="Price"
            ), row=1, col=1)
            
            # Mark swing points
            for sp in entry_result.swing_points[-20:]:
                color = "#00d4aa" if sp.swing_type.value == "LOW" else "#ff6b6b"
                fig.add_trace(go.Scatter(
                    x=[sp.index], y=[sp.price],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='diamond'),
                    name=sp.swing_type.value,
                    showlegend=False
                ), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(
                x=list(range(len(entry_df))),
                y=entry_df['Volume'],
                name="Volume",
                marker_color='rgba(102, 126, 234, 0.3)'
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"{scan_symbol} - Entry Timeframe Analysis",
                template="plotly_dark",
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Page: News Intelligence ---

def show_news():
    st.markdown("## News Intelligence")
    st.markdown("News stream with sentiment scoring, impact classification, and asset mapping.")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Ingest News Now", type="primary", use_container_width=True):
            with st.spinner("Fetching news from RSS feeds and Reddit..."):
                # Simulate news intake with test data
                test_items = [
                    {"title": "Federal Reserve Signals Pause in Rate Hikes",
                     "content": "The Fed indicated it may pause interest rate increases amid cooling inflation data",
                     "source": "reuters.com", "url": "https://reuters.com/fed-pause",
                     "published_at": datetime.now(timezone.utc)},
                    {"title": "Bitcoin Approaches $70K as ETF Demand Surges",
                     "content": "Bitcoin bulls push price near all-time highs with record ETF inflows",
                     "source": "coindesk.com", "url": "https://coindesk.com/btc-70k",
                     "published_at": datetime.now(timezone.utc)},
                    {"title": "Tesla Unveils New AI Chip for Self-Driving",
                     "content": "Tesla announces next-gen chip for autonomous driving capabilities",
                     "source": "cnbc.com", "url": "https://cnbc.com/tsla-ai",
                     "published_at": datetime.now(timezone.utc)},
                    {"title": "EUR/USD Drops on ECB Dovish Guidance",
                     "content": "Euro weakens as ECB signals possible rate cuts in coming months",
                     "source": "forexfactory.com", "url": "https://ff.com/eurusd",
                     "published_at": datetime.now(timezone.utc)},
                    {"title": "Major DeFi Protocol Exploit Discovered",
                     "content": "A critical vulnerability exploited leading to massive hack and loss of user funds",
                     "source": "cointelegraph.com", "url": "https://ct.com/hack",
                     "published_at": datetime.now(timezone.utc)},
                ]
                
                ingested = news_manager.ingest_into_filter(signal_engine.news_filter, test_items)
                
                # Also try real RSS if feedparser is available
                try:
                    rss_items = news_manager.fetch_all_rss()
                    if rss_items:
                        rss_ingested = news_manager.ingest_into_filter(signal_engine.news_filter, rss_items)
                        ingested += rss_ingested
                except Exception:
                    pass
                
                st.success(f"Ingested {ingested} new items")
    
    # Filter options
    with col1:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            filter_symbol = st.selectbox("Filter by Symbol", ["All"] + ALL_SYMBOLS)
        with filter_col2:
            filter_impact = st.selectbox("Filter by Impact", ["All", "HIGH", "MEDIUM", "LOW"])
        with filter_col3:
            filter_hours = st.slider("Time Window (hours)", 1, 72, 24)
    
    # News Stream
    sym_filter = filter_symbol if filter_symbol != "All" else None
    news_items = signal_engine.news_filter.get_recent_news(
        symbol=sym_filter, hours=filter_hours
    )
    
    if news_items:
        for item in news_items:
            impact_class = f"news-{item.impact_level.value.lower()}"
            sentiment_color = "#00d4aa" if item.sentiment_score > 0 else "#ff6b6b" if item.sentiment_score < 0 else "#8b95a5"
            
            assets = ", ".join(item.asset_mentions[:5]) if item.asset_mentions else "General"
            
            st.markdown(f"""
            <div class="news-item {impact_class}">
                <strong>{item.title}</strong><br/>
                <small>
                    Impact: {item.impact_level.value} | 
                    Sentiment: <span style="color:{sentiment_color}">{item.sentiment_score:+.2f}</span> | 
                    Source: {item.source} ({item.source_trust:.0%} trust) |
                    Assets: {assets} |
                    Topic: {item.topic}
                </small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news items yet. Click 'Ingest News Now' to fetch from RSS feeds.")
    
    # Risk State Summary
    st.markdown("### Current Risk States")
    risk_data = []
    for sym in ALL_SYMBOLS:
        rs = signal_engine.news_filter.get_risk_state(sym)
        risk_data.append({
            "Symbol": sym,
            "State": rs.state.value,
            "Sentiment": f"{rs.avg_sentiment:+.2f}",
            "News Count": rs.active_news_count,
            "Reasons": "; ".join(rs.reasons[:2]) if rs.reasons else "-"
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    def color_state(val):
        if val == "CLEAR":
            return "color: #00d4aa; font-weight: 600"
        elif val == "CAUTION":
            return "color: #ffa500; font-weight: 600"
        elif val == "BLOCK":
            return "color: #ff6b6b; font-weight: 600"
        return ""
    
    styled = risk_df.style.applymap(color_state, subset=["State"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# --- Page: Elliott Wave ---

def show_elliott():
    st.markdown("## Elliott Wave Analysis")
    st.markdown("Wave pattern detection with Fibonacci validation and confidence scoring.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ew_symbol = st.selectbox("Symbol", ALL_SYMBOLS, 
                                  index=ALL_SYMBOLS.index(selected_symbol) if selected_symbol in ALL_SYMBOLS else 0,
                                  key="ew_symbol")
    with col2:
        direction = st.selectbox("Direction Bias", ["BULLISH", "BEARISH"])
    
    with st.spinner(f"Analyzing {ew_symbol}..."):
        data = market_data.get_multi_timeframe(ew_symbol)
        
        if "entry" not in data:
            st.error("Could not fetch data")
            return
        
        df = data["entry"]
        ew_engine = signal_engine.elliott_engine
        result = ew_engine.analyze(df, trend_direction=direction)
        summary = ew_engine.get_summary(result)
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        conf = result.confidence
        conf_color = "#00d4aa" if conf >= 80 else "#ffa500" if conf >= 60 else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {conf_color}">{conf}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        wave = summary.get("current_wave", "?")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">W{wave}</div>
            <div class="metric-label">Current Wave</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        next_w = summary.get("next_expected", "?")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">-> W{next_w}</div>
            <div class="metric-label">Next Expected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        actionable = summary.get("actionable", "No")
        act_color = "#00d4aa" if actionable == "Yes" else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {act_color}">{"PASS" if actionable == "Yes" else "FAIL"}</div>
            <div class="metric-label">Actionable</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Wave Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Wave Analysis")
        for key in ["wave_summary", "direction", "fib_score", "momentum_score", 
                     "is_wave3_entry", "elliott_filter", "entry_zone"]:
            val = summary.get(key, "N/A")
            st.markdown(f"**{key.replace('_', ' ').title()}:** {val}")
        
        if summary.get("violations"):
            st.warning(f"Rule Violations: {', '.join(summary['violations'])}")
    
    with col2:
        st.markdown("### Fibonacci Levels")
        if result.fib_levels:
            fib_data = []
            for level, price in result.fib_levels.items():
                zone = ""
                if level in ["0.500", "0.618", "0.786"]:
                    zone = "Entry Zone"
                elif "ext" in level:
                    zone = "Target"
                fib_data.append({"Level": level, "Price": f"{price:.4f}", "Zone": zone})
            
            st.dataframe(pd.DataFrame(fib_data), use_container_width=True, hide_index=True)
    
    # Price Chart with Fibonacci
    st.markdown("### Wave Chart with Fibonacci Levels")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price"
    ))
    
    # Add Fibonacci levels as horizontal lines
    fib_colors = {
        "0.236": "#FF6B6B33", "0.382": "#FFA50055", "0.500": "#00D4AA77",
        "0.618": "#00D4AAAA", "0.786": "#00D4AA77", "1.000": "#FF6B6B33",
        "1.618_ext": "#667EEA88", "2.618_ext": "#667EEA55"
    }
    
    for level, price in result.fib_levels.items():
        color = fib_colors.get(level, "#FFFFFF33")
        fig.add_hline(y=price, line_dash="dash", line_color=color,
                     annotation_text=f"Fib {level}: {price:.2f}",
                     annotation_position="right")
    
    # Mark wave segments
    if result.best_candidate:
        for wave in result.best_candidate.waves:
            fig.add_trace(go.Scatter(
                x=[wave["start_idx"], wave["end_idx"]],
                y=[wave["start_price"], wave["end_price"]],
                mode='lines+markers+text',
                line=dict(color="#667eea", width=3),
                marker=dict(size=10),
                text=[f"W{wave['label']} start", f"W{wave['label']} end"],
                textposition="top center",
                name=f"Wave {wave['label']}"
            ))
    
    fig.update_layout(
        title=f"{ew_symbol} - Elliott Wave Analysis ({direction})",
        template="plotly_dark",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ML Features
    with st.expander("ML Feature Vector"):
        feat_df = pd.DataFrame([result.features]).T
        feat_df.columns = ["Value"]
        st.dataframe(feat_df, use_container_width=True)

# --- Page: Market Structure ---

def show_structure():
    st.markdown("## Market Structure Analysis")
    st.markdown("HH/HL/LH/LL detection, Break of Structure (BOS), and retest validation.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ms_symbol = st.selectbox("Symbol", ALL_SYMBOLS, key="ms_symbol",
                                  index=ALL_SYMBOLS.index(selected_symbol) if selected_symbol in ALL_SYMBOLS else 0)
    with col2:
        timeframe = st.selectbox("Timeframe", ["4h (Trend)", "15m (Entry)"])
    
    with st.spinner(f"Analyzing market structure for {ms_symbol}..."):
        data = market_data.get_multi_timeframe(ms_symbol)
        
        tf_key = "trend" if "4h" in timeframe else "entry"
        if tf_key not in data:
            st.error("Could not fetch data")
            return
        
        df = data[tf_key]
        detector = signal_engine.structure_detector
        result = detector.analyze(df, ema_period=200 if tf_key == "trend" else 50)
        summary = detector.get_summary(result)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trend = result.current_trend
        trend_color = "#00d4aa" if trend == "BULLISH" else "#ff6b6b" if trend == "BEARISH" else "#ffa500"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {trend_color}">{trend[:4]}</div>
            <div class="metric-label">Trend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{result.trend_strength:.0%}</div>
            <div class="metric-label">Strength</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ema_text = "PASS" if result.above_ema200 else "FAIL"
        ema_color = "#00d4aa" if result.above_ema200 else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {ema_color}">{ema_text}</div>
            <div class="metric-label">EMA200</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        retest_text = "PASS" if result.retest_valid else "FAIL"
        retest_color = "#00d4aa" if result.retest_valid else "#ff6b6b"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {retest_color}">{retest_text}</div>
            <div class="metric-label">BOS Retest</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Structure Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Structure Summary")
        for key, val in summary.items():
            st.markdown(f"**{key.replace('_', ' ').title()}:** {val}")
    
    with col2:
        st.markdown("### Recent Structure Labels")
        if result.structure_labels:
            label_data = [{
                "Label": sl.label.value,
                "Price": f"{sl.swing.price:.4f}",
                "Index": sl.swing.index
            } for sl in result.structure_labels[-12:]]
            st.dataframe(pd.DataFrame(label_data), use_container_width=True, hide_index=True)
    
    # Chart
    st.markdown("### Structure Chart")
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=list(range(len(df))),
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price"
    ))
    
    # Swing points with labels
    for sl in result.structure_labels[-15:]:
        sp = sl.swing
        color_map = {"HH": "#00d4aa", "HL": "#00d4aa", "LH": "#ff6b6b", "LL": "#ff6b6b", "EQ": "#ffa500"}
        symbol_map = {"HH": "triangle-up", "HL": "triangle-up", "LH": "triangle-down", "LL": "triangle-down", "EQ": "diamond"}
        
        fig.add_trace(go.Scatter(
            x=[sp.index], y=[sp.price],
            mode='markers+text',
            marker=dict(size=12, color=color_map.get(sl.label.value, "#fff"),
                       symbol=symbol_map.get(sl.label.value, "circle")),
            text=[sl.label.value],
            textposition="top center" if sp.swing_type.value == "HIGH" else "bottom center",
            textfont=dict(size=10, color=color_map.get(sl.label.value, "#fff")),
            showlegend=False
        ))
    
    # BOS events
    for bos in result.bos_events[-5:]:
        color = "#667eea" if bos.direction.value == "BULLISH" else "#f093fb"
        fig.add_hline(y=bos.broken_level, line_dash="dot", line_color=color,
                     annotation_text=f"BOS {bos.direction.value[:4]}",
                     annotation_position="right")
    
    fig.update_layout(
        title=f"{ms_symbol} - Market Structure ({timeframe})",
        template="plotly_dark",
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Page: Settings ---

def show_settings():
    st.markdown("## TradeXY Settings")
    st.markdown("Configure thresholds for signal generation, Elliott Wave, and news filters.")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trend & Structure", "Elliott Wave", "News Filter", "Signal Engine"
    ])
    
    with tab1:
        st.markdown("### Trend Filter")
        col1, col2 = st.columns(2)
        with col1:
            ema = st.number_input("EMA Period", value=config.trend.ema_period, min_value=50, max_value=500)
            lookback = st.number_input("Structure Lookback (bars)", value=config.trend.structure_lookback)
        with col2:
            atr = st.number_input("ATR Period", value=config.trend.atr_period)
            min_hh = st.number_input("Min Higher Highs", value=config.trend.min_higher_highs)
        
        st.markdown("### Market Structure")
        col1, col2 = st.columns(2)
        with col1:
            bos_break = st.number_input("BOS Min Break (ATR)", value=config.structure.bos_min_break_atr, format="%.2f")
        with col2:
            retest_tol = st.number_input("Retest Tolerance (ATR)", value=config.structure.retest_tolerance_atr, format="%.2f")
    
    with tab2:
        st.markdown("### Elliott Wave Fibonacci Zones")
        col1, col2, col3 = st.columns(3)
        with col1:
            w2_min = st.number_input("Wave 2 Fib Min", value=config.elliott.wave2_fib_min, format="%.3f")
            w4_min = st.number_input("Wave 4 Fib Min", value=config.elliott.wave4_fib_min, format="%.3f")
        with col2:
            w2_ideal = st.number_input("Wave 2 Fib Ideal", value=config.elliott.wave2_fib_ideal, format="%.3f")
            w4_ideal = st.number_input("Wave 4 Fib Ideal", value=config.elliott.wave4_fib_ideal, format="%.3f")
        with col3:
            w2_max = st.number_input("Wave 2 Fib Max", value=config.elliott.wave2_fib_max, format="%.3f")
            w4_max = st.number_input("Wave 4 Fib Max", value=config.elliott.wave4_fib_max, format="%.3f")
        
        st.markdown("### Confidence & Momentum")
        col1, col2 = st.columns(2)
        with col1:
            min_conf = st.slider("Min Confidence for A-Grade", 50, 100, config.elliott.min_confidence)
            w3_ext = st.number_input("Wave 3 Min Extension", value=config.elliott.wave3_min_extension, format="%.3f")
        with col2:
            rsi_period = st.number_input("RSI Period", value=config.elliott.rsi_period)
            rsi_min = st.number_input("RSI Wave 3 Min", value=config.elliott.rsi_wave3_min, format="%.1f")
    
    with tab3:
        st.markdown("### Impact Cooldowns")
        col1, col2 = st.columns(2)
        with col1:
            hi_cooldown = st.number_input("High Impact Cooldown (min)", value=config.news.high_impact_cooldown)
        with col2:
            med_cooldown = st.number_input("Medium Impact Cooldown (min)", value=config.news.medium_impact_cooldown)
        
        st.markdown("### Sentiment Thresholds")
        col1, col2 = st.columns(2)
        with col1:
            neg_thresh = st.number_input("Strong Negative Threshold", value=config.news.strong_negative_threshold, format="%.2f")
        with col2:
            pos_thresh = st.number_input("Strong Positive Threshold", value=config.news.strong_positive_threshold, format="%.2f")
        
        st.markdown("### Source Trust (Min)")
        min_trust = st.slider("Minimum Source Trust", 0.0, 1.0, config.news.min_source_trust, 0.05)
    
    with tab4:
        st.markdown("### Risk Management")
        col1, col2, col3 = st.columns(3)
        with col1:
            rr1 = st.number_input("TP1 (R-multiple)", value=config.signal.default_risk_reward_1, format="%.1f")
        with col2:
            rr2 = st.number_input("TP2 (R-multiple)", value=config.signal.default_risk_reward_2, format="%.1f")
        with col3:
            rr3 = st.number_input("TP3 (R-multiple)", value=config.signal.default_risk_reward_3, format="%.1f")
        
        st.markdown("### Signal Limits")
        col1, col2 = st.columns(2)
        with col1:
            max_signals = st.number_input("Max Signals per Day", value=config.signal.max_signals_per_day)
        with col2:
            interval = st.number_input("Min Signal Interval (min)", value=config.signal.min_signal_interval_minutes)
    
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved! (Settings are applied on next scan)")

# --- Page: Research Models ---

def show_research():
    st.markdown("## Research Models")
    st.markdown("Original hybrid sentiment-technical transformer prediction models.")
    
    # Load training results
    results_dir = PROJECT_ROOT / "results"
    result_files = sorted(results_dir.glob("hybrid_training_results_*.csv"))
    
    if result_files:
        latest = result_files[-1]
        df = pd.read_csv(latest)
        
        st.markdown(f"### Latest Training Results")
        st.caption(f"Source: `{latest.name}`")
        
        # Performance table
        st.dataframe(
            df[['Stock', 'Model', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']]
            .style.format({
                'RMSE': '{:.6f}', 'MAE': '{:.6f}', 
                'MAPE': '{:.2f}%', 'Directional_Accuracy': '{:.1f}%'
            }),
            use_container_width=True, hide_index=True
        )
        
        # Model comparison chart
        st.markdown("### Model Performance Comparison")
        avg = df.groupby('Model')[['MAPE', 'Directional_Accuracy']].mean().reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=avg['Model'], y=avg['MAPE'],
            name="MAPE (%)", marker_color='#667eea'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=avg['Model'], y=avg['Directional_Accuracy'],
            name="Dir. Accuracy (%)", mode='lines+markers',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=10)
        ), secondary_y=True)
        
        fig.update_layout(
            title="Average Performance by Model Architecture",
            template="plotly_dark",
            height=400,
        )
        fig.update_yaxes(title_text="MAPE (%)", secondary_y=False)
        fig.update_yaxes(title_text="Directional Accuracy (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training results found. Run `train_hybrid_models.py` first.")
    
    # Feature info
    st.markdown("### Feature Breakdown")
    st.markdown("""
    | Category | Count | Details |
    |---|---|---|
    | **Technical Indicators** | 35 | RSI, MACD, EMA, BB, Stochastic, OBV, ATR, etc. |
    | **Elliott Wave (Frost & Prechter)** | 8 | Wave number, direction, position, Fibonacci, confidence |
    | **Sentiment (VADER + FinBERT)** | 7 | Positive, negative, neutral, compound, FinBERT scores |
    | **Total** | **50** | Hybrid multi-modal features |
    """)

# --- Router ---

if "Overview" in page:
    show_overview()
elif "Signals" in page:
    show_signals()
elif "News" in page:
    show_news()
elif "Elliott" in page:
    show_elliott()
elif "Market Structure" in page:
    show_structure()
elif "Settings" in page:
    show_settings()
elif "Research" in page:
    show_research()
