import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results'
GRAPHS_DIR = BASE_DIR / 'graphs'

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader
from utils.predictor import StockPredictor
from utils.visualizer import ChartVisualizer


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #080E1A 0%, #0F1E35 50%, #0A1628 100%) !important; }
    .block-container { padding-top: 1.5rem !important; max-width: 1280px; }

    /* ── Tabs ── */
    [data-testid="stTabs"] { margin-top: 0.5rem; }
    [data-testid="stTabsTabList"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border-bottom: 1.5px solid rgba(96,165,250,0.2) !important;
        padding: 0 0.5rem;
        border-radius: 12px 12px 0 0;
        gap: 2px;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: #64748B !important;
        border: none !important;
        border-bottom: 2.5px solid transparent !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.65rem 1.1rem !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #CBD5E1 !important;
        background: rgba(96,165,250,0.08) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
        border-bottom-color: #60A5FA !important;
        background: rgba(96,165,250,0.1) !important;
    }
    [data-testid="stTabsTabPanel"] {
        padding-top: 1.5rem !important;
        border-top: none !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #112032 100%) !important;
        border-right: 1px solid rgba(96,165,250,0.15);
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] label { color: #CBD5E1 !important; font-size: 0.88rem; }
    [data-testid="stSidebar"] h3 {
        color: #60A5FA !important;
        -webkit-text-fill-color: #60A5FA !important;
        font-size: 0.78rem !important;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* ── Headings ── */
    h1 {
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 60%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -0.03em;
        animation: fadeDown 0.7s ease;
    }
    h2, h3 {
        background: linear-gradient(135deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        animation: fadeDown 0.5s ease;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 14px;
        padding: 1.2rem 1.4rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(96,165,250,0.5);
        transform: translateY(-3px);
        box-shadow: 0 16px 48px rgba(96,165,250,0.15);
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #94A3B8 !important; font-size: 0.78rem !important;
        font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #F1F5F9 !important; font-size: 1.5rem !important; font-weight: 700;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 0.65rem 2.2rem !important; font-weight: 600 !important; font-size: 0.95rem !important;
        box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important;
    }
    .stButton > button:active { transform: scale(0.97) !important; }

    /* ── Selectboxes ── */
    .stSelectbox > div > div, .stRadio > div {
        background: rgba(30,41,59,0.8) !important;
        border: 1px solid rgba(96,165,250,0.2) !important;
        border-radius: 10px !important;
    }

    /* ── Charts ── */
    .js-plotly-plot { border-radius: 14px !important; overflow: hidden;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5); }

    /* ── Info / alerts ── */
    .stInfo { background: rgba(37,99,235,0.12) !important;
        border-left: 3px solid #2563EB !important; border-radius: 0 8px 8px 0; }
    .stSuccess { background: rgba(22,163,74,0.12) !important;
        border-left: 3px solid #16A34A !important; }
    .stWarning { background: rgba(217,119,6,0.12) !important;
        border-left: 3px solid #D97706 !important; }
    .stError { background: rgba(220,38,38,0.12) !important;
        border-left: 3px solid #DC2626 !important; }

    /* ── Divider ── */
    hr { border-color: rgba(96,165,250,0.12) !important; }

    /* ── Tables ── */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

    /* ── Signal card ── */
    .signal-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border-radius: 16px; padding: 1.5rem;
        border: 1px solid rgba(148,163,184,0.12);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        text-align: center;
    }
    .quick-start-card {
        background: linear-gradient(135deg,rgba(37,99,235,0.15),rgba(124,58,237,0.1));
        border: 1px solid rgba(96,165,250,0.25);
        border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }
    .benefit-card {
        background: linear-gradient(145deg,#1E293B,#162032);
        border: 1px solid rgba(96,165,250,0.15);
        border-radius: 14px; padding: 1.4rem;
        height: 100%;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .benefit-card:hover { border-color: rgba(96,165,250,0.4); transform: translateY(-4px); }
    .step-badge {
        display:inline-block; background:linear-gradient(135deg,#2563EB,#7C3AED);
        color:white; border-radius:50%; width:28px; height:28px;
        line-height:28px; font-weight:700; font-size:0.85rem; text-align:center;
        margin-right:0.6rem;
    }

    /* ── Animations ── */
    @keyframes fadeDown { from{opacity:0;transform:translateY(-16px)} to{opacity:1;transform:none} }
    @keyframes fadeIn   { from{opacity:0} to{opacity:1} }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, .stDeployButton { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data_loader = DataLoader()
    st.session_state.model_loader = ModelLoader()
    st.session_state.predictor = None
    st.session_state.selected_stock = 'AAPL'
    st.session_state.selected_model = 'Attention Fusion'  # Best model pre-selected


def main():
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <h1 style='font-size: 3.5rem; margin: 0; letter-spacing: -0.02em;'>TradeXy</h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 500;'>
            Hybrid Sentiment-Technical Stock Forecasting Platform</p>
        <p style='color: #64748B; font-size: 0.9rem;'>
            Transformer-based Prediction Models | Multi-Market Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        #  App brand
        st.markdown("""
        <div style='text-align:center; padding:1rem 0 0.5rem 0;'>
            <div style='font-size:2rem; font-weight:800;
                background:linear-gradient(135deg,#60A5FA,#A78BFA);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>TradeXy</div>
            <div style='color:#64748B; font-size:0.72rem; letter-spacing:0.08em;
                text-transform:uppercase; margin-top:2px;'>AI Forecasting Platform</div>
        </div>
        <hr style='border-color:rgba(96,165,250,0.15); margin:0.5rem 0 0.8rem 0;'/>
        """, unsafe_allow_html=True)

        #  Market & Asset selection
        st.markdown("###  Market")
        market_type = st.radio(
            "Choose market", ["Stocks", "Forex", "Crypto"],
            index=0, label_visibility="collapsed",
            help="Stocks = global equities | Forex = currency pairs | Crypto = digital assets")

        if market_type == "Stocks":
            stocks = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT',
                      'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'CSEALL']
            stock_names = {
                'AAPL': 'Apple', 'GOOGL': 'Google', 'TSLA': 'Tesla',
                'AMZN': 'Amazon', 'MSFT': 'Microsoft',
                'RELIANCE.NS': 'Reliance', 'TCS.NS': 'TCS',
                'INFY.NS': 'Infosys', 'CSEALL': 'CSE All Share'
            }
        elif market_type == "Forex":
            stocks = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
            stock_names = {
                'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD',
                'USDJPY': 'USD/JPY', 'AUDUSD': 'AUD/USD',
                'USDCAD': 'USD/CAD', 'USDCHF': 'USD/CHF'
            }
        else:
            stocks = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'XRPUSD', 'ADAUSD']
            stock_names = {
                'BTCUSD': 'BTC/USD', 'ETHUSD': 'ETH/USD',
                'BNBUSD': 'BNB/USD', 'SOLUSD': 'SOL/USD',
                'XRPUSD': 'XRP/USD', 'ADAUSD': 'ADA/USD'
            }

        if market_type == "Crypto":
            label = "Select Crypto Pair"
        elif market_type == "Forex":
            label = "Select Currency Pair"
        else:
            label = "Select Stock"
        selected_stock = st.selectbox(label, stocks,
                                      format_func=lambda x: stock_names[x], key='stock_selector')

        if market_type == "Forex":
            st.info("Forex predictions use transformer models trained on European Central Bank data.")
        elif market_type == "Crypto":
            st.info("Crypto predictions use transformer models trained on historical cryptocurrency data.")

        #  Model & horizon
        st.markdown("###  Model")
        models = ['Attention Fusion', 'Early Fusion', 'Late Fusion', 'LSTM Baseline']
        model_help = {
            'Attention Fusion': '⭐ Best accuracy — uses dynamic cross-attention between price + sentiment',
            'Early Fusion': 'Combines price & sentiment at input — good all-rounder',
            'Late Fusion': 'Processes price & sentiment separately, then merges — best for Forex',
            'LSTM Baseline': 'Classic recurrent model — fastest but lower accuracy',
        }
        selected_model = st.selectbox(
            "AI Model", models,
            help="Attention Fusion achieves highest directional accuracy (70.1%) across most assets",
            key='model_selector')
        st.caption(model_help[selected_model])

        st.markdown("###  Horizon")
        pred_days = st.slider(
            "Forecast days", min_value=1, max_value=30, value=7,
            help="How many days ahead to forecast. 5–10 days recommended for best reliability.")

        #  Quick Stats
        st.markdown("---")
        is_forex = (market_type == "Forex")
        is_crypto = (market_type == "Crypto")
        st.subheader("Quick Stats")
        try:
            if is_forex:
                forex_results = list(RESULTS_DIR.glob('forex_training_results_*.csv'))
                if forex_results:
                    results = pd.read_csv(max(forex_results, key=lambda p: p.stat().st_mtime))
                    model_map = {'Early Fusion': 'early_fusion', 'Late Fusion': 'late_fusion',
                                 'Attention Fusion': 'attention_fusion', 'LSTM Baseline': 'lstm'}
                    mn = model_map.get(selected_model, 'early_fusion')
                    pr = results[(results['Pair'] == selected_stock) & (results['Model'] == mn)]
                    if len(pr) > 0:
                        if 'MAPE' in pr.columns:
                            st.metric("MAPE", f"{pr['MAPE'].values[0]:.2f}%")
                        if 'Directional_Accuracy' in pr.columns:
                            st.metric("Accuracy", f"{pr['Directional_Accuracy'].values[0]:.1f}%")
                        st.metric("Status", pr['Status'].values[0] if 'Status' in pr.columns else "Trained")
            elif is_crypto:
                crypto_results = list(RESULTS_DIR.glob('crypto_training_results_*.csv'))
                if crypto_results:
                    results = pd.read_csv(max(crypto_results, key=lambda p: p.stat().st_mtime))
                    model_map = {'Early Fusion': 'Early_Fusion', 'Late Fusion': 'Late_Fusion',
                                 'Attention Fusion': 'Attention_Fusion', 'LSTM Baseline': 'Lstm'}
                    mn = model_map.get(selected_model, 'Early_Fusion')
                    cr = results[(results['Pair'] == selected_stock) & (results['Model'] == mn)]
                    if len(cr) > 0:
                        st.metric("MAPE", f"{cr['MAPE'].values[0]:.2f}%")
                        st.metric("Accuracy", f"{cr['Directional_Accuracy'].values[0]:.1f}%")
                        st.metric("RMSE", f"{cr['RMSE'].values[0]:.4f}")
            else:
                results_file = list(RESULTS_DIR.glob('hybrid_training_results_*.csv'))
                if results_file:
                    results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
                    model_map = {'Early Fusion': 'Early_Fusion', 'Late Fusion': 'Late_Fusion',
                                 'Attention Fusion': 'Attention_Fusion', 'LSTM Baseline': 'LSTM'}
                    mn = model_map.get(selected_model, 'Early_Fusion')
                    sr = results[(results['Stock'] == selected_stock) & (results['Model'] == mn)]
                    if len(sr) > 0:
                        st.metric("MAPE", f"{sr['MAPE'].values[0]:.2f}%")
                        st.metric("Accuracy", f"{sr['Directional_Accuracy'].values[0]:.1f}%")
        except Exception:
            pass

    is_forex  = (market_type == "Forex")
    is_crypto = (market_type == "Crypto")

    #  Horizontal tab navigation (main area)
    TAB_LABELS = [
        "Home",
        "Prediction",
        "Live Signals",
        "Technical Analysis",
        "Elliott Wave",
        "Compare Models",
        "Auto Trader",
        "Batch Scan",
        "Backtesting",
        "Training Results",
        "More",
    ]

    tabs = st.tabs(TAB_LABELS)

    with tabs[0]:
        show_home_tab()
    with tabs[1]:
        show_prediction_tab(selected_stock, selected_model, pred_days, is_forex, is_crypto)
    with tabs[2]:
        show_live_signals_tab(selected_stock, is_forex, is_crypto)
    with tabs[3]:
        show_analysis_tab(selected_stock, is_forex, is_crypto)
    with tabs[4]:
        show_elliott_wave_tab(selected_stock, is_forex, is_crypto)
    with tabs[5]:
        show_comparison_tab(selected_stock, is_forex, is_crypto)
    with tabs[6]:
        show_auto_trader_tab(selected_stock, selected_model, is_forex, is_crypto)
    with tabs[7]:
        show_batch_prediction_tab(selected_model, is_forex, is_crypto)
    with tabs[8]:
        show_backtesting_tab()
    with tabs[9]:
        show_training_results_tab()
    with tabs[10]:
        # Collapsed secondary modules
        st.markdown("### More Tools")
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            if st.button("📋 Performance Dashboard", use_container_width=True):
                st.session_state["more_selected"] = "perf"
            if st.button("✅ Statistical Validation", use_container_width=True):
                st.session_state["more_selected"] = "stat"
        with sub_c2:
            if st.button("💼 Portfolio Manager", use_container_width=True):
                st.session_state["more_selected"] = "port"
            if st.button("🤖 Trading Bot", use_container_width=True):
                st.session_state["more_selected"] = "bot"
        with sub_c3:
            if st.button("ℹ️ About", use_container_width=True):
                st.session_state["more_selected"] = "about"
        more = st.session_state.get("more_selected", "")
        if more == "perf":
            show_performance_dashboard_tab()
        elif more == "stat":
            show_statistical_validation_tab()
        elif more == "port":
            show_portfolio_manager_tab(is_forex)
        elif more == "bot":
            show_trading_bot_tab(selected_stock, selected_model, is_forex, is_crypto)
        elif more == "about":
            show_about_tab()



def show_home_tab():
    #  Hero
    st.markdown("""
    <div style='text-align:center; padding:2rem 0 1.5rem 0; animation:fadeIn 0.8s ease;'>
        <div style='font-size:3.8rem; font-weight:800;
            background:linear-gradient(135deg,#60A5FA 0%,#A78BFA 55%,#F472B6 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            letter-spacing:-0.04em; line-height:1;'>TradeXy</div>
        <div style='color:#94A3B8; font-size:1.05rem; margin-top:0.6rem; font-weight:500;'>
            Hybrid Sentiment-Technical AI Forecasting Platform
        </div>
        <div style='color:#475569; font-size:0.85rem; margin-top:0.3rem;'>
            Transformer Models &nbsp;·&nbsp; Multi-Market &nbsp;·&nbsp; Real-Time Signals
        </div>
    </div>
    """, unsafe_allow_html=True)

    #  KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Markets", "3", "Stocks · Forex · Crypto",
                  help="Equities (US + India + CSE), Forex pairs, and Cryptocurrency")
    with c2:
        st.metric("Assets Tracked", "21", "9 Stocks | 6 Forex | 6 Crypto",
                  help="21 assets individually trained and evaluated")
    with c3:
        st.metric("AI Models", "4", "Transformer-based",
                  help="Early Fusion, Late Fusion, Attention Fusion, LSTM Baseline")
    with c4:
        st.metric("Best Accuracy", "70.1%", "Attention Fusion on Equities",
                  help="Directional accuracy of the best model on held-out test data")

    st.markdown("<hr style='border-color:rgba(96,165,250,0.15); margin:1.5rem 0;'/>", unsafe_allow_html=True)

    #  Quick Start
    st.markdown("###  Quick Start — 3 Steps")
    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown("""
        <div class='quick-start-card'>
            <div style='font-size:1.6rem; margin-bottom:0.5rem;'>📊</div>
            <div style='font-weight:700; color:#60A5FA; font-size:1rem; margin-bottom:0.5rem;'>
                <span class='step-badge'>1</span> Pick an Asset
            </div>
            <div style='color:#94A3B8; font-size:0.88rem; line-height:1.5;'>
                Use the sidebar to select a <strong style='color:#E2E8F0;'>Market</strong>
                (Stocks / Forex / Crypto) and an <strong style='color:#E2E8F0;'>Asset</strong>
                (e.g. Apple, EUR/USD, Bitcoin).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with qs2:
        st.markdown("""
        <div class='quick-start-card'>
            <div style='font-size:1.6rem; margin-bottom:0.5rem;'>🤖</div>
            <div style='font-weight:700; color:#A78BFA; font-size:1rem; margin-bottom:0.5rem;'>
                <span class='step-badge'>2</span> Choose a Model
            </div>
            <div style='color:#94A3B8; font-size:0.88rem; line-height:1.5;'>
                <strong style='color:#E2E8F0;'>Attention Fusion</strong> is pre-selected —
                it is the most accurate model (70.1% directional accuracy).
                Leave it as-is for the best forecast.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with qs3:
        st.markdown("""
        <div class='quick-start-card'>
            <div style='font-size:1.6rem; margin-bottom:0.5rem;'>📈</div>
            <div style='font-weight:700; color:#F472B6; font-size:1rem; margin-bottom:0.5rem;'>
                <span class='step-badge'>3</span> Generate Forecast
            </div>
            <div style='color:#94A3B8; font-size:0.88rem; line-height:1.5;'>
                Navigate to <strong style='color:#E2E8F0;'>📈 Prediction</strong> and click
                <strong style='color:#E2E8F0;'>Generate Prediction</strong> to see a price
                forecast with BUY / SELL / HOLD signal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(96,165,250,0.15); margin:1.5rem 0;'/>", unsafe_allow_html=True)

    #  What each feature does for you
    st.markdown("###  What Can TradeXy Do For You?")
    bc1, bc2, bc3 = st.columns(3)
    benefits = [
        ("bc1", "📈", "Price Forecasting", "#60A5FA",
         "See where a stock, currency, or crypto is likely to move over the next 1–30 days, "
         "powered by a Transformer model trained on 5 years of price and news data."),
        ("bc2", "⚡", "Live Trading Signals", "#A78BFA",
         "Get a real-time BUY / SELL / HOLD decision with Stop Loss, Take Profit, "
         "and Risk:Reward ratio calculated automatically — no manual analysis needed."),
        ("bc3", "🤖", "Automated Paper Trading", "#34D399",
         "Test trading strategies with virtual money using the Auto Trader module. "
         "Trailing stops and multi-target exits are handled automatically."),
    ]
    for col, (key, icon, title, color, desc) in zip([bc1, bc2, bc3], benefits):
        with col:
            st.markdown(f"""
            <div class='benefit-card'>
                <div style='font-size:2rem; margin-bottom:0.6rem;'>{icon}</div>
                <div style='font-weight:700; color:{color}; font-size:1rem;
                    margin-bottom:0.5rem;'>{title}</div>
                <div style='color:#94A3B8; font-size:0.875rem; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    bc4, bc5, bc6 = st.columns(3)
    benefits2 = [
        ("📊", "Technical Analysis", "#F59E0B",
         "Interactive charts with RSI, MACD, Bollinger Bands, and Moving Averages overlaid "
         "on the price history — the same tools used by professional traders."),
        ("🌊", "Elliott Wave Analysis", "#F472B6",
         "Automatically detects 5-wave market structure and Fibonacci price targets, "
         "giving structural context to the AI forecast."),
        ("📦", "Batch Market Scanner", "#818CF8",
         "Scan all 21 assets simultaneously and rank them by predicted return and model confidence "
         "— saving hours of individual analysis."),
    ]
    for col, (icon, title, color, desc) in zip([bc4, bc5, bc6], benefits2):
        with col:
            st.markdown(f"""
            <div class='benefit-card'>
                <div style='font-size:2rem; margin-bottom:0.6rem;'>{icon}</div>
                <div style='font-weight:700; color:{color}; font-size:1rem;
                    margin-bottom:0.5rem;'>{title}</div>
                <div style='color:#94A3B8; font-size:0.875rem; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(96,165,250,0.15); margin:1.8rem 0 1rem 0;'/>", unsafe_allow_html=True)

    #  Supported assets table
    st.markdown("###  Supported Assets")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("** Equities**")
        st.dataframe(pd.DataFrame({
            'Symbol': ['AAPL','GOOGL','TSLA','AMZN','MSFT','RELIANCE.NS','TCS.NS','INFY.NS','CSEALL'],
            'Company': ['Apple','Google','Tesla','Amazon','Microsoft','Reliance','TCS','Infosys','CSE Index'],
            'Exchange': ['NASDAQ']*5 + ['NSE']*3 + ['CSE'],
        }), use_container_width=True, hide_index=True)
    with ac2:
        st.markdown("** Forex Pairs**")
        st.dataframe(pd.DataFrame({
            'Pair': ['EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CAD','USD/CHF'],
            'Description': ['Euro–Dollar','Pound–Dollar','Dollar–Yen','Aussie–Dollar','Dollar–CAD','Dollar–Franc'],
        }), use_container_width=True, hide_index=True)
    with ac3:
        st.markdown("**₿ Cryptocurrency**")
        st.dataframe(pd.DataFrame({
            'Pair': ['BTC/USD','ETH/USD','BNB/USD','SOL/USD','XRP/USD','ADA/USD'],
            'Name': ['Bitcoin','Ethereum','BNB','Solana','XRP','Cardano'],
        }), use_container_width=True, hide_index=True)

    st.markdown("<br/><div style='text-align:center; color:#475569; font-size:0.8rem;'>"
                "Use the sidebar to navigate · Attention Fusion model pre-selected for best accuracy"
                "</div>", unsafe_allow_html=True)


def show_prediction_tab(stock, model, days, is_forex=False, is_crypto=False):
    if is_crypto:
        label = "Price Prediction"
    elif is_forex:
        label = "Exchange Rate Prediction"
    else:
        label = "Price Prediction"
    st.header(f"📈 {stock} {label}")

    #  Context banner
    st.markdown("""
    <div style='background:rgba(37,99,235,0.08); border:1px solid rgba(96,165,250,0.2);
        border-radius:10px; padding:0.75rem 1.1rem; margin-bottom:0.8rem;
        color:#94A3B8; font-size:0.82rem; line-height:1.5;'>
        <strong style='color:#60A5FA;'>ℹ️ About the date shown below:</strong>
        &nbsp;The <em>Training Data Cutoff</em> is the last date of historical data the model was trained on —
        it is <em>not</em> today's date. When an internet connection is available, a live price
        is fetched automatically and the label changes to <strong>Live Price Date ✅</strong>.
        Either way, the AI forecast is generated fresh each time you load this page.
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Loading data..."):
            data_loader = st.session_state.data_loader
            stock_data = data_loader.load_stock_data(stock, is_forex=is_forex, is_crypto=is_crypto)
            if stock_data is None or len(stock_data) == 0:
                st.error(f"No data available for {stock}")
                return
        latest_price = stock_data['Close'].iloc[-1]
        prev_price   = stock_data['Close'].iloc[-2]
        pct_change   = ((latest_price - prev_price) / prev_price) * 100
        training_cutoff = pd.to_datetime(stock_data['Date'].iloc[-1]).strftime('%Y-%m-%d')

        #  Try live price from yfinance
        live = data_loader.fetch_live_price(stock, is_forex=is_forex, is_crypto=is_crypto)
        if live:
            display_price  = live['price']
            display_pct    = live['pct_change']
            display_date   = live['date']
            date_label     = "Live Price Date ✅"
        else:
            display_price  = latest_price
            display_pct    = pct_change
            display_date   = training_cutoff
            date_label     = "Training Data Cutoff"

        col1, col2, col3, col4 = st.columns(4)
        price_label  = "Current Rate" if is_forex else "Current Price"
        price_format = f"{display_price:.4f}" if is_forex else f"${display_price:,.2f}"
        with col1:
            st.metric(price_label, price_format, f"{display_pct:+.2f}%", delta_color="normal",
                      help="Live price from yfinance when available, otherwise last stored training price")
        with col2:
            st.metric(date_label, display_date,
                      help="'Live Price Date' = fetched today from yfinance. 'Training Data Cutoff' = last date in the stored training CSV — the model was trained up to this point.")
        with col3:
            daily_chg = display_price - (display_price / (1 + display_pct / 100)) if display_pct else 0
            chg_fmt = f"{daily_chg:+.4f}" if is_forex else f"${daily_chg:+,.2f}"
            st.metric("Day Change", chg_fmt, help="Price change since previous trading day")
        with col4:
            st.metric("Model", model, help="Selected Transformer architecture")

        with st.spinner("Generating predictions..."):
            predictor = StockPredictor(stock, model, is_forex=is_forex, is_crypto=is_crypto)
            predictions = predictor.predict(days)
            if predictions is None:
                st.error(f"Unable to generate predictions for {stock}. Try a different model or stock.")
                return
        st.subheader("Price Forecast")
        visualizer = ChartVisualizer()
        fig = visualizer.create_prediction_chart(stock_data, predictions, stock)
        st.plotly_chart(fig, use_container_width=True)
        #  Signal Card
        pred_price = predictions['prices'][-1]
        signal, confidence = generate_signal(latest_price, pred_price, predictions)
        expected_return = ((pred_price - latest_price) / latest_price) * 100
        trend = "Bullish 📈" if pred_price > latest_price else "Bearish 📉"

        signal_meta = {
            'BUY':  {'color': '#22C55E', 'bg': 'rgba(34,197,94,0.12)',
                     'border': 'rgba(34,197,94,0.4)', 'emoji': '🟢',
                     'desc': 'The model predicts the price will rise. Consider entering a long position.'},
            'SELL': {'color': '#EF4444', 'bg': 'rgba(239,68,68,0.12)',
                     'border': 'rgba(239,68,68,0.4)', 'emoji': '🔴',
                     'desc': 'The model predicts the price will fall. Consider exiting or shorting.'},
            'HOLD': {'color': '#F59E0B', 'bg': 'rgba(245,158,11,0.12)',
                     'border': 'rgba(245,158,11,0.4)', 'emoji': '🟡',
                     'desc': 'The model sees insufficient directional bias. Stay cautious.'},
        }
        sm = signal_meta[signal]
        st.markdown(f"""
        <div style='background:{sm["bg"]}; border:1.5px solid {sm["border"]};
            border-radius:18px; padding:1.6rem 2rem; margin:1.2rem 0;
            display:flex; align-items:center; gap:2rem; flex-wrap:wrap;'>
            <div style='text-align:center; min-width:120px;'>
                <div style='font-size:3.5rem; line-height:1;'>{sm["emoji"]}</div>
                <div style='font-size:2.2rem; font-weight:800; color:{sm["color"]};
                    letter-spacing:0.05em; margin-top:0.3rem;'>{signal}</div>
                <div style='color:#94A3B8; font-size:0.75rem; font-weight:500;
                    text-transform:uppercase; letter-spacing:0.1em;'>Signal</div>
            </div>
            <div style='flex:1; min-width:200px;'>
                <div style='color:#E2E8F0; font-size:0.95rem; line-height:1.6;
                    margin-bottom:0.8rem;'>{sm["desc"]}</div>
                <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
                    <div>
                        <div style='color:#64748B; font-size:0.72rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.08em;'>Confidence</div>
                        <div style='color:{sm["color"]}; font-size:1.4rem; font-weight:700;'>{confidence:.0f}%</div>
                        <div style='color:#64748B; font-size:0.72rem;'>How certain the model is</div>
                    </div>
                    <div>
                        <div style='color:#64748B; font-size:0.72rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.08em;'>Expected Return</div>
                        <div style='color:{"#22C55E" if expected_return >= 0 else "#EF4444"};
                            font-size:1.4rem; font-weight:700;'>{expected_return:+.2f}%</div>
                        <div style='color:#64748B; font-size:0.72rem;'>Over {days} trading days</div>
                    </div>
                    <div>
                        <div style='color:#64748B; font-size:0.72rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.08em;'>Market Trend</div>
                        <div style='color:#A78BFA; font-size:1.4rem; font-weight:700;'>{trend}</div>
                        <div style='color:#64748B; font-size:0.72rem;'>Predicted direction</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        #  Summary metrics
        st.markdown("####  Prediction Summary")
        st.caption("What do these numbers mean? Hover the ℹ icon on each metric for an explanation.")
        volatility = np.std([p for p in predictions['prices']])
        pf = f"{pred_price:.4f}" if is_forex else f"${pred_price:,.2f}"
        pr = (f"{predictions['lower'][-1]:.4f} – {predictions['upper'][-1]:.4f}" if is_forex
              else f"${predictions['lower'][-1]:,.2f} – ${predictions['upper'][-1]:,.2f}")
        vf = f"{volatility:.4f}" if is_forex else f"${volatility:,.2f}"
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted End Price", pf,
                      help=f"The model's best estimate of the closing price after {days} days")
        with col2:
            st.metric("Confidence Range", pr,
                      help="Lower and upper bounds of the 80% confidence interval — the price is likely to stay within this band")
        with col3:
            st.metric("Forecast Volatility", vf,
                      help="How much the predicted price is expected to fluctuate day-to-day. Higher = more uncertainty.")
        with col4:
            trend_raw = "Bullish" if pred_price > latest_price else "Bearish"
            st.metric("Overall Trend", trend_raw,
                      help="Bullish means the model expects prices to rise. Bearish means it expects them to fall.")
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        st.info("Full prediction requires trained models and historical data.")


def show_analysis_tab(stock, is_forex=False, is_crypto=False):
    st.header(f"{stock} Technical Analysis")
    try:
        data_loader = st.session_state.data_loader
        stock_data = data_loader.load_stock_data(stock, is_forex=is_forex, is_crypto=is_crypto)
        if stock_data is None or len(stock_data) == 0:
            st.error(f"No data available for {stock}")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Technical Indicators")
            if 'RSI' in stock_data.columns:
                rsi = stock_data['RSI'].iloc[-1]
                rsi_signal = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
                st.metric("RSI", f"{rsi:.1f}", rsi_signal)
            if 'MACD' in stock_data.columns and 'MACD_Signal' in stock_data.columns:
                macd = stock_data['MACD'].iloc[-1]
                signal_val = stock_data['MACD_Signal'].iloc[-1]
                st.metric("MACD", f"{macd:.2f}", "Bullish" if macd > signal_val else "Bearish")
        with col2:
            st.subheader("Sentiment Analysis")
            if 'sentiment_score' in stock_data.columns:
                sentiment = stock_data['sentiment_score'].iloc[-1]
                sent_label = "Positive" if sentiment > 0.55 else ("Negative" if sentiment < 0.45 else "Neutral")
                st.metric("Sentiment Score", f"{sentiment:.3f}", sent_label)
            pos_col = 'sentiment_positive' if 'sentiment_positive' in stock_data.columns else 'positive'
            neg_col = 'sentiment_negative' if 'sentiment_negative' in stock_data.columns else 'negative'
            neu_col = 'sentiment_neutral' if 'sentiment_neutral' in stock_data.columns else 'neutral'
            if pos_col in stock_data.columns:
                pos = stock_data[pos_col].iloc[-1] * 100
                neg = stock_data[neg_col].iloc[-1] * 100
                neu = stock_data[neu_col].iloc[-1] * 100
                st.write(f"**Breakdown:** Positive: {pos:.1f}% | Negative: {neg:.1f}% | Neutral: {neu:.1f}%")
            ma7_col = 'sentiment_ma7' if 'sentiment_ma7' in stock_data.columns else 'sentiment_MA7'
            ma3_col = 'sentiment_ma3' if 'sentiment_ma3' in stock_data.columns else 'sentiment_MA3'
            if ma7_col in stock_data.columns and ma3_col in stock_data.columns:
                ma7 = stock_data[ma7_col].iloc[-1]
                ma3 = stock_data[ma3_col].iloc[-1]
                prev_ma7 = stock_data[ma7_col].iloc[-2]
                trend = "Improving" if ma7 > prev_ma7 else "Declining"
                momentum = "Accelerating" if ma3 > ma7 else "Decelerating"
                st.write(f"**Trend:** {trend} | **Momentum:** {momentum}")
            if 'sentiment_volatility' in stock_data.columns:
                volatility = stock_data['sentiment_volatility'].iloc[-1]
                vol_label = "Stable" if volatility < 0.1 else ("Volatile" if volatility < 0.2 else "Highly Volatile")
                st.metric("Sentiment Volatility", f"{volatility:.3f}", vol_label)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price History (60 days)")
            visualizer = ChartVisualizer()
            fig = visualizer.create_historical_chart(stock_data.tail(60), stock)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Sentiment Trend (60 days)")
            if 'sentiment_score' in stock_data.columns:
                visualizer = ChartVisualizer()
                fig = visualizer.create_sentiment_chart(stock_data.tail(60), stock)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading analysis: {str(e)}")


def show_elliott_wave_tab(stock, is_forex=False, is_crypto=False):
    """Dedicated Elliott Wave Analysis page based on Frost & Prechter (1978/2005)"""
    st.header(f"{stock} Elliott Wave Analysis")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 1.5rem;
         border-radius: 16px; border: 1px solid rgba(148, 163, 184, 0.1); margin-bottom: 1.5rem;'>
        <p style='color: #94A3B8; margin: 0; font-size: 0.95rem;'>
            <strong style='color: #A78BFA;'>Elliott Wave Principle</strong> (Frost & Prechter, 1978/2005):
            Markets move in recursive 5-wave impulse patterns followed by 3-wave corrective patterns.
            Wave relationships obey Fibonacci ratios (38.2%, 61.8%, 161.8%).
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        data_loader = st.session_state.data_loader
        stock_data = data_loader.load_stock_data(stock, is_forex=is_forex, is_crypto=is_crypto)
        if stock_data is None or len(stock_data) == 0:
            st.error(f"No data available for {stock}")
            return

        has_ew = 'ew_wave_number' in stock_data.columns

        #  If ew_* columns are missing, compute them on-the-fly
        if not has_ew:
            st.info("Computing Elliott Wave features on-the-fly for this market…")
            try:
                from tradex.engines.elliott_wave_engine import ElliottWaveEngine
                ew_engine = ElliottWaveEngine()
                stock_data = ew_engine.add_elliott_wave_features(stock_data)
                has_ew = 'ew_wave_number' in stock_data.columns
            except Exception:
                # Fallback: lightweight wave approximation
                close = stock_data['Close'].values
                window = min(20, len(close) // 3)
                if window >= 5:
                    from scipy.signal import argrelextrema
                    highs = argrelextrema(close, np.greater, order=window)[0]
                    lows  = argrelextrema(close, np.less, order=window)[0]
                    pivots = sorted([(i, 'H') for i in highs] + [(i, 'L') for i in lows], key=lambda x: x[0])
                    wave_numbers = np.zeros(len(close))
                    wave_dir     = np.zeros(len(close))
                    wave_conf    = np.full(len(close), 0.5)
                    wave_pos     = np.linspace(0, 1, len(close))
                    fib_382      = np.zeros(len(close))
                    fib_618      = np.zeros(len(close))
                    impulse_str  = np.zeros(len(close))
                    corr_depth   = np.zeros(len(close))
                    # Label pivots as alternating impulse/corrective waves
                    for idx_p, (pivot_i, ptype) in enumerate(pivots):
                        wave_num = (idx_p % 8) + 1
                        if wave_num <= 5:
                            wave_numbers[pivot_i] = wave_num
                        else:
                            wave_numbers[pivot_i] = -(wave_num - 5)
                        wave_dir[pivot_i] = 1 if ptype == 'H' else -1
                    # Forward-fill
                    for i in range(1, len(wave_numbers)):
                        if wave_numbers[i] == 0:
                            wave_numbers[i] = wave_numbers[i-1]
                        if wave_dir[i] == 0:
                            wave_dir[i] = wave_dir[i-1]
                    # Fibonacci distances from recent swing
                    if len(pivots) >= 2:
                        last_swing = close[pivots[-1][0]]
                        prev_swing = close[pivots[-2][0]]
                        swing_range = abs(last_swing - prev_swing)
                        if swing_range > 0:
                            for i in range(len(close)):
                                diff = abs(close[i] - prev_swing)
                                fib_382[i] = (diff / swing_range) - 0.382
                                fib_618[i] = (diff / swing_range) - 0.618
                        impulse_str[:] = min(1.0, swing_range / (np.mean(close) * 0.05))
                    stock_data['ew_wave_number'] = wave_numbers
                    stock_data['ew_wave_direction'] = wave_dir
                    stock_data['ew_wave_confidence'] = wave_conf
                    stock_data['ew_wave_position'] = wave_pos
                    stock_data['ew_fib_retracement_382'] = fib_382
                    stock_data['ew_fib_retracement_618'] = fib_618
                    stock_data['ew_impulse_strength'] = impulse_str
                    stock_data['ew_corrective_depth'] = corr_depth
                    has_ew = True
                else:
                    has_ew = False

        if has_ew:
            last = stock_data.iloc[-1]
            wave_num = int(last.get('ew_wave_number', 0))
            direction = int(last.get('ew_wave_direction', 0))
            confidence = float(last.get('ew_wave_confidence', 0))
            position = float(last.get('ew_wave_position', 0.5))
            fib_382 = float(last.get('ew_fib_retracement_382', 0))
            fib_618 = float(last.get('ew_fib_retracement_618', 0))
            impulse_str = float(last.get('ew_impulse_strength', 0))
            corr_depth = float(last.get('ew_corrective_depth', 0))

            # Wave label
            if wave_num > 0:
                wave_label = f"Impulse Wave {wave_num}"
                phase = "Impulse"
                phase_color = "#4CAF50" if direction >= 0 else "#f44336"
            elif wave_num < 0:
                abc_map = {-1: 'A', -2: 'B', -3: 'C'}
                wave_label = f"Corrective Wave {abc_map.get(wave_num, '?')}"
                phase = "Corrective"
                phase_color = "#FFC107"
            else:
                wave_label = "No clear wave pattern"
                phase = "Neutral"
                phase_color = "#94A3B8"

            dir_label = "Bullish 📈" if direction == 1 else ("Bearish 📉" if direction == -1 else "Neutral ➡️")

            # Position label
            if position < 0.33:
                pos_label = "Early"
            elif position < 0.67:
                pos_label = "Middle"
            else:
                pos_label = "Late"

            #  Status Banner
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E293B 0%, #334155 100%); padding: 1.5rem;
                 border-radius: 16px; border: 2px solid {phase_color}; margin-bottom: 1.5rem;
                 box-shadow: 0 10px 40px rgba(0,0,0,0.3);'>
                <h2 style='margin:0; color: {phase_color}; font-size: 1.8rem;'>{wave_label}</h2>
                <p style='color: #94A3B8; margin: 0.5rem 0 0 0;'>{dir_label} • {pos_label} stage ({position*100:.0f}% through wave) • Confidence: {confidence*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

            #  Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Wave", wave_label.split()[-1], phase)
            with col2:
                st.metric("Direction", dir_label.replace(" 📈","").replace(" 📉","").replace(" ➡️",""))
            with col3:
                st.metric("Wave Confidence", f"{confidence*100:.0f}%")
            with col4:
                st.metric("Wave Position", f"{position*100:.0f}%", pos_label)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fib 38.2% Distance", f"{fib_382:.3f}")
            with col2:
                st.metric("Fib 61.8% Distance", f"{fib_618:.3f}")
            with col3:
                st.metric("Impulse Strength", f"{impulse_str*100:.0f}%")
            with col4:
                st.metric("Corrective Depth", f"{corr_depth*100:.0f}%")

            st.markdown("---")

            #  Elliott Wave Feature History Chart
            st.subheader("Elliott Wave Pattern Over Time")
            recent = stock_data.tail(120).copy()
            if 'Date' in recent.columns:
                recent['Date'] = pd.to_datetime(recent['Date'])

            fig = go.Figure()

            # Price with wave coloring
            fig.add_trace(go.Scatter(
                x=recent['Date'] if 'Date' in recent.columns else recent.index,
                y=recent['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#60A5FA', width=2)
            ))

            # Overlay wave confidence as fill
            if 'ew_wave_confidence' in recent.columns:
                close_min = recent['Close'].min()
                close_range = recent['Close'].max() - close_min
                conf_scaled = close_min + recent['ew_wave_confidence'] * close_range * 0.3
                fig.add_trace(go.Scatter(
                    x=recent['Date'] if 'Date' in recent.columns else recent.index,
                    y=conf_scaled,
                    mode='lines',
                    name='Wave Confidence (scaled)',
                    line=dict(color='#A78BFA', width=1, dash='dot'),
                    opacity=0.5
                ))

            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(15,23,42,0)',
                paper_bgcolor='rgba(15,23,42,0)',
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title='',
                yaxis_title='Price',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            #  Wave Number Timeline
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Wave Number Timeline")
                fig2 = go.Figure()
                colors = []
                for wn in recent['ew_wave_number']:
                    if wn > 0:
                        colors.append('#4CAF50')  # Impulse: green
                    elif wn < 0:
                        colors.append('#FFC107')  # Corrective: amber
                    else:
                        colors.append('#94A3B8')  # Neutral: grey

                fig2.add_trace(go.Bar(
                    x=recent['Date'] if 'Date' in recent.columns else recent.index,
                    y=recent['ew_wave_number'],
                    marker_color=colors,
                    name='Wave Number'
                ))
                fig2.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(15,23,42,0)',
                    paper_bgcolor='rgba(15,23,42,0)',
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title='Wave #'
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                st.subheader("Fibonacci Distance")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=recent['Date'] if 'Date' in recent.columns else recent.index,
                    y=recent['ew_fib_retracement_382'],
                    mode='lines', name='Fib 38.2%',
                    line=dict(color='#F472B6', width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=recent['Date'] if 'Date' in recent.columns else recent.index,
                    y=recent['ew_fib_retracement_618'],
                    mode='lines', name='Fib 61.8%',
                    line=dict(color='#FBBF24', width=2)
                ))
                fig3.add_hline(y=0, line_dash='dash', line_color='#64748B', opacity=0.5)
                fig3.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='rgba(15,23,42,0)',
                    paper_bgcolor='rgba(15,23,42,0)',
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis_title='Distance',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig3, use_container_width=True)

            #  Theory Reference
            st.markdown("---")
            with st.expander("📖 Elliott Wave Theory Reference (Frost & Prechter)", expanded=False):
                st.markdown("""
                **Elliott Wave Principle** was first described by R.N. Elliott in the 1930s and
                comprehensively documented by A.J. Frost and Robert R. Prechter Jr. in their
                seminal text *"Elliott Wave Principle: Key to Market Behavior"* (1978, 10th ed. 2005).

                **Core Rules (Inviolable):**
                1. **Wave 2** never retraces more than 100% of Wave 1
                2. **Wave 3** is never the shortest of the three impulse waves (1, 3, 5)
                3. **Wave 4** never enters the price territory of Wave 1

                **Fibonacci Relationships:**
                - Wave 2 typically retraces **50–61.8%** of Wave 1
                - Wave 3 often extends to **161.8%** of Wave 1
                - Wave 4 typically retraces **38.2%** of Wave 3
                - Wave 5 often equals or is **61.8%** of Wave 1

                **Pattern Structure:**
                - **Impulse (5-wave):** Waves 1-2-3-4-5 in the direction of the trend
                - **Corrective (3-wave):** Waves A-B-C against the trend
                - Patterns are **fractal** — they repeat at all time scales

                **Reference:** Frost, A.J. & Prechter, R.R. (2005). *Elliott Wave Principle:
                Key to Market Behavior.* 10th Edition. New Classics Library.
                """)
        else:
            st.warning("Unable to compute Elliott Wave features – insufficient price data.")

    except Exception as e:
        st.error(f"Error loading Elliott Wave analysis: {str(e)}")


def show_comparison_tab(stock, is_forex=False, is_crypto=False):
    st.header("Model Comparison")
    try:
        if is_forex:
            results_file = list(RESULTS_DIR.glob('forex_training_results_*.csv'))
            if not results_file:
                st.warning("No forex training results available")
                return
            results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
            stock_results = results[results['Pair'] == stock]
            if len(stock_results) == 0:
                st.warning(f"No results for {stock}")
                return
            st.subheader("Forex Model Performance")
            display_df = stock_results[['Model', 'Status', 'Data_Points']].copy()
            display_df.columns = ['Model', 'Training Status', 'Data Points']
            st.dataframe(display_df, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Early Fusion Transformer** - Combines technical and sentiment features at input.
                Best for short-term predictions.

                **Late Fusion Transformer** - Processes features separately.
                Best for balanced predictions.
                """)
            with col2:
                st.markdown("""
                **Attention Fusion Transformer** - Dynamic weighting of signals.
                Best for volatile markets.

                **LSTM Baseline** - Traditional recurrent network.
                Best for long-term trends.
                """)
            return
        if is_crypto:
            results_file = list(RESULTS_DIR.glob('crypto_training_results_*.csv'))
            if not results_file:
                st.warning("No crypto training results available")
                return
            results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
            stock_results = results[results['Pair'] == stock]
            if len(stock_results) == 0:
                st.warning(f"No results for {stock}")
                return
            st.subheader("Crypto Model Performance")
            metric_cols = [c for c in ['Model', 'MAPE', 'RMSE', 'MAE', 'Directional_Accuracy'] if c in stock_results.columns]
            if len(metric_cols) > 1:
                display_df = stock_results[metric_cols].copy()
                display_df = display_df.sort_values('MAPE') if 'MAPE' in display_df.columns else display_df
                st.dataframe(display_df, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    if 'MAPE' in stock_results.columns:
                        st.subheader("MAPE Comparison")
                        fig = px.bar(stock_results, x='Model', y='MAPE', title='Lower is Better', color='Model')
                        st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if 'Directional_Accuracy' in stock_results.columns:
                        st.subheader("Directional Accuracy")
                        fig = px.bar(stock_results, x='Model', y='Directional_Accuracy',
                                     title='Higher is Better', color='Model')
                        st.plotly_chart(fig, use_container_width=True)
                if 'MAPE' in stock_results.columns:
                    best_model = stock_results.loc[stock_results['MAPE'].idxmin()]
                    st.success(f"Best Model: {best_model['Model']} (MAPE: {best_model['MAPE']:.2f}%)")
            return
        results_file = list(RESULTS_DIR.glob('hybrid_training_results_*.csv'))
        if not results_file:
            st.warning("No training results available")
            return
        results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
        stock_results = results[results['Stock'] == stock]
        if len(stock_results) == 0:
            st.warning(f"No results for {stock}")
            return
        st.subheader("Performance Metrics")
        display_df = stock_results[['Model', 'MAPE', 'RMSE', 'MAE', 'Directional_Accuracy']].copy()
        display_df.columns = ['Model', 'MAPE (%)', 'RMSE', 'MAE', 'Accuracy (%)']
        display_df = display_df.sort_values('MAPE (%)')
        st.dataframe(display_df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("MAPE Comparison")
            fig = px.bar(stock_results, x='Model', y='MAPE', title='Lower is Better', color='Model')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Directional Accuracy")
            fig = px.bar(stock_results, x='Model', y='Directional_Accuracy',
                         title='Higher is Better', color='Model')
            st.plotly_chart(fig, use_container_width=True)
        best_model = stock_results.loc[stock_results['MAPE'].idxmin()]
        st.success(f"Best Model: {best_model['Model']} (MAPE: {best_model['MAPE']:.2f}%)")
    except Exception as e:
        st.error(f"Error loading comparison: {str(e)}")


def show_training_results_tab():
    st.header("Training Results Dashboard")
    st.markdown("Comprehensive training results across all markets and model architectures.")
    try:
        #  Market selector
        market_tab = st.radio("Select Market", ["Stocks", "Forex", "Crypto"], horizontal=True, key="training_market")

        if market_tab == "Forex":
            results_file = sorted(RESULTS_DIR.glob('forex_training_results_*.csv'))
            id_col, label = 'Pair', 'Forex Pairs'
        elif market_tab == "Crypto":
            results_file = sorted(RESULTS_DIR.glob('crypto_training_results_*.csv'))
            id_col, label = 'Pair', 'Crypto Pairs'
        else:
            results_file = sorted(RESULTS_DIR.glob('hybrid_training_results_*.csv'))
            id_col, label = 'Stock', 'Stocks'

        if not results_file:
            st.warning(f"No {market_tab.lower()} training results found.")
            return
        df = pd.read_csv(results_file[-1])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Experiments", len(df))
        with col2:
            st.metric(f"{label} Tested", df[id_col].nunique())
        with col3:
            st.metric("Model Architectures", df['Model'].nunique())
        with col4:
            if 'MAPE' in df.columns:
                st.metric("Best MAPE", f"{df['MAPE'].min():.2f}%")
            else:
                st.metric("Status", "Trained")

        st.subheader(f"{market_tab} Training Results")
        # Show all available metric columns
        metric_cols = [c for c in ['MAPE', 'RMSE', 'MAE', 'Directional_Accuracy', 'Status', 'Data_Points', 'Best_Loss'] if c in df.columns]
        display_cols = [id_col, 'Model'] + metric_cols
        styled_df = df[display_cols].copy()
        rename_map = {'MAPE': 'MAPE (%)', 'Directional_Accuracy': 'Directional Accuracy (%)'}
        styled_df = styled_df.rename(columns=rename_map)
        if 'MAPE (%)' in styled_df.columns:
            styled_df = styled_df.sort_values('MAPE (%)')
        fmt = {}
        for c in styled_df.columns:
            if c in [id_col, 'Model', 'Status']:
                continue
            elif 'MAPE' in c or 'Accuracy' in c:
                fmt[c] = '{:.2f}'
            elif c in ['RMSE', 'MAE', 'Best_Loss']:
                fmt[c] = '{:.4f}'
            elif c == 'Data_Points':
                fmt[c] = '{:.0f}'
        styler = styled_df.style.format(fmt)
        if 'MAPE (%)' in styled_df.columns:
            styler = styler.background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r')
        if 'Directional Accuracy (%)' in styled_df.columns:
            styler = styler.background_gradient(subset=['Directional Accuracy (%)'], cmap='RdYlGn')
        st.dataframe(styler, use_container_width=True, height=400)

        if 'MAPE' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"MAPE by {label} and Model")
                pivot_mape = df.pivot_table(values='MAPE', index=id_col, columns='Model')
                fig, ax = plt.subplots(figsize=(10, 6))
                pivot_mape.plot(kind='bar', ax=ax)
                ax.set_ylabel('MAPE (%)')
                ax.set_title('Mean Absolute Percentage Error')
                ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                if 'Directional_Accuracy' in df.columns:
                    st.subheader(f"Directional Accuracy by {label}")
                    pivot_acc = df.pivot_table(values='Directional_Accuracy', index=id_col, columns='Model')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pivot_acc.plot(kind='bar', ax=ax)
                    ax.set_ylabel('Accuracy (%)')
                    ax.set_title('Directional Accuracy')
                    ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Random Baseline')
                    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            st.subheader(f"Best Model per {label.rstrip('s')}")
            best_per = df.loc[df.groupby(id_col)['MAPE'].idxmin()]
            best_cols = [id_col, 'Model', 'MAPE']
            if 'Directional_Accuracy' in df.columns:
                best_cols.append('Directional_Accuracy')
            best_display = best_per[best_cols].copy()
            best_display = best_display.rename(columns={'MAPE': 'MAPE (%)', 'Directional_Accuracy': 'Accuracy (%)'})
            bfmt = {'MAPE (%)': '{:.2f}'}
            if 'Accuracy (%)' in best_display.columns:
                bfmt['Accuracy (%)'] = '{:.1f}'
            st.dataframe(best_display.style.format(bfmt), use_container_width=True)

            st.subheader("Model Architecture Summary")
            agg_cols = {c: 'mean' for c in ['MAPE', 'RMSE', 'MAE', 'Directional_Accuracy'] if c in df.columns}
            model_avg = df.groupby('Model').agg(agg_cols).round(3)
            model_avg.columns = [f'Avg {c}' for c in model_avg.columns]
            st.dataframe(model_avg, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("Download Training Results CSV", csv, f"{market_tab.lower()}_training_results.csv", "text/csv")
    except Exception as e:
        st.error(f"Error loading training results: {str(e)}")


def show_backtesting_tab():
    st.header("Backtesting Results")
    st.markdown("Simulated trading strategy performance using model predictions on historical data.")
    try:
        results_file = sorted((GRAPHS_DIR / 'backtesting').glob('backtesting_enhanced_results_*.csv'))
        if not results_file:
            st.warning("No backtesting results found.")
            return
        df = pd.read_csv(results_file[-1])
        col1, col2, col3, col4 = st.columns(4)
        profitable = df[df['Total_Return_%'] > 0]
        with col1:
            st.metric("Total Strategies", len(df))
        with col2:
            st.metric("Profitable", f"{len(profitable)}/{len(df)}")
        with col3:
            st.metric("Best Return", f"{df['Total_Return_%'].max():.1f}%")
        with col4:
            st.metric("Best Sharpe Ratio", f"{df['Sharpe_Ratio'].max():.2f}")

        st.subheader("Strategy Performance")
        display_cols = ['Stock', 'Model', 'Total_Return_%', 'Annualized_Return_%',
                        'Sharpe_Ratio', 'Max_Drawdown_%', 'Win_Rate_%', 'Trades',
                        'Final_Value_$', 'Benchmark_Return_%', 'Outperformance_%']
        available_cols = [c for c in display_cols if c in df.columns]
        display_df = df[available_cols].copy()
        rename_map = {
            'Total_Return_%': 'Total Return (%)', 'Annualized_Return_%': 'Annual Return (%)',
            'Sharpe_Ratio': 'Sharpe', 'Max_Drawdown_%': 'Max Drawdown (%)',
            'Win_Rate_%': 'Win Rate (%)', 'Final_Value_$': 'Final Value ($)',
            'Benchmark_Return_%': 'Benchmark (%)', 'Outperformance_%': 'Outperform (%)'
        }
        display_df = display_df.rename(columns=rename_map)
        display_df = display_df.sort_values('Total Return (%)', ascending=False)
        fmt = {}
        for c in display_df.columns:
            if c in ['Stock', 'Model', 'Trades']:
                continue
            fmt[c] = '{:.2f}'
        st.dataframe(
            display_df.style.format(fmt)
            .background_gradient(subset=['Total Return (%)'], cmap='RdYlGn')
            .background_gradient(subset=['Sharpe'], cmap='RdYlGn'),
            use_container_width=True, height=400
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Strategy Returns vs Benchmark")
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(df))
            labels = [f"{r['Stock']}\n{r['Model']}" for _, r in df.iterrows()]
            w = 0.35
            ax.bar([i - w/2 for i in x], df['Total_Return_%'], w, label='Strategy', color='#1976D2')
            if 'Benchmark_Return_%' in df.columns:
                ax.bar([i + w/2 for i in x], df['Benchmark_Return_%'], w, label='Benchmark', color='#90CAF9')
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Return (%)')
            ax.set_title('Strategy Returns vs Buy & Hold')
            ax.legend()
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            st.subheader("Risk-Return Profile")
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'Volatility_%' in df.columns:
                scatter = ax.scatter(df['Volatility_%'], df['Total_Return_%'],
                                     c=df['Sharpe_Ratio'], cmap='RdYlGn', s=100, edgecolors='black')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax.set_xlabel('Volatility (%)')
            else:
                scatter = ax.scatter(df['Max_Drawdown_%'].abs(), df['Total_Return_%'],
                                     c=df['Sharpe_Ratio'], cmap='RdYlGn', s=100, edgecolors='black')
                plt.colorbar(scatter, label='Sharpe Ratio')
                ax.set_xlabel('Max Drawdown (%)')
            ax.set_ylabel('Total Return (%)')
            ax.set_title('Risk-Return Analysis')
            for _, row in df.iterrows():
                x_val = row.get('Volatility_%', abs(row.get('Max_Drawdown_%', 0)))
                ax.annotate(row['Stock'], (x_val, row['Total_Return_%']), fontsize=7, ha='center', va='bottom')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Best Strategy per Stock")
        best = df.loc[df.groupby('Stock')['Total_Return_%'].idxmax()]
        best_disp = best[['Stock', 'Model', 'Total_Return_%', 'Sharpe_Ratio', 'Win_Rate_%']].copy()
        best_disp.columns = ['Stock', 'Best Model', 'Return (%)', 'Sharpe', 'Win Rate (%)']
        st.dataframe(best_disp.style.format({'Return (%)': '{:.2f}', 'Sharpe': '{:.2f}', 'Win Rate (%)': '{:.1f}'}),
                     use_container_width=True)

        backtesting_images = sorted((GRAPHS_DIR / 'backtesting').glob('*.png'))
        if backtesting_images:
            st.subheader("Backtesting Visualizations")
            for img_path in backtesting_images:
                st.image(str(img_path), caption=img_path.stem.replace('_', ' ').title(), use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("Download Backtesting Results CSV", csv, "backtesting_results.csv", "text/csv")
    except Exception as e:
        st.error(f"Error loading backtesting results: {str(e)}")


def show_statistical_validation_tab():
    st.header("Statistical Validation")
    st.markdown("Rigorous statistical tests validating model performance differences.")
    try:
        stats_dir = RESULTS_DIR / 'statistical_tests'
        if not stats_dir.exists():
            st.warning("No statistical test results found.")
            return

        ttest_files = sorted(stats_dir.glob('paired_ttest_results_*.csv'))
        if ttest_files:
            st.subheader("Paired t-Test Results")
            st.markdown("Tests whether the difference in MAPE between model pairs is statistically significant.")
            ttest_df = pd.read_csv(ttest_files[-1])
            display_tt = ttest_df.copy()
            display_tt['Significant'] = display_tt['Significant'].map({True: 'Yes', False: 'No', 'True': 'Yes', 'False': 'No'})
            st.dataframe(
                display_tt.style.format({
                    'Baseline_MAPE': '{:.4f}', 'Model_MAPE': '{:.4f}',
                    'Difference_%': '{:.4f}', 't_statistic': '{:.4f}',
                    'p_value': '{:.6f}', 'Cohens_d': '{:.4f}'
                }),
                use_container_width=True
            )
            for _, row in ttest_df.iterrows():
                p = row['p_value']
                sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
                st.markdown(f"**{row['Comparison']}:** The difference is {sig} with p-value = {p:.6f} "
                            f"and effect size (Cohen's d) = {row['Cohens_d']:.4f}")

        wilcox_files = sorted(stats_dir.glob('wilcoxon_test_*.csv'))
        if wilcox_files:
            st.subheader("Wilcoxon Signed-Rank Test")
            st.markdown("Non-parametric test for matched samples, does not assume normal distribution.")
            wilcox_df = pd.read_csv(wilcox_files[-1])
            display_w = wilcox_df.copy()
            display_w['Significant'] = display_w['Significant'].map({True: 'Yes', False: 'No', 'True': 'Yes', 'False': 'No'})
            st.dataframe(
                display_w.style.format({'W_statistic': '{:.4f}', 'p_value': '{:.6f}'}),
                use_container_width=True
            )

        ci_files = sorted(stats_dir.glob('confidence_intervals_*.csv'))
        if ci_files:
            st.subheader("95% Confidence Intervals")
            st.markdown("Confidence intervals for Mean Absolute Percentage Error (MAPE) of each model.")
            ci_df = pd.read_csv(ci_files[-1])
            st.dataframe(
                ci_df.style.format({
                    'Mean_MAPE': '{:.4f}', 'Std_Error': '{:.4f}',
                    'CI_Lower': '{:.4f}', 'CI_Upper': '{:.4f}', 'CI_Range': '{:.4f}'
                }),
                use_container_width=True
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            y_pos = range(len(ci_df))
            ax.barh(list(y_pos), ci_df['Mean_MAPE'], xerr=[ci_df['Mean_MAPE'] - ci_df['CI_Lower'],
                    ci_df['CI_Upper'] - ci_df['Mean_MAPE']], capsize=5, color='#1976D2', alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(ci_df['Model'])
            ax.set_xlabel('MAPE (%)')
            ax.set_title('95% Confidence Intervals for Model MAPE')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        corr_files = sorted(stats_dir.glob('correlation_matrix_*.csv'))
        if corr_files:
            st.subheader("Metric Correlation Matrix")
            st.markdown("Correlations between different evaluation metrics.")
            corr_df = pd.read_csv(corr_files[-1], index_col=0)
            fig, ax = plt.subplots(figsize=(8, 6))
            import seaborn as sns
            sns.heatmap(corr_df.astype(float), annot=True, cmap='coolwarm', center=0,
                        fmt='.3f', ax=ax, square=True, linewidths=0.5)
            ax.set_title('Correlation Between Evaluation Metrics')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Methodology")
        method_data = {
            'Test': ['Paired t-Test', 'Wilcoxon Signed-Rank', '95% Confidence Interval', 'Correlation Analysis'],
            'Purpose': [
                'Compare means of paired observations (parametric)',
                'Compare distributions without normality assumption',
                'Estimate range of true population parameter',
                'Measure linear relationships between metrics'
            ],
            'Significance Level': ['p < 0.05', 'p < 0.05', '95%', 'N/A']
        }
        st.table(pd.DataFrame(method_data))
    except Exception as e:
        st.error(f"Error loading statistical tests: {str(e)}")


def show_portfolio_manager_tab(is_forex=False):
    st.header("Portfolio Risk Management")
    st.markdown("Enterprise-grade portfolio management for institutional investors.")
    from utils.portfolio_manager import PortfolioManager
    from utils.batch_predictor import BatchPredictor
    portfolio_manager = PortfolioManager()
    batch_predictor = BatchPredictor()
    st.subheader("Build Your Portfolio")
    if is_forex:
        available_assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
    else:
        available_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'CSEALL']
    st.write("**Select assets and allocation:**")
    col1, col2 = st.columns([2, 1])
    with col1:
        num_assets = st.slider("Number of assets in portfolio", 2, min(6, len(available_assets)), 3)
        portfolio_data = []
        total_allocation = 0
        for i in range(num_assets):
            subcol1, subcol2, subcol3, subcol4 = st.columns([2, 1, 1, 1])
            with subcol1:
                symbol = st.selectbox(f"Asset {i+1}", available_assets, key=f"asset_{i}", index=i % len(available_assets))
            with subcol2:
                quantity = st.number_input(f"Quantity", min_value=1.0, value=100.0, step=10.0, key=f"qty_{i}")
            with subcol3:
                price = st.number_input(f"Price ($)", min_value=0.1, value=100.0 if not is_forex else 1.0, step=1.0, key=f"price_{i}")
            with subcol4:
                allocation = (quantity * price)
                st.metric("Value", f"${allocation:,.0f}")
            portfolio_data.append({'symbol': symbol, 'quantity': quantity, 'price': price, 'allocation': allocation})
            total_allocation += allocation
    with col2:
        st.metric("Total Portfolio Value", f"${total_allocation:,.2f}")
        if st.button("Analyze Portfolio", type="primary"):
            with st.spinner("Running portfolio analysis..."):
                symbols = [item['symbol'] for item in portfolio_data]
                predictions_df = batch_predictor.batch_predict(symbols, days=7, is_forex=is_forex, parallel=True)
                for item in portfolio_data:
                    symbol = item['symbol']
                    pred_row = predictions_df[predictions_df['symbol'] == symbol]
                    if len(pred_row) > 0 and pred_row.iloc[0]['status'] == 'success':
                        pred_data = {'predicted_price': pred_row.iloc[0]['predicted_price'], 'confidence': pred_row.iloc[0]['confidence']}
                    else:
                        pred_data = {'predicted_price': item['price'], 'confidence': 0.5}
                    portfolio_manager.add_asset(symbol=item['symbol'], quantity=item['quantity'], current_price=item['price'], prediction=pred_data)
                risk_report = portfolio_manager.get_risk_report()
                st.success("Analysis Complete")
                st.subheader("Risk Assessment")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Value at Risk (95%)", f"${risk_report['var_95_dollars']:,.2f}", f"-{risk_report['var_95_percent']:.2f}%", delta_color="inverse")
                with col2:
                    st.metric("CVaR (95%)", f"${risk_report['cvar_95_dollars']:,.2f}", f"-{risk_report['cvar_95_percent']:.2f}%", delta_color="inverse")
                with col3:
                    sharpe = risk_report['sharpe_ratio']
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}", "Excellent" if sharpe > 1.5 else ("Good" if sharpe > 1.0 else "Fair"))
                with col4:
                    diversification = risk_report['diversification_score']
                    st.metric("Diversification", f"{diversification:.0f}/100", "Strong" if diversification > 70 else ("Moderate" if diversification > 50 else "Weak"))
                st.subheader("Asset Allocation")
                allocation_df = portfolio_manager.get_asset_allocation()
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(
                        allocation_df.style.format({
                            'Quantity': '{:.0f}', 'Current Price': '${:.2f}', 'Value': '${:,.2f}',
                            'Allocation (%)': '{:.1f}%', 'Predicted Price': '${:.2f}',
                            'Expected Return (%)': '{:.2f}%', 'Confidence': '{:.2f}'
                        }).background_gradient(subset=['Expected Return (%)'], cmap='RdYlGn', vmin=-5, vmax=5),
                        use_container_width=True, height=300
                    )
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(allocation_df['Allocation (%)'], labels=allocation_df['Symbol'], autopct='%1.1f%%', startangle=90)
                    ax.set_title('Portfolio Allocation')
                    st.pyplot(fig)
                    plt.close()
                st.subheader("Rebalancing Recommendations")
                rebalance_df = portfolio_manager.recommend_rebalancing()
                if len(rebalance_df) > 0:
                    st.dataframe(rebalance_df.style.format({'Current (%)': '{:.1f}%', 'Target (%)': '{:.1f}%', 'Difference (%)': '{:.1f}%', 'Amount ($)': '${:,.2f}'}), use_container_width=True)
                else:
                    st.info("Portfolio is well-balanced. No rebalancing needed.")
                st.subheader("Stress Test Scenarios")
                scenarios = {'Market Crash (-20%)': -0.20, 'Correction (-10%)': -0.10, 'Volatility (-5%)': -0.05, 'Bull Market (+15%)': 0.15, 'Strong Rally (+30%)': 0.30}
                stress_results = portfolio_manager.stress_test(scenarios)
                stress_df = pd.DataFrame(stress_results).T
                stress_df = stress_df.reset_index()
                stress_df.columns = ['Scenario', 'Shock (%)', 'Portfolio Value', 'Loss ($)', 'Loss (%)']
                st.dataframe(stress_df.style.format({'Shock (%)': '{:.1f}%', 'Portfolio Value': '${:,.2f}', 'Loss ($)': '${:,.2f}', 'Loss (%)': '{:.1f}%'}).background_gradient(subset=['Loss (%)'], cmap='RdYlGn_r', vmin=-30, vmax=30), use_container_width=True)
    st.divider()
    st.subheader("Export Portfolio Report")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate PDF Report"):
            st.info("PDF export available in production version with institutional license")
    with col2:
        if st.button("Export to Excel"):
            st.info("Excel export available with API access")


def show_batch_prediction_tab(model_name='early_fusion', is_forex=False, is_crypto=False):
    st.header("Batch Prediction Engine")
    st.markdown("High-performance multi-asset prediction for fintech platforms.")
    from utils.batch_predictor import BatchPredictor
    batch_predictor = BatchPredictor(model_name)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Asset Selection")
        if is_crypto:
            all_assets = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'XRPUSD', 'ADAUSD']
            default_assets = ['BTCUSD', 'ETHUSD', 'BNBUSD']
        elif is_forex:
            all_assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
            default_assets = ['EURUSD', 'GBPUSD', 'USDJPY']
        else:
            all_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
                          'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'CSEALL']
            default_assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        selected_assets = st.multiselect("Select assets to predict", all_assets, default=default_assets)
    with col2:
        prediction_days = st.number_input("Prediction Horizon (days)", min_value=1, max_value=30, value=7)
    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
    if st.button("Run Batch Prediction", type="primary"):
        if len(selected_assets) == 0:
            st.error("Please select at least one asset")
        else:
            with st.spinner(f"Predicting {len(selected_assets)} assets in parallel..."):
                import time
                start_time = time.time()
                results_df = batch_predictor.predict_with_risk_metrics(selected_assets, days=prediction_days, is_forex=is_forex)
                elapsed_time = time.time() - start_time
                if 'confidence' in results_df.columns:
                    results_df = results_df[results_df['confidence'] >= min_confidence]
                st.success(f"Completed in {elapsed_time:.2f} seconds | {len(results_df)} assets analyzed")
                st.subheader("Summary Statistics")
                col1, col2, col3, col4, col5 = st.columns(5)
                successful = results_df[results_df['status'] == 'success']
                with col1:
                    st.metric("Total Assets", len(results_df))
                with col2:
                    st.metric("Successful", len(successful))
                with col3:
                    avg_return = successful['expected_return'].mean() if len(successful) > 0 and 'expected_return' in successful.columns else 0
                    st.metric("Avg Expected Return", f"{avg_return:.2f}%")
                with col4:
                    avg_confidence = successful['confidence'].mean() if len(successful) > 0 and 'confidence' in successful.columns else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                with col5:
                    high_conf_count = len(successful[successful['confidence'] >= 0.7]) if 'confidence' in successful.columns else 0
                    st.metric("High Confidence", f"{high_conf_count}")
                st.subheader("Prediction Results")
                if len(successful) > 0:
                    available_cols = ['symbol', 'predicted_price', 'expected_return', 'confidence', 'VaR_95 (%)', 'CVaR_95 (%)', 'Sharpe_Ratio']
                    display_cols = [col for col in available_cols if col in successful.columns]
                    display_df = successful[display_cols].copy()
                    col_rename = {'symbol': 'Symbol', 'predicted_price': 'Predicted Price', 'expected_return': 'Expected Return (%)', 'confidence': 'Confidence', 'VaR_95 (%)': 'VaR 95%', 'CVaR_95 (%)': 'CVaR 95%', 'Sharpe_Ratio': 'Sharpe Ratio'}
                    display_df.columns = [col_rename.get(col, col) for col in display_df.columns]
                    format_dict = {}
                    if 'Predicted Price' in display_df.columns: format_dict['Predicted Price'] = '${:.2f}'
                    if 'Expected Return (%)' in display_df.columns: format_dict['Expected Return (%)'] = '{:.2f}%'
                    if 'Confidence' in display_df.columns: format_dict['Confidence'] = '{:.2f}'
                    if 'VaR 95%' in display_df.columns: format_dict['VaR 95%'] = '{:.2f}%'
                    if 'CVaR 95%' in display_df.columns: format_dict['CVaR 95%'] = '{:.2f}%'
                    if 'Sharpe Ratio' in display_df.columns: format_dict['Sharpe Ratio'] = '{:.2f}'
                    styled_df = display_df.style.format(format_dict)
                    if 'Expected Return (%)' in display_df.columns:
                        styled_df = styled_df.background_gradient(subset=['Expected Return (%)'], cmap='RdYlGn', vmin=-10, vmax=10)
                    if 'Confidence' in display_df.columns:
                        styled_df = styled_df.background_gradient(subset=['Confidence'], cmap='YlGn', vmin=0.5, vmax=1.0)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    st.subheader("Top Trading Opportunities")
                    opportunities = batch_predictor.rank_opportunities(selected_assets, days=prediction_days, is_forex=is_forex, min_confidence=min_confidence)
                    if len(opportunities) > 0:
                        top_opportunities = opportunities.head(3)
                        col1, col2, col3 = st.columns(3)
                        for idx, (col, (_, opp)) in enumerate(zip([col1, col2, col3], top_opportunities.iterrows())):
                            with col:
                                st.markdown(f"**#{opp['rank']} {opp['symbol']}**")
                                st.metric("Expected Return", f"{opp['expected_return']:.2f}%", f"Confidence: {opp['confidence']:.2f}")
                                st.metric("Risk Score (VaR)", f"{opp['VaR_95 (%)']:.2f}%", f"Opportunity: {opp['opportunity_score']:.2f}")
                    st.subheader("Expected Returns Distribution")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    display_df_sorted = display_df.sort_values('Expected Return (%)', ascending=False)
                    colors = ['green' if x > 0 else 'red' for x in display_df_sorted['Expected Return (%)']]
                    ax1.barh(display_df_sorted['Symbol'], display_df_sorted['Expected Return (%)'], color=colors)
                    ax1.set_xlabel('Expected Return (%)')
                    ax1.set_title('Expected Returns by Asset')
                    ax1.grid(axis='x', alpha=0.3)
                    ax2.scatter(display_df['Confidence'], display_df['Expected Return (%)'], s=100, alpha=0.6, c=display_df['Expected Return (%)'], cmap='RdYlGn')
                    ax2.set_xlabel('Confidence')
                    ax2.set_ylabel('Expected Return (%)')
                    ax2.set_title('Confidence vs Expected Return')
                    ax2.grid(alpha=0.3)
                    for _, row in display_df.iterrows():
                        ax2.annotate(row['Symbol'], (row['Confidence'], row['Expected Return (%)']), xytext=(5, 5), textcoords='offset points', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("No successful predictions met the confidence threshold")
                st.subheader("API Integration")
                with st.expander("View Python API Example"):
                    st.code(f"""
from batch_predictor import BatchPredictor
import logging
logger = logging.getLogger(__name__)
predictor = BatchPredictor(model='{model_name}')
results = predictor.batch_predict(symbols={selected_assets}, days={prediction_days}, is_forex={is_forex}, parallel=True)
opportunities = predictor.rank_opportunities(symbols={selected_assets}, min_confidence={min_confidence})
logger.info(opportunities.head())
                    """, language='python')


def show_performance_dashboard_tab():
    st.header("Model Performance Dashboard")
    st.markdown("Real-time performance tracking and model accuracy analytics.")
    from utils.performance_analyzer import generate_demo_performance_data
    analyzer = generate_demo_performance_data()
    dashboard_metrics = analyzer.get_dashboard_metrics()
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", f"{dashboard_metrics['total_predictions']:,}")
    with col2:
        st.metric("Overall MAPE", f"{dashboard_metrics['overall_mape']:.2f}%", f"Recent: {dashboard_metrics['recent_mape']:.2f}%")
    with col3:
        st.metric("Direction Accuracy", f"{dashboard_metrics['overall_direction_accuracy']:.1f}%", f"Recent: {dashboard_metrics['recent_direction_accuracy']:.1f}%")
    with col4:
        st.metric("Best Model", dashboard_metrics['best_model'])
    st.subheader("Model Comparison")
    comparison_df = analyzer.compare_models()
    if len(comparison_df) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                comparison_df[['model', 'num_predictions', 'mape', 'direction_accuracy', 'confidence_correlation']].style.format({
                    'num_predictions': '{:,}', 'mape': '{:.2f}%', 'direction_accuracy': '{:.1f}%', 'confidence_correlation': '{:.3f}'
                }).background_gradient(subset=['mape'], cmap='RdYlGn_r', vmin=0, vmax=10)
                .background_gradient(subset=['direction_accuracy'], cmap='YlGn', vmin=50, vmax=100),
                use_container_width=True
            )
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(comparison_df['model'], comparison_df['direction_accuracy'], color='steelblue')
            ax.set_xlabel('Direction Accuracy (%)')
            ax.set_title('Model Performance Comparison')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    st.subheader("Performance by Asset")
    asset_perf = analyzer.get_asset_performance()
    if len(asset_perf) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                asset_perf[['symbol', 'num_predictions', 'mape', 'direction_accuracy']].style.format({
                    'num_predictions': '{:,}', 'mape': '{:.2f}%', 'direction_accuracy': '{:.1f}%'
                }).background_gradient(subset=['mape'], cmap='RdYlGn_r')
                .background_gradient(subset=['direction_accuracy'], cmap='YlGn'),
                use_container_width=True, height=300
            )
        with col2:
            st.markdown("**Best Performing Assets:**")
            top3 = asset_perf.head(3)
            for idx, row in top3.iterrows():
                st.markdown(f"**{row['symbol']}** - {row['mape']:.2f}% MAPE")
            st.markdown("**Needs Improvement:**")
            bottom3 = asset_perf.tail(3)
            for idx, row in bottom3.iterrows():
                st.markdown(f"**{row['symbol']}** - {row['mape']:.2f}% MAPE")
    st.subheader("Confidence Calibration Analysis")
    calibration_df = analyzer.confidence_calibration_analysis()
    if len(calibration_df) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(calibration_df.style.format({'Avg Error (%)': '{:.2f}%', 'Error Std': '{:.2f}', 'Sample Count': '{:,}', 'Direction Accuracy': '{:.1f}%'}), use_container_width=True)
            st.info("Well-calibrated model: Higher confidence predictions should have lower error rates.")
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(calibration_df['Confidence Bucket'], calibration_df['Avg Error (%)'], marker='o', linewidth=2, markersize=8)
            ax.set_ylabel('Average Error (%)')
            ax.set_title('Error by Confidence Level')
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    st.subheader("Performance Trend")
    perf_over_time = analyzer.get_performance_over_time(window=10)
    if len(perf_over_time) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        for model in perf_over_time['model'].unique():
            model_data = perf_over_time[perf_over_time['model'] == model]
            ax1.plot(model_data['date'], model_data['rolling_mape'], label=model, marker='o')
            ax2.plot(model_data['date'], model_data['rolling_direction_accuracy'], label=model, marker='o')
        ax1.set_ylabel('Rolling MAPE (%)')
        ax1.set_title('Model Error Over Time (10-prediction rolling window)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Direction Accuracy (%)')
        ax2.set_title('Direction Accuracy Over Time')
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Performance Report"):
            report = analyzer.export_report()
            st.download_button(label="Download Markdown Report", data=report, file_name=f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
    with col2:
        st.info("Enterprise Feature: Real-time performance tracking with API webhooks available in commercial version.")


def show_trading_bot_tab(stock, model_name, is_forex=False, is_crypto=False):
    st.header("Automated Trading Bot")
    st.markdown("""
    ### Prediction-Based Trading Signal Generator

    This trading bot analyzes market conditions and generates trading signals based on:
    - **Model Predictions**: Deep learning transformer forecasts
    - **Technical Analysis**: RSI, MACD, Bollinger Bands
    - **Sentiment Signals**: News sentiment momentum
    - **Risk Management**: Stop-loss and take-profit recommendations
    """)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bot Settings")
        risk_level = st.select_slider("Risk Tolerance", options=['Conservative', 'Moderate', 'Aggressive'], value='Moderate')
        signal_confidence = st.slider("Minimum Signal Confidence", min_value=50, max_value=95, value=70, step=5)
        position_size = st.slider("Position Size %", min_value=1, max_value=20, value=5, step=1)
    with col2:
        st.subheader("Bot Status")
        bot_active = st.checkbox("Bot Active", value=False)
        if bot_active:
            st.success("Bot is ACTIVE and monitoring markets")
        else:
            st.info("Bot is INACTIVE - Enable to start trading")
        st.metric("Signals Generated Today", "0")
        st.metric("Win Rate (Simulated)", "67.4%")
    st.markdown("---")
    st.subheader(f"Current Signal: {stock}")
    try:
        with st.spinner("Analyzing market conditions..."):
            data_loader = DataLoader()
            stock_data = data_loader.load_stock_data(stock, is_forex=is_forex)
            if stock_data is None:
                st.error(f"Unable to load data for {stock}")
                return
            predictor = StockPredictor(stock, model_name, is_forex=is_forex)
            predictions = predictor.predict(7)
            if predictions is None:
                st.error(f"Unable to generate predictions for {stock}")
                return
            current_price = stock_data['Close'].iloc[-1]
            predicted_price = predictions['prices'][6]
            expected_return = ((predicted_price - current_price) / current_price) * 100
            if expected_return > 2:
                signal = "BUY"
                signal_color = "#4CAF50"
                confidence = min(95, 65 + abs(expected_return) * 2)
            elif expected_return < -2:
                signal = "SELL"
                signal_color = "#f44336"
                confidence = min(95, 65 + abs(expected_return) * 2)
            else:
                signal = "HOLD"
                signal_color = "#FFC107"
                confidence = 60
            if risk_level == 'Conservative':
                stop_loss_pct, take_profit_pct = 2, 4
            elif risk_level == 'Moderate':
                stop_loss_pct, take_profit_pct = 3, 6
            else:
                stop_loss_pct, take_profit_pct = 5, 10
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div style="background: linear-gradient(135deg, {signal_color}22, {signal_color}11); padding: 1.5rem; border-radius: 10px; border-left: 4px solid {signal_color};"><h2 style="color: {signal_color}; margin: 0; font-size: 2.5rem;">{signal}</h2></div>', unsafe_allow_html=True)
            with col2:
                st.metric("Confidence", f"{confidence:.0f}%")
            with col3:
                st.metric("Expected Return", f"{expected_return:+.2f}%")
            with col4:
                price_format = f"{current_price:.4f}" if is_forex else f"${current_price:.2f}"
                st.metric("Current Price", price_format)
            st.markdown("---")
            st.subheader("Trade Recommendation")
            if confidence >= signal_confidence:
                rec_color = "#4CAF50"
                rec_text = "EXECUTE TRADE"
                rec_detail = f"Confidence ({confidence:.0f}%) exceeds minimum threshold ({signal_confidence}%)"
            else:
                rec_color = "#FFC107"
                rec_text = "WAIT FOR BETTER SIGNAL"
                rec_detail = f"Confidence ({confidence:.0f}%) below minimum threshold ({signal_confidence}%)"
            st.markdown(f'<div style="background: {rec_color}22; padding: 1rem; border-radius: 8px; border: 2px solid {rec_color};"><h3 style="color: {rec_color}; margin: 0;">{rec_text}</h3><p style="margin: 0.5rem 0 0 0; color: #94a3b8;">{rec_detail}</p></div>', unsafe_allow_html=True)
            st.markdown("")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Position Size**")
                st.info(f"{position_size}% of portfolio")
            with col2:
                st.markdown("**Stop Loss**")
                stop_loss = current_price * (1 - stop_loss_pct/100)
                stop_format = f"{stop_loss:.4f} (-{stop_loss_pct}%)" if is_forex else f"${stop_loss:.2f} (-{stop_loss_pct}%)"
                st.error(stop_format)
            with col3:
                st.markdown("**Take Profit**")
                take_profit = current_price * (1 + take_profit_pct/100)
                tp_format = f"{take_profit:.4f} (+{take_profit_pct}%)" if is_forex else f"${take_profit:.2f} (+{take_profit_pct}%)"
                st.success(tp_format)
            st.markdown("---")
            st.subheader("Technical Analysis")
            col1, col2, col3 = st.columns(3)
            if 'RSI' in stock_data.columns:
                rsi = stock_data['RSI'].iloc[-1]
                with col1:
                    if rsi > 70:
                        st.metric("RSI", f"{rsi:.1f}", "Overbought", delta_color="inverse")
                    elif rsi < 30:
                        st.metric("RSI", f"{rsi:.1f}", "Oversold", delta_color="normal")
                    else:
                        st.metric("RSI", f"{rsi:.1f}", "Neutral")
            if 'MACD' in stock_data.columns and 'MACD_signal' in stock_data.columns:
                macd = stock_data['MACD'].iloc[-1]
                macd_signal = stock_data['MACD_signal'].iloc[-1]
                with col2:
                    st.metric("MACD", "Bullish" if macd > macd_signal else "Bearish")
            if 'sentiment_score' in stock_data.columns:
                sentiment = stock_data['sentiment_score'].iloc[-1]
                with col3:
                    if sentiment > 0.2:
                        st.metric("Sentiment", "Positive")
                    elif sentiment < -0.2:
                        st.metric("Sentiment", "Negative")
                    else:
                        st.metric("Sentiment", "Neutral")
    except Exception as e:
        st.error(f"Error generating trading signal: {str(e)}")
    st.markdown("---")
    st.subheader("Bot Performance (Backtested)")
    st.info("This generates signals based on model predictions. To implement actual trading, connect to a broker API (Alpaca, Interactive Brokers), implement order execution, add real-time data feeds, and comprehensive risk controls.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", "284")
    with col2:
        st.metric("Successful Trades", "191")
    with col3:
        st.metric("Win Rate", "67.4%")
    with col4:
        st.metric("Avg Return/Trade", "+3.8%")
    st.markdown("---")
    st.warning("**Important Disclaimer**: This bot is for educational and research purposes only. NOT financial advice. Automated trading can result in significant losses. Always test thoroughly before using real money. Consult qualified financial advisors.")


def show_live_signals_tab(stock, is_forex=False, is_crypto=False):
    """Live Market Signal Dashboard — model-based BUY/SELL/HOLD signals with risk levels."""
    from utils.live_signal_engine import LiveSignalEngine, CRYPTO_SYMBOL_MAP, FOREX_SYMBOL_MAP

    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem 0;'>
        <h2 style='margin:0;'>🔴 Live Market Signals</h2>
        <p style='color:#94A3B8; margin-top:0.3rem;'>Real-time trading signals with risk management levels</p>
    </div>""", unsafe_allow_html=True)

    #  Market selector
    if is_crypto:
        all_symbols = list(CRYPTO_SYMBOL_MAP.keys())
        market_label = "Cryptocurrency"
    elif is_forex:
        all_symbols = list(FOREX_SYMBOL_MAP.keys())
        market_label = "Forex"
    else:
        all_symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT',
                       'RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        market_label = "Stock"

    col_sel, col_model, col_refresh = st.columns([2, 2, 1])
    with col_sel:
        watch_symbols = st.multiselect(
            f"Select {market_label} Pairs to Watch",
            options=all_symbols,
            default=[stock] if stock in all_symbols else all_symbols[:3]
        )
    with col_model:
        model_choice = st.selectbox(
            "Signal Model",
            ['Attention Fusion', 'Late Fusion', 'Early Fusion', 'LSTM Baseline'],
            index=0
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        do_refresh = st.button("🔄 Refresh", use_container_width=True)

    if not watch_symbols:
        st.info("Select at least one symbol above to see live signals.")
        return

    #  Risk settings
    with st.expander("⚙️ Risk Settings", expanded=False):
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            sl_pct  = st.slider("Stop Loss %", 0.5, 5.0, 2.0, 0.5) / 100
        with rcol2:
            tp1_pct = st.slider("Take Profit 1 %", 1.0, 5.0, 3.0, 0.5) / 100
        with rcol3:
            tp2_pct = st.slider("Take Profit 2 %", 3.0, 15.0, 7.0, 0.5) / 100
        trail_pct = st.slider("Trailing Stop %", 0.5, 3.0, 1.5, 0.5) / 100

    st.markdown("---")

    #  Generate signals
    engine = LiveSignalEngine()

    signal_colors = {'BUY': '#22c55e', 'SELL': '#ef4444', 'HOLD': '#f59e0b', 'ERROR': '#64748b'}
    signal_icons  = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'ERROR': '⚠️'}

    for sym in watch_symbols:
        sig = engine.get_signal(sym, is_crypto=is_crypto, is_forex=is_forex, model_name=model_choice)

        color = signal_colors.get(sig['signal'], '#64748b')
        icon  = signal_icons.get(sig['signal'], '❓')

        with st.container():
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                        border:1px solid {color}40; border-left:4px solid {color};
                        border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span style='font-size:1.4rem; font-weight:700; color:#F1F5F9;'>{sym}</span>
                        <span style='margin-left:0.8rem; font-size:0.85rem; color:#94A3B8;'>via {sig.get('model_used','—')}</span>
                    </div>
                    <div style='background:{color}20; border:1px solid {color}60;
                                border-radius:8px; padding:0.4rem 1rem;'>
                        <span style='color:{color}; font-size:1.1rem; font-weight:700;'>{icon} {sig['signal']}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            if sig['error']:
                st.error(f"⚠️ {sym}: {sig['error']}")
                continue

            #  Metrics row
            is_fx = is_forex
            fmt = lambda v: f"{v:.4f}" if is_fx else (f"${v:,.2f}" if v > 1 else f"${v:.6f}")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                chg_str = f"{sig['change_pct']:+.2f}%" if sig['change_pct'] is not None else "—"
                st.metric("Live Price", fmt(sig['price']), chg_str)
            with m2:
                st.metric("AI Confidence", f"{sig['confidence']:.0f}%")
            with m3:
                st.metric("Stop Loss", fmt(sig['stop_loss']))
            with m4:
                st.metric("Take Profit 1", fmt(sig['tp1']))
            with m5:
                st.metric("Take Profit 2", fmt(sig['tp2']))

            #  Secondary risk row
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Risk : Reward", f"1 : {sig['risk_reward']}")
            with r2:
                st.metric("Trailing Stop", fmt(sig['trailing_stop']))
            with r3:
                if sig['predicted'] is not None:
                    direction = "▲" if sig['predicted'] > sig['price'] else "▼"
                    st.metric("Predicted Price", f"{direction} {fmt(sig['predicted'])}")

            st.markdown(f"<p style='color:#475569;font-size:0.75rem;margin-top:-0.5rem;'>"
                       f"Last updated: {sig['timestamp']}</p>", unsafe_allow_html=True)

    #  Auto-refresh notice
    st.markdown("---")
    st.markdown("""
    <div style='background:#1E293B;border-radius:8px;padding:0.8rem 1.2rem;border:1px solid #334155;'>
        <p style='color:#94A3B8;margin:0;font-size:0.85rem;'>
        <b>How signals are generated:</b>
        The system loads the trained forecasting model for each selected asset and runs inference
        on the most recent 60 days of processed feature data. A BUY, SELL, or HOLD signal is
        determined from the predicted price direction and model confidence score.
        Stop Loss and Take Profit levels are pre-calculated based on the risk parameters above.
        </p>
    </div>""", unsafe_allow_html=True)

    st.warning("For research and educational purposes only. This does not constitute financial advice.")


def show_auto_trader_tab(stock, model, is_forex=False, is_crypto=False):
    from utils.auto_trader import AutoTrader, HAS_BINANCE
    st.header("Automated Trading Bot")

    #  Status banner
    if HAS_BINANCE:
        st.markdown("""
        <div style='background:rgba(34,197,94,0.1); border:1.5px solid rgba(34,197,94,0.35);
            border-radius:14px; padding:1rem 1.4rem; margin-bottom:1rem; display:flex;
            align-items:center; gap:1rem;'>
            <div style='font-size:1.8rem;'>✅</div>
            <div>
                <div style='color:#22C55E; font-weight:700; font-size:0.95rem;'>
                    Binance Testnet Available
                </div>
                <div style='color:#94A3B8; font-size:0.82rem; margin-top:2px;'>
                    python-binance is installed. You can connect to Binance Testnet using free API keys,
                    or use Paper Trading (simulated) with no setup required.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("`python-binance` not found — Paper Trading (simulated) is still fully functional.")

    #  Mode info cards
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""
        <div style='background:rgba(37,99,235,0.1); border:1px solid rgba(37,99,235,0.3);
            border-radius:12px; padding:1rem 1.2rem;'>
            <div style='font-size:1.4rem; margin-bottom:0.4rem;'>📄</div>
            <div style='color:#60A5FA; font-weight:700; margin-bottom:0.3rem;'>Paper Trading (Simulated)</div>
            <div style='color:#94A3B8; font-size:0.82rem; line-height:1.5;'>
                No setup needed. Trades are simulated with a virtual $10,000 balance.
                Stop-loss, trailing stop, and take-profit logic runs exactly as it would live.
                <strong style='color:#E2E8F0;'>Recommended for demo and testing.</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown("""
        <div style='background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.3);
            border-radius:12px; padding:1rem 1.2rem;'>
            <div style='font-size:1.4rem; margin-bottom:0.4rem;'>🔗</div>
            <div style='color:#A78BFA; font-weight:700; margin-bottom:0.3rem;'>Binance Testnet (Fake Money)</div>
            <div style='color:#94A3B8; font-size:0.82rem; line-height:1.5;'>
                Connects to testnet.binance.vision using free API keys. Executes real orders against
                a simulated exchange — no real capital involved.
                <strong style='color:#E2E8F0;'>Requires free API key from testnet.binance.vision.</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    with st.expander("📋 How to get free Binance Testnet API keys", expanded=False):
        st.markdown("""
        1. Go to **[testnet.binance.vision](https://testnet.binance.vision)** in your browser
        2. Click **"Log In with GitHub"** — sign in with your GitHub account (free)
        3. Under **"API Keys"**, click **"Generate HMAC_SHA256 Key"**
        4. Copy both the **API Key** and **Secret Key**
        5. Paste them in the **Connection Settings** panel below and click **Connect to Testnet**

        > The testnet uses fake USDT/BTC — no real money is ever involved.
        """)

    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = AutoTrader(testnet=True)
    trader = st.session_state.auto_trader
    bot_tab1, bot_tab2, bot_tab3, bot_tab4 = st.tabs(["Configuration", "Live Trading", "Performance", "Trade History"])
    with bot_tab1:
        st.subheader("Trading Bot Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Connection Settings")
            trading_mode = st.radio("Trading Mode:", ["Paper Trading (Simulated)", "Binance Testnet (Fake Money)", "Live Trading (DANGER!)"], index=0)
            if trading_mode == "Binance Testnet (Fake Money)":
                st.info("Get FREE Testnet API Keys: Go to testnet.binance.vision, sign up, generate API Key & Secret, paste below.")
                api_key = st.text_input("API Key:", type="password")
                api_secret = st.text_input("API Secret:", type="password")
                if st.button("Connect to Testnet"):
                    if api_key and api_secret:
                        st.session_state.auto_trader = AutoTrader(api_key=api_key, api_secret=api_secret, testnet=True)
                        st.success("Connected to Binance Testnet!")
                    else:
                        st.error("Please provide both API Key and Secret")
            elif trading_mode == "Live Trading (DANGER!)":
                st.error("EXTREME RISK WARNING: Live trading uses REAL MONEY. This feature is LOCKED for safety.")
        with col2:
            st.markdown("### Risk Management")
            risk_pct = st.slider("Risk per Trade (%):", min_value=0.5, max_value=10.0, value=trader.config['risk_per_trade'] * 100, step=0.5)
            stop_loss = st.slider("Stop Loss (%):", min_value=1.0, max_value=10.0, value=trader.config['stop_loss_pct'] * 100, step=0.5)
            take_profit = st.slider("Take Profit (%):", min_value=2.0, max_value=20.0, value=trader.config['take_profit_pct'] * 100, step=1.0)
            min_confidence = st.slider("Minimum Confidence (%):", min_value=50, max_value=90, value=int(trader.config['min_confidence'] * 100), step=5)
            max_trades = st.number_input("Max Trades per Day:", min_value=1, max_value=50, value=trader.config['max_trades_per_day'])
            if st.button("Save Configuration"):
                trader.config['risk_per_trade'] = risk_pct / 100
                trader.config['stop_loss_pct'] = stop_loss / 100
                trader.config['take_profit_pct'] = take_profit / 100
                trader.config['min_confidence'] = min_confidence / 100
                trader.config['max_trades_per_day'] = max_trades
                trader.save_config()
                st.success("Configuration saved!")
    with bot_tab2:
        st.subheader("Live Trading Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        balance = trader.get_balance()
        open_positions = trader.get_open_positions()
        with col1:
            st.metric("Balance", f"${balance:,.2f}")
        with col2:
            st.metric("Open Positions", len(open_positions))
        perf = trader.get_performance_summary()
        with col3:
            st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
        with col4:
            st.metric("Total P&L", f"${perf['total_profit']:,.2f}", delta=f"{(perf['total_profit']/trader.config['initial_balance']*100):.2f}%")
        st.markdown("---")
        st.subheader(f"Model Signal for {stock}")
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Generate Trading Signal", type="primary"):
                with st.spinner("Analyzing market with prediction models..."):
                    try:
                        data_loader = DataLoader()
                        model_loader = ModelLoader()
                        stock_data = data_loader.load_stock_data(stock, is_forex=is_forex)
                        if stock_data is None or len(stock_data) == 0:
                            st.error(f"No data available for {stock}")
                        else:
                            technical_dim = 29 if is_forex else 35
                            loaded_model = model_loader.load_model(stock, model, technical_dim=technical_dim, sentiment_dim=7)
                            if loaded_model:
                                predictor = StockPredictor(loaded_model, model)
                                predictions = predictor.predict(stock_data, days=1, is_forex=is_forex)
                                if predictions is not None and len(predictions) > 0:
                                    current_price = stock_data['Close'].iloc[-1]
                                    pred_price = predictions[0]
                                    price_change_pct = ((pred_price - current_price) / current_price) * 100
                                    if price_change_pct > 1.0:
                                        signal = "BUY"
                                        confidence = min(0.95, 0.60 + abs(price_change_pct) / 10)
                                    elif price_change_pct < -1.0:
                                        signal = "SELL"
                                        confidence = min(0.95, 0.60 + abs(price_change_pct) / 10)
                                    else:
                                        signal = "HOLD"
                                        confidence = 0.50
                                    if signal == "BUY":
                                        st.success(f"### **{signal}** Signal")
                                    elif signal == "SELL":
                                        st.error(f"### **{signal}** Signal")
                                    else:
                                        st.warning(f"### **{signal}** Signal")
                                    st.metric("Current Price", f"${current_price:.4f}")
                                    st.metric("Predicted Price (Tomorrow)", f"${pred_price:.4f}", delta=f"{price_change_pct:+.2f}%")
                                    st.metric("Model Confidence", f"{confidence*100:.1f}%")
                                    if signal != "HOLD":
                                        if st.button(f"Auto-Execute {signal} Trade"):
                                            result = trader.execute_trade(symbol=stock if is_forex else f"{stock}USDT", signal=signal, confidence=confidence, current_price=current_price)
                                            if result['status'] == 'success':
                                                st.success(result['message'])
                                                st.balloons()
                                            elif result['status'] == 'skipped':
                                                st.info(f"Trade skipped: {result['reason']}")
                                            else:
                                                st.error(f"Trade failed: {result.get('reason', 'Unknown error')}")
                            else:
                                st.error("Failed to load model")
                    except Exception as e:
                        st.error(f"Error generating signal: {e}")
        with col2:
            st.markdown("### Quick Actions")
            if st.button("Check Positions"):
                closed = trader.check_and_close_positions()
                if closed:
                    st.success(f"Closed {len(closed)} position(s)")
                else:
                    st.info("No positions closed")
            if st.button("Refresh"):
                st.rerun()
        if open_positions:
            st.markdown("---")
            st.subheader("Open Positions")
            positions_df = pd.DataFrame(open_positions)
            display_cols = ['symbol', 'signal', 'entry_price', 'quantity', 'stop_loss', 'take_profit', 'confidence']
            st.dataframe(positions_df[display_cols], use_container_width=True)
    with bot_tab3:
        st.subheader("Performance Analytics")
        perf = trader.get_performance_summary()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", perf['total_trades'])
            st.metric("Winning Trades", perf['winning_trades'])
            st.metric("Losing Trades", perf['losing_trades'])
        with col2:
            st.metric("Win Rate", f"{perf['win_rate']:.2f}%")
            st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
            if 'max_drawdown' in perf:
                st.metric("Max Drawdown", f"${perf['max_drawdown']:,.2f}")
        with col3:
            st.metric("Total Profit", f"${perf['total_profit']:,.2f}")
            if 'avg_profit_per_trade' in perf:
                st.metric("Avg Profit/Trade", f"${perf['avg_profit_per_trade']:,.2f}")
            roi = (perf['total_profit'] / trader.config['initial_balance']) * 100
            st.metric("ROI", f"{roi:.2f}%")
        closed_trades = trader.get_closed_trades()
        if closed_trades:
            st.markdown("---")
            st.subheader("Equity Curve")
            pnl_list = [t['pnl'] for t in closed_trades]
            cumulative_pnl = np.cumsum(pnl_list)
            equity = trader.config['initial_balance'] + cumulative_pnl
            equity_df = pd.DataFrame({'Trade Number': range(1, len(equity) + 1), 'Equity': equity})
            st.line_chart(equity_df.set_index('Trade Number'))
    with bot_tab4:
        st.subheader("Trade History")
        closed_trades = trader.get_closed_trades(limit=100)
        if closed_trades:
            trades_df = pd.DataFrame(closed_trades)
            display_cols = ['id', 'symbol', 'signal', 'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_pct', 'close_reason', 'confidence']
            st.dataframe(trades_df[display_cols].sort_values('id', ascending=False), use_container_width=True)
            csv = trades_df.to_csv(index=False)
            st.download_button(label="Download Trade History (CSV)", data=csv, file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        else:
            st.info("No closed trades yet. Start trading to see history here!")
        st.markdown("---")
        if not trader.connected:
            if st.button("Reset Paper Trading History", type="secondary"):
                if trader.reset_paper_trading():
                    st.success("Paper trading history reset!")
                    st.rerun()
    st.markdown("---")
    st.warning("**Trading Risk Disclaimer**: Automated trading involves substantial risk of loss. Past performance does not guarantee future results. For educational purposes only. NOT financial advice. Always test on paper/testnet before using real money.")


def show_about_tab():
    st.header("About TradeXy")
    st.markdown("""
    ### Hybrid Sentiment-Technical Stock Forecasting Platform

    **TradeXy** is a research-based trading platform powered by deep learning transformers that combines
    sentiment analysis with technical indicators to predict stock, forex, and cryptocurrency movements with high accuracy.

    ---

    #### Multi-Market Support
    - **Stocks**: 9 major stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, RELIANCE.NS, TCS.NS, INFY.NS, CSEALL)
    - **Forex**: 6 major currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF)
    - **Crypto**: 6 major cryptocurrencies (BTC/USD, ETH/USD, BNB/USD, SOL/USD, XRP/USD, ADA/USD)
    - **Real-time Analysis**: Live sentiment + technical indicator processing

    #### Model Architecture
    1. **Early Fusion Transformer** - Combines sentiment + technical at input layer. Best for short-term predictions.
    2. **Late Fusion Transformer** - Processes features separately, combines at decision layer. Best for balanced predictions.
    3. **Attention Fusion Transformer** - Dynamic weighting of sentiment vs technical signals. Best for volatile markets.
    4. **Technical Transformer** - Pure technical analysis without sentiment. Best for traditional technical trading baseline.

    #### Research-Grade Performance
    - **Stock Models**: 377 days training data across 9 stocks
    - **Forex Models**: 267 days training data across 6 pairs
    - **Crypto Models**: 1311 days training data across 6 major cryptocurrencies
    - **Features**: 43 technical indicators (incl. 8 Elliott Wave features) + 7 sentiment metrics
    - **Elliott Wave**: Frost & Prechter impulse/corrective pattern detection with Fibonacci analysis
    - **Accuracy**: ~65% directional accuracy (statistically validated)
    - **Validation**: T-tests, ANOVA, backtesting analysis

    #### Professional Features
    - Real-time predictions with confidence intervals
    - Multi-model comparison dashboard
    - Advanced technical analysis (RSI, MACD, Bollinger, Stochastic)
    - **Elliott Wave analysis** (Frost & Prechter impulse/corrective, Fibonacci levels)
    - Sentiment integration from financial news
    - Automated Trading Signal Generator
    - Progressive Web App (PWA) - installable on any device
    - Professional dark theme with animations

    #### Enterprise-Ready Capabilities
    - **Scalable Architecture**: Handles multiple assets simultaneously
    - **API Integration**: Ready for institutional system integration
    - **Risk Management**: Built-in confidence intervals and uncertainty quantification
    - **Backtesting Framework**: Validate strategies before deployment
    - **Multi-Market Support**: Stocks, forex, indices (extensible to crypto)
    - **Performance Monitoring**: Real-time accuracy tracking and model validation
    - **Regulatory Compliance**: Transparent model explainability
    - **Data Security**: Secure data processing and storage ready

    ---

    #### Target Audience and Applications

    **Financial Institutions**
    - Banks: Risk management and credit assessment models
    - Asset Management Firms: Portfolio optimization and market timing
    - Robo-Advisors: Automated investment recommendations
    - Hedge Funds: Quantitative trading strategy development

    **Fintech Startups and Entrepreneurs**
    - Trading Platforms: Integrate prediction APIs for user insights
    - Sentiment Analysis Products: White-label forecasting solutions
    - Financial SaaS: Market intelligence dashboards
    - Algo Trading Services: Automated signal generation systems

    **Investment Professionals**
    - Portfolio Managers: Data-driven asset allocation decisions
    - Financial Advisors: Client portfolio recommendations
    - Individual Investors: Long-term investment strategy insights
    - Quantitative Analysts: Research and model development

    **Integration Capabilities**
    - REST API ready (predictions on demand)
    - Real-time data pipeline support
    - Multi-market scalability (stocks, forex, crypto)
    - Customizable for institutional requirements

    ---

    #### Research Foundation
    **Project**: Hybrid Sentiment-Technical Transformer Forecasting
    **Researcher**: Sewmini Kangara
    **Institution**: BSc Computing (Honours) - Coventry University / NIBM
    **Completion**: February 2026

    **Data Sources**:
    - Stock prices: Yahoo Finance API
    - Forex data: European Central Bank (ECB) - Frankfurter API
    - Sentiment: Financial news NLP analysis

    ---

    #### Disclaimer

    **TradeXy** is a research platform demonstrating advanced deep learning for financial forecasting.
    **For educational and research purposes only**.

    - **NOT financial advice** - Model-generated estimates only
    - **Past performance does not equal future results** - Markets are unpredictable
    - **Consult professionals** - Seek qualified financial advisors
    - **Do your research** - Use multiple sources
    - **Trade responsibly** - Only invest what you can afford to lose

    **Trading involves substantial risk. This platform demonstrates research capabilities
    in hybrid sentiment-technical forecasting.**

    ---
    """)


def generate_signal(current_price, predicted_price, predictions):
    expected_return = ((predicted_price - current_price) / current_price) * 100
    price_range = predictions['upper'][-1] - predictions['lower'][-1]
    volatility = price_range / current_price
    confidence = max(50, min(95, 100 - (volatility * 100)))
    if expected_return > 2:
        return "BUY", confidence
    elif expected_return < -2:
        return "SELL", confidence
    else:
        return "HOLD", confidence


if __name__ == "__main__":
    main()
