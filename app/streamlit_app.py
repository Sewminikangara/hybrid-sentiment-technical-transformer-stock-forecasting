

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader
from utils.predictor import StockPredictor
from utils.visualizer import ChartVisualizer

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="TradeXy - AI Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Sewminikangara',
        'Report a bug': 'https://github.com/Sewminikangara',
        'About': "# TradeXy\nProfessional AI Trading Intelligence Platform"
    }
)

# PWA Meta tags and PROFESSIONAL custom CSS with animations
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="TradeXy">
<meta name="theme-color" content="#0F172A">
<meta name="description" content="TradeXy - Professional AI Trading Intelligence. Real-time predictions powered by deep learning transformers.">

<link rel="icon" type="image/png" href="⚡">
<link rel="apple-touch-icon" sizes="180x180" href="⚡">

<style>
    /* TRADEXY PROFESSIONAL THEME */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Dark Professional Theme */
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Animated Header */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    h1, h2, h3 {
        animation: fadeInDown 0.6s ease-out;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Professional Metric Cards */
    .stMetric {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInRight 0.5s ease-out;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(96, 165, 250, 0.3);
        border-color: rgba(96, 165, 250, 0.5);
    }
    
    /* Animated Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        animation: pulse 2s infinite;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5);
        animation: none;
    }
    
    /* Professional Select Boxes */
    .stSelectbox, .stRadio {
        animation: fadeInDown 0.4s ease-out;
    }
    
    /* Trading Signals with Glow */
    .buy-signal {
        color: #10B981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        font-weight: 700;
        animation: pulse 1.5s infinite;
    }
    
    .sell-signal {
        color: #EF4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        font-weight: 700;
        animation: pulse 1.5s infinite;
    }
    
    .hold-signal {
        color: #F59E0B;
        text-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
        font-weight: 700;
    }
    
    /* Chart Container */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        animation: fadeInDown 0.7s ease-out;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.35rem !important; }
        .stMetric {
            padding: 1rem;
        }
    }
    
    /* Loading Animations */
    .stSpinner > div {
        border-color: #3B82F6 transparent transparent transparent !important;
    }
    
    /* Data Tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data_loader = DataLoader()
    st.session_state.model_loader = ModelLoader()
    st.session_state.predictor = None
    st.session_state.selected_stock = 'AAPL'
    st.session_state.selected_model = 'Early Fusion'

def main():
    """Main application"""
    
    # Professional Header with Branding
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0; animation: fadeInDown 0.8s ease-out;'>
        <h1 style='font-size: 3.5rem; margin: 0; letter-spacing: -0.02em;'>
            ⚡ TradeXy
        </h1>
        <p style='color: #94A3B8; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 500;'>
            Professional AI Trading Intelligence Platform
        </p>
        <p style='color: #64748B; font-size: 0.9rem;'>
            Powered by Deep Learning Transformers | Real-time Market Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Professional Settings Panel
    with st.sidebar:
        st.markdown("### ⚙️ Trading Settings")
        
        # Market Type selector with modern style
        market_type = st.radio(
            "📊 Select Market",
            ["Stocks", "Forex"],
            index=0,
            help="Choose between stock markets or forex currency pairs"
        )
        
        # Stock selector
        if market_type == "Stocks":
            stocks = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 
                      'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'CSEALL']
            stock_names = {
                'AAPL': 'Apple',
                'GOOGL': 'Google',
                'TSLA': 'Tesla',
                'AMZN': 'Amazon',
                'MSFT': 'Microsoft',
                'RELIANCE.NS': 'Reliance',
                'TCS.NS': 'TCS',
                'INFY.NS': 'Infosys',
                'CSEALL': 'CSE All Share'
            }
        else:  # Forex
            stocks = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
            stock_names = {
                'EURUSD': 'EUR/USD - Euro vs US Dollar',
                'GBPUSD': 'GBP/USD - British Pound vs US Dollar',
                'USDJPY': 'USD/JPY - US Dollar vs Japanese Yen',
                'AUDUSD': 'AUD/USD - Australian Dollar vs US Dollar',
                'USDCAD': 'USD/CAD - US Dollar vs Canadian Dollar',
                'USDCHF': 'USD/CHF - US Dollar vs Swiss Franc'
            }
        
        label = "Select Currency Pair" if market_type == "Forex" else "Select Stock"
        selected_stock = st.selectbox(
            label,
            stocks,
            format_func=lambda x: stock_names[x],
            key='stock_selector'
        )
        
        # Show forex disclaimer
        if market_type == "Forex":
            st.info("Forex data available - AI predictions coming soon! Currently showing real exchange rates from European Central Bank.")
        
        # Model selector
        models = ['Early Fusion', 'Late Fusion', 'Attention Fusion', 'LSTM Baseline']
        selected_model = st.selectbox(
            "Select Model",
            models,
            key='model_selector'
        )
        
        # Prediction horizon
        pred_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to predict"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("Quick Stats")
        try:
            results_file = list(Path('../results').glob('hybrid_training_results_*.csv'))
            if results_file:
                results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
                model_map = {
                    'Early Fusion': 'Early_Fusion',
                    'Late Fusion': 'Late_Fusion',
                    'Attention Fusion': 'Attention_Fusion',
                    'LSTM Baseline': 'LSTM'
                }
                model_name = model_map.get(selected_model, 'Early_Fusion')
                stock_result = results[(results['Stock'] == selected_stock) & 
                                      (results['Model'] == model_name)]
                
                if len(stock_result) > 0:
                    st.metric("MAPE", f"{stock_result['MAPE'].values[0]:.2f}%")
                    st.metric("Accuracy", f"{stock_result['Directional_Accuracy'].values[0]:.1f}%")
        except:
            pass
        
        st.markdown("---")
        st.markdown("### Install App")
        st.markdown("""
        **On iPhone:**
        1. Tap Share button
        2. Scroll down
        3. Tap 'Add to Home Screen'
        
        **On Android:**
        1. Tap menu (⋮)
        2. Tap 'Install app'
        """)
    
    # Main content - Use selectbox instead of tabs for AI Assistant compatibility
    view_mode = st.selectbox(
        "Select View:",
        ["Prediction", "Analysis", "Compare Models", "AI Assistant", "About"],
        index=0
    )
    
    is_forex = (market_type == "Forex")
    
    if view_mode == "Prediction":
        show_prediction_tab(selected_stock, selected_model, pred_days, is_forex)
    elif view_mode == "Analysis":
        show_analysis_tab(selected_stock, is_forex)
    elif view_mode == "Compare Models":
        show_comparison_tab(selected_stock, is_forex)
    elif view_mode == "AI Assistant":
        show_ai_assistant_tab(selected_stock, selected_model)
    elif view_mode == "About":
        show_about_tab()

def show_prediction_tab(stock, model, days, is_forex=False):
    """Main prediction interface"""
    
    label = "Exchange Rate Prediction" if is_forex else "Price Prediction"
    st.header(f"{stock} {label}")
    
    if is_forex:
        st.info("💱 Forex predictions use the same AI transformer models trained on currency pairs from European Central Bank data.")
    
    try:
        with st.spinner("Loading data..."):
            data_loader = st.session_state.data_loader
            stock_data = data_loader.load_stock_data(stock, is_forex=is_forex)
            
            if stock_data is None or len(stock_data) == 0:
                st.error(f"No data available for {stock}")
                return
        
        latest_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100
        
        col1, col2, col3 = st.columns(3)
        price_label = "Current Rate" if is_forex else "Current Price"
        price_format = f"{latest_price:.4f}" if is_forex else f"${latest_price:.2f}"
        
        with col1:
            st.metric(
                price_label,
                price_format,
                f"{pct_change:+.2f}%",
                delta_color="normal"
            )
        
        with col2:
            latest_date = pd.to_datetime(stock_data['Date'].iloc[-1]).strftime('%Y-%m-%d')
            st.metric("Last Update", latest_date)
        
        with col3:
            st.metric("Model", model)
        
        with st.spinner("Generating predictions..."):
            predictor = StockPredictor(stock, model, is_forex=is_forex)
            predictions = predictor.predict(days)
            
            if predictions is None:
                st.error(f"Unable to generate predictions for {stock}. Please try a different model or stock.")
                return
        
        st.subheader("Price Forecast")
        visualizer = ChartVisualizer()
        fig = visualizer.create_prediction_chart(stock_data, predictions, stock)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Trading Signal")
        pred_price = predictions['prices'][-1]
        signal, confidence = generate_signal(latest_price, pred_price, predictions)
        
        signal_colors = {'BUY': '#4CAF50', 'SELL': '#f44336', 'HOLD': '#FFC107'}
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color: {signal_colors[signal]}; margin: 0;">
                    {signal}
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{confidence:.0f}%")
        
        with col3:
            expected_return = ((pred_price - latest_price) / latest_price) * 100
            st.metric("Expected Return", f"{expected_return:+.2f}%")
        
        # Key metrics
        st.subheader("Prediction Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        volatility = np.std([p for p in predictions['prices']])
        price_format_pred = f"{pred_price:.4f}" if is_forex else f"${pred_price:.2f}"
        price_format_range = f"{predictions['lower'][-1]:.4f} - {predictions['upper'][-1]:.4f}" if is_forex else f"${predictions['lower'][-1]:.2f} - ${predictions['upper'][-1]:.2f}"
        volatility_format = f"{volatility:.4f}" if is_forex else f"${volatility:.2f}"
        
        with col1:
            st.metric("Predicted Price", price_format_pred)
        
        with col2:
            st.metric("Price Range", price_format_range)
        
        with col3:
            st.metric("Volatility", volatility_format)
        
        with col4:
            trend = "Bullish" if pred_price > latest_price else "Bearish"
            st.metric("Trend", trend)
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        st.info("This is a demo. Full prediction requires trained models.")

def show_analysis_tab(stock, is_forex=False):
    """Technical and sentiment analysis"""
    
    st.header(f"{stock} {'Exchange Rate' if is_forex else 'Technical'} Analysis")
    
    try:
        data_loader = st.session_state.data_loader
        stock_data = data_loader.load_stock_data(stock, is_forex=is_forex)
        
        if stock_data is None:
            st.warning("No data available")
            return
        
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Technical Indicators")
            
            # RSI
            if 'RSI' in stock_data.columns:
                rsi = stock_data['RSI'].iloc[-1]
                rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_signal)
            
            # MACD
            if 'MACD' in stock_data.columns and 'MACD_Signal' in stock_data.columns:
                macd = stock_data['MACD'].iloc[-1]
                signal = stock_data['MACD_Signal'].iloc[-1]
                macd_signal = "Bullish" if macd > signal else "Bearish"
                st.metric("MACD", f"{macd:.2f}", macd_signal)
        
        with col2:
            st.subheader("Sentiment Analysis")
            
            # Current sentiment score
            if 'sentiment_score' in stock_data.columns:
                sentiment = stock_data['sentiment_score'].iloc[-1]
                sent_label = "Positive" if sentiment > 0.55 else "Negative" if sentiment < 0.45 else "Neutral"
                st.metric("Sentiment Score", f"{sentiment:.3f}", sent_label)
            
            # Sentiment breakdown - handle both column naming conventions
            pos_col = 'sentiment_positive' if 'sentiment_positive' in stock_data.columns else 'positive'
            neg_col = 'sentiment_negative' if 'sentiment_negative' in stock_data.columns else 'negative'
            neu_col = 'sentiment_neutral' if 'sentiment_neutral' in stock_data.columns else 'neutral'
            
            if pos_col in stock_data.columns:
                pos = stock_data[pos_col].iloc[-1] * 100
                neg = stock_data[neg_col].iloc[-1] * 100
                neu = stock_data[neu_col].iloc[-1] * 100
                
                st.write(f"**Breakdown:**")
                st.write(f"• Positive: {pos:.1f}%")
                st.write(f"• Negative: {neg:.1f}%")
                st.write(f"• Neutral: {neu:.1f}%")
            
            # Sentiment trend analysis
            ma7_col = 'sentiment_ma7' if 'sentiment_ma7' in stock_data.columns else 'sentiment_MA7'
            ma3_col = 'sentiment_ma3' if 'sentiment_ma3' in stock_data.columns else 'sentiment_MA3'
            
            if ma7_col in stock_data.columns and ma3_col in stock_data.columns:
                ma7 = stock_data[ma7_col].iloc[-1]
                ma3 = stock_data[ma3_col].iloc[-1]
                prev_ma7 = stock_data[ma7_col].iloc[-2]
                
                trend = "Improving" if ma7 > prev_ma7 else "Declining"
                momentum = "Accelerating" if ma3 > ma7 else "Decelerating"
                
                st.write(f"**Trend:** {trend}")
                st.write(f"**Momentum:** {momentum}")
            
            # Sentiment volatility
            if 'sentiment_volatility' in stock_data.columns:
                volatility = stock_data['sentiment_volatility'].iloc[-1]
                vol_label = "Stable" if volatility < 0.1 else "Volatile" if volatility < 0.2 else "Highly Volatile"
                st.metric("Sentiment Volatility", f"{volatility:.3f}", vol_label)
        
        # Charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Price chart
            st.subheader("Price History (60 days)")
            visualizer = ChartVisualizer()
            fig = visualizer.create_historical_chart(stock_data.tail(60), stock)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment trend chart
            st.subheader("Sentiment Trend (60 days)")
            if 'sentiment_score' in stock_data.columns:
                fig = visualizer.create_sentiment_chart(stock_data.tail(60), stock)
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analysis: {str(e)}")

def show_comparison_tab(stock, is_forex=False):
    """Compare all models"""
    
    st.header("⚖️ Model Comparison")
    
    try:
        # Load results based on market type
        if is_forex:
            results_file = list(Path('../results').glob('forex_training_results_*.csv'))
            if not results_file:
                st.warning("No forex training results available")
                return
            
            results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
            stock_results = results[results['Pair'] == stock]
            
            if len(stock_results) == 0:
                st.warning(f"No results for {stock}")
                return
            
            # Display forex results
            st.subheader("Forex Model Performance")
            
            display_df = stock_results[['Model', 'Status', 'Data_Points']].copy()
            display_df.columns = ['Model', 'Training Status', 'Data Points']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Show model info
            st.subheader("Model Descriptions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Early Fusion Transformer**
                - Combines technical & sentiment features at input
                - Best for: Short-term predictions
                - Complexity: Medium
                
                **Late Fusion Transformer**
                - Processes features separately
                - Best for: Balanced predictions
                - Complexity: Medium
                """)
            
            with col2:
                st.markdown("""
                **Attention Fusion Transformer**
                - Dynamic weighting of signals
                - Best for: Volatile markets
                - Complexity: High
                
                **LSTM Baseline**
                - Traditional recurrent network
                - Best for: Long-term trends
                - Complexity: Low
                """)
            
            return
        
        # Stock results below
        # Load results
        results_file = list(Path('../results').glob('hybrid_training_results_*.csv'))
        if not results_file:
            st.warning("No training results available")
            return
        
        results = pd.read_csv(max(results_file, key=lambda p: p.stat().st_mtime))
        stock_results = results[results['Stock'] == stock]
        
        if len(stock_results) == 0:
            st.warning(f"No results for {stock}")
            return
        
        # Comparison table
        st.subheader("Performance Metrics")
        
        display_df = stock_results[['Model', 'MAPE', 'RMSE', 'MAE', 'Directional_Accuracy']].copy()
        display_df.columns = ['Model', 'MAPE (%)', 'RMSE', 'MAE', 'Accuracy (%)']
        display_df = display_df.sort_values('MAPE (%)')
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("MAPE Comparison")
            fig = px.bar(stock_results, x='Model', y='MAPE', 
                        title='Lower is Better',
                        color='Model')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Directional Accuracy")
            fig = px.bar(stock_results, x='Model', y='Directional_Accuracy',
                        title='Higher is Better',
                        color='Model')
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = stock_results.loc[stock_results['MAPE'].idxmin()]
        st.success(f"Best Model: {best_model['Model']} (MAPE: {best_model['MAPE']:.2f}%)")
        
    except Exception as e:
        st.error(f"Error loading comparison: {str(e)}")

def show_ai_assistant_tab(stock, model):
    """AI chatbot to answer user questions"""
    
    st.header("🤖 TradeXy AI Assistant")
    st.markdown("*Ask me anything about predictions, models, or trading strategies!*")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hi! I'm TradeXy AI, your professional trading assistant. I can help you understand predictions for {stock} using the {model} model. What would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about predictions, models, or trading..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_ai_response(prompt, stock, model)
                st.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Explain this prediction"):
            question = f"Can you explain the current prediction for {stock}?"
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_ai_response(question, stock, model)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()
    
    with col2:
        if st.button("Should I buy now?"):
            question = f"Based on the {model} model, should I buy {stock} now?"
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_ai_response(question, stock, model)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()
    
    with col3:
        if st.button("What's the risk?"):
            question = f"What are the risks of trading {stock} right now?"
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_ai_response(question, stock, model)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

def generate_ai_response(question, stock, model):
    """Generate intelligent responses based on stock data and predictions"""
    
    try:
        # Load data
        data_loader = DataLoader()
        stock_data = data_loader.load_stock_data(stock)
        
        # Get current price and prediction
        if stock_data is not None and len(stock_data) > 0:
            current_price = stock_data['Close'].iloc[-1]
            
            # Generate prediction
            predictor = StockPredictor(stock, model)
            predictions = predictor.predict(7)
            
            if predictions:
                pred_price = predictions['prices'][-1]
                price_change = ((pred_price - current_price) / current_price) * 100
                
                # Determine sentiment
                if 'sentiment_score' in stock_data.columns:
                    sentiment = stock_data['sentiment_score'].iloc[-1]
                    sent_label = "positive" if sentiment > 0.05 else "negative" if sentiment < -0.05 else "neutral"
                else:
                    sent_label = "neutral"
        else:
            current_price = 0
            pred_price = 0
            price_change = 0
            sent_label = "neutral"
        
        # Answer common questions
        question_lower = question.lower()
        
        if "explain" in question_lower or "what" in question_lower and "prediction" in question_lower:
            return f"""Based on the **{model}** model, here's the analysis for **{stock}**:

**Current Price:** ${current_price:.2f}
**7-Day Prediction:** ${pred_price:.2f}
**Expected Change:** {price_change:+.2f}%

The model analyzes 35 technical indicators (RSI, MACD, moving averages, etc.) combined with 7 sentiment features from news analysis. 

The prediction suggests the stock is likely to {'rise' if price_change > 0 else 'fall' if price_change < 0 else 'remain stable'} over the next week. Market sentiment is currently {sent_label}.

**How it works:** The hybrid transformer model uses attention mechanisms to identify patterns in historical prices and news sentiment, similar to how ChatGPT processes text."""
        
        elif "buy" in question_lower or "sell" in question_lower or "trade" in question_lower:
            signal = "BUY" if price_change > 2 else "SELL" if price_change < -2 else "HOLD"
            confidence = min(abs(price_change) * 10, 85)
            
            return f"""**Trading Signal for {stock}:** {signal}

**Analysis:**
- Current Price: ${current_price:.2f}
- Predicted Price (7 days): ${pred_price:.2f}
- Expected Return: {price_change:+.2f}%
- Confidence: {confidence:.0f}%
- Model: {model}

**Recommendation:**
{'[BUY] Consider buying - The model predicts upward movement.' if signal == 'BUY' else '[SELL] Consider selling or avoid buying - The model predicts downward movement.' if signal == 'SELL' else '[HOLD] Hold your position - No clear direction predicted.'}

**Risk Warning:** This is AI-generated analysis based on historical data. Always do your own research and never invest more than you can afford to lose. Past performance does not guarantee future results."""
        
        elif "risk" in question_lower:
            volatility = abs(price_change) / 7
            risk_level = "HIGH" if volatility > 1.5 else "MODERATE" if volatility > 0.5 else "LOW"
            
            return f"""**Risk Analysis for {stock}:**

**Risk Level:** {risk_level}

**Key Risks:**
1. **Price Volatility:** Expected daily change of ±{volatility:.2f}%
2. **Model Uncertainty:** AI predictions have ~65% directional accuracy
3. **Market Sentiment:** Currently {sent_label} - sentiment can shift quickly
4. **Data Limitations:** Based on historical patterns up to Dec 2024

**Risk Management Tips:**
- Set stop-loss at {current_price * 0.95:.2f} (-5%)
- Take profit target at {current_price * 1.05:.2f} (+5%)
- Only invest 2-5% of your portfolio in a single stock
- Monitor news and earnings reports

**Remember:** Higher potential returns come with higher risks!"""
        
        elif "model" in question_lower or "how" in question_lower:
            return f"""**About the {model} Model:**

**{model}** is one of four AI models available:

1. **Early Fusion Transformer**: Combines technical + sentiment data at the input layer
2. **Late Fusion Transformer**: Processes technical and sentiment separately, then combines
3. **Attention Fusion Transformer**: Uses dynamic weighting - decides which signals to trust more
4. **LSTM Baseline**: Traditional recurrent neural network for comparison

**Your Current Model ({model}):**
- Trained on 377 days of real stock data
- Uses 35 technical indicators + 7 sentiment features
- Achieved ~65% directional accuracy in backtesting
- Based on transformer architecture (same as ChatGPT!)

**How It Works:**
1. Analyzes last 60 days of price + sentiment data
2. Identifies patterns using attention mechanisms
3. Predicts next 7 days of price movements
4. Provides confidence intervals (±5%)

Want to try a different model? Use the Compare tab to see all 4 models side-by-side!"""
        
        elif "sentiment" in question_lower:
            if stock_data is not None and 'sentiment_score' in stock_data.columns:
                sentiment = stock_data['sentiment_score'].iloc[-1]
                trend = "improving" if stock_data['sentiment_ma7'].iloc[-1] > stock_data['sentiment_ma7'].iloc[-2] else "declining"
                
                return f"""**Sentiment Analysis for {stock}:**

**Current Sentiment:** {sentiment:.3f} ({sent_label.upper()})
**Trend:** {trend.capitalize()}
**Source:** News articles and financial reports

**What This Means:**
{'Positive news sentiment suggests investor confidence is rising. This often precedes price increases.' if sentiment > 0.05 else 'Negative news sentiment suggests concerns about the stock. Be cautious.' if sentiment < -0.05 else 'Neutral sentiment indicates no strong market opinion. Price may be driven by technical factors.'}

**Sentiment Features Used:**
- Daily sentiment score (positive/negative/neutral)
- 3-day and 7-day moving averages (trends)
- Sentiment volatility (stability)

The AI model combines this sentiment data with technical indicators to make more informed predictions than price-only models."""
            else:
                return "Sentiment data is not available for this stock."
        
        else:
            return f"""I can help you with:

1. **Predictions** - "Explain the prediction for {stock}"
2. **Trading Advice** - "Should I buy {stock}?"
3. **Risk Assessment** - "What are the risks?"
4. **Model Information** - "How does {model} work?"
5. **Sentiment Analysis** - "What's the sentiment for {stock}?"

Or ask me anything else about stock trading and AI predictions!

**Current Status:**
- Stock: {stock}
- Model: {model}
- Current Price: ${current_price:.2f}
- Predicted Change: {price_change:+.2f}%"""
    
    except Exception as e:
        return f"I encountered an error: {str(e)}. Please try asking your question differently or check if the stock data is available."

def show_about_tab():
    """About the platform"""
    
    st.header("⚡ About TradeXy")
    
    st.markdown("""
    ### Professional AI Trading Intelligence Platform
    
    **TradeXy** is a cutting-edge trading platform powered by deep learning transformers that combines sentiment analysis with technical indicators to predict stock and forex movements with research-grade accuracy.
    
    ---
    
    #### 🎯 Multi-Market Support
    - **Stocks**: 9 major stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, RELIANCE.NS, TCS.NS, INFY.NS, CSEALL)
    - **Forex**: 6 major currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF)
    - **Real-time Analysis**: Live sentiment + technical indicator processing
    
    #### 🤖 AI Model Architecture
    
    1. **Early Fusion Transformer**
       - Combines sentiment + technical at input layer
       - Best for: Short-term predictions with high correlation
    
    2. **Late Fusion Transformer**
       - Processes features separately, combines at decision layer
       - Best for: Balanced predictions in mixed signals
    
    3. **Attention Fusion Transformer**
       - Dynamic weighting of sentiment vs technical signals
       - Best for: Volatile markets with rapid changes
    
    4. **Technical Transformer**
       - Pure technical analysis without sentiment
       - Best for: Traditional technical trading baseline
    
    #### 📊 Research-Grade Performance
    - **Stock Models**: 377 days training data across 9 stocks
    - **Forex Models**: 267 days training data across 6 pairs
    - **Features**: 35-42 technical indicators + 7 sentiment metrics
    - **Accuracy**: ~65% directional accuracy (statistically validated)
    - **Validation**: T-tests, ANOVA, backtesting analysis
    
    #### ⚡ Professional Features
    - Real-time predictions with confidence intervals
    - Multi-model comparison dashboard
    - Advanced technical analysis (RSI, MACD, Bollinger, Stochastic)
    - Sentiment integration from financial news
    - AI Trading Assistant for insights
    - Progressive Web App (PWA) - installable on any device
    - Professional dark theme with animations
    
    ---
    
    #### 🔬 Research Foundation
    **Project**: Hybrid Sentiment-Technical Transformer Forecasting  
    **Researcher**: Sewmini Kangara  
    **Institution**: BSc Computing (Honours) - Coventry University / NIBM  
    **Completion**: February 2026  
    
    **Data Sources**:
    - Stock prices: Yahoo Finance API
    - Forex data: European Central Bank (ECB) - Frankfurter API
    - Sentiment: Financial news NLP analysis
    
    **Repository**: [GitHub](https://github.com/Sewminikangara/hybrid-sentiment-technical-transformer-stock-forecasting)
    
    ---
    
    #### ⚠️ Professional Disclaimer
    
    **TradeXy** is a research platform demonstrating advanced deep learning for financial forecasting. **For educational and research purposes only**.
    
    - ❌ **NOT financial advice** - AI-generated estimates only
    - ⚠️ **Past performance ≠ Future results** - Markets are unpredictable
    - 💼 **Consult professionals** - Seek qualified financial advisors
    - 📊 **Do your research** - Use multiple sources
    - ⚡ **Trade responsibly** - Only invest what you can afford to lose
    
    **Trading involves substantial risk. This showcases AI research capabilities.**
    
    ---
    
    *Built with 🧠 Deep Learning | Powered by ⚡ Transformers | Made with 💜 for Research*
    """)

def generate_signal(current_price, predicted_price, predictions):
    """Generate trading signal"""
    
    expected_return = ((predicted_price - current_price) / current_price) * 100
    
    # Calculate confidence based on prediction range
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
