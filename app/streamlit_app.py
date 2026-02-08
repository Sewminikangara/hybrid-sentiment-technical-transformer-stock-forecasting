"""
Stock Price Prediction PWA
Hybrid Sentiment-Technical Transformer Models
Progressive Web App for iOS/Android/Desktop

Author: Sewmini Kangara
Date: February 2026
"""

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
    page_title="MarketMind - AI Stock Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Sewminikangara',
        'Report a bug': 'https://github.com/Sewminikangara',
        'About': "# MarketMind\nCreating Future Trends with AI"
    }
)

# PWA Meta tags and custom CSS
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="MarketMind">
<meta name="theme-color" content="#667eea">
<meta name="description" content="MarketMind - AI-powered stock predictions. Creating future trends with hybrid sentiment-technical transformer models.">

<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
            max-width: 100%;
        }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        
        /* Larger touch targets */
        .stButton button {
            height: 3rem;
            font-size: 1rem;
        }
        .stSelectbox select {
            height: 3rem;
            font-size: 1rem;
        }
    }
    
    /* Custom styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .buy-signal { color: #4CAF50; font-weight: bold; }
    .sell-signal { color: #f44336; font-weight: bold; }
    .hold-signal { color: #FFC107; font-weight: bold; }
    
    /* Hide Streamlit branding on mobile */
    @media (max-width: 768px) {
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    }
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
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("MarketMind")
        st.markdown("*Creating Future Trends with AI*")
    with col2:
        st.image("https://img.shields.io/badge/AI-Transformer-blueviolet", use_column_width=True)
    
    # Sidebar - Stock & Model Selection
    with st.sidebar:
        st.header("Settings")
        
        # Stock selector
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
        
        selected_stock = st.selectbox(
            "Select Stock",
            stocks,
            format_func=lambda x: stock_names[x],
            key='stock_selector'
        )
        
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
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analysis", "Compare", "About"])
    
    with tab1:
        show_prediction_tab(selected_stock, selected_model, pred_days)
    
    with tab2:
        show_analysis_tab(selected_stock)
    
    with tab3:
        show_comparison_tab(selected_stock)
    
    with tab4:
        show_about_tab()

def show_prediction_tab(stock, model, days):
    """Main prediction interface"""
    
    st.header(f"{stock} Price Prediction")
    
    try:
        # Load data
        with st.spinner("Loading data..."):
            data_loader = st.session_state.data_loader
            stock_data = data_loader.load_stock_data(stock)
            
            if stock_data is None or len(stock_data) == 0:
                st.error(f"No data available for {stock}")
                return
        
        # Get latest price and change
        latest_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2]
        price_change = latest_price - prev_price
        pct_change = (price_change / prev_price) * 100
        
        # Display current price
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Price",
                f"${latest_price:.2f}",
                f"{pct_change:+.2f}%",
                delta_color="normal"
            )
        
        with col2:
            latest_date = pd.to_datetime(stock_data['Date'].iloc[-1]).strftime('%Y-%m-%d')
            st.metric("Last Update", latest_date)
        
        with col3:
            st.metric("Model", model)
        
        # Generate prediction
        with st.spinner("Generating predictions..."):
            predictor = StockPredictor(stock, model)
            predictions = predictor.predict(days)
            
            if predictions is None:
                st.warning("Model not available. Showing sample prediction.")
                # Generate sample predictions
                predictions = {
                    'dates': [datetime.now() + timedelta(days=i) for i in range(1, days+1)],
                    'prices': [latest_price * (1 + np.random.normal(0, 0.02)) for _ in range(days)],
                    'lower': [latest_price * (1 - 0.05) for _ in range(days)],
                    'upper': [latest_price * (1 + 0.05) for _ in range(days)]
                }
        
        # Prediction chart
        st.subheader("Price Forecast")
        visualizer = ChartVisualizer()
        fig = visualizer.create_prediction_chart(stock_data, predictions, stock)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading signal
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
        
        with col1:
            st.metric("Predicted Price", f"${pred_price:.2f}")
        
        with col2:
            st.metric("Price Range", f"${predictions['lower'][-1]:.2f} - ${predictions['upper'][-1]:.2f}")
        
        with col3:
            volatility = np.std([p for p in predictions['prices']])
            st.metric("Volatility", f"${volatility:.2f}")
        
        with col4:
            trend = "Bullish" if pred_price > latest_price else "Bearish"
            st.metric("Trend", trend)
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        st.info("This is a demo. Full prediction requires trained models.")

def show_analysis_tab(stock):
    """Technical and sentiment analysis"""
    
    st.header("Technical Analysis")
    
    try:
        data_loader = st.session_state.data_loader
        stock_data = data_loader.load_stock_data(stock)
        
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
                sent_label = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
                st.metric("Sentiment Score", f"{sentiment:.3f}", sent_label)
            
            # Sentiment breakdown
            if 'sentiment_positive' in stock_data.columns:
                pos = stock_data['sentiment_positive'].iloc[-1] * 100
                neg = stock_data['sentiment_negative'].iloc[-1] * 100
                neu = stock_data['sentiment_neutral'].iloc[-1] * 100
                
                st.write(f"**Breakdown:**")
                st.write(f"• Positive: {pos:.1f}%")
                st.write(f"• Negative: {neg:.1f}%")
                st.write(f"• Neutral: {neu:.1f}%")
            
            # Sentiment trend analysis
            if 'sentiment_ma7' in stock_data.columns and 'sentiment_ma3' in stock_data.columns:
                ma7 = stock_data['sentiment_ma7'].iloc[-1]
                ma3 = stock_data['sentiment_ma3'].iloc[-1]
                prev_ma7 = stock_data['sentiment_ma7'].iloc[-2]
                
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

def show_comparison_tab(stock):
    """Compare all models"""
    
    st.header("⚖️ Model Comparison")
    
    try:
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
        st.subheader("📊 Performance Metrics")
        
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
        st.success(f"🏆 Best Model: {best_model['Model']} (MAPE: {best_model['MAPE']:.2f}%)")
        
    except Exception as e:
        st.error(f"Error loading comparison: {str(e)}")

def show_about_tab():
    """About the research"""
    
    st.header("ℹ️ About This Research")
    
    st.markdown("""
    ### 🎓 Hybrid Sentiment-Technical Transformer Models
    
    This application demonstrates a novel approach to stock price prediction using:
    
    #### 🔬 Technology Stack
    - **Deep Learning**: Transformer architecture with attention mechanisms
    - **Hybrid Features**: Technical indicators + Sentiment analysis
    - **Real-time Data**: News sentiment integrated with price movements
    
    #### 📊 Models Available
    
    1. **Early Fusion Transformer**
       - Combines sentiment and technical features at input layer
       - Best for: Short-term predictions
    
    2. **Late Fusion Transformer**
       - Processes features separately, combines at decision layer
       - Best for: Balanced predictions
    
    3. **Attention Fusion Transformer**
       - Dynamic weighting of sentiment vs technical signals
       - Best for: Volatile markets
    
    4. **LSTM Baseline**
       - Traditional recurrent network for comparison
       - Best for: Long-term trends
    
    #### 📈 Performance
    - Trained on 9 stocks (US, India, Sri Lanka markets)
    - Tested on 377 days of real sentiment data
    - Average directional accuracy: 54-56%
    - Statistical significance proven via t-tests and ANOVA
    
    #### 🚀 Features
    - ✅ Real-time predictions
    - ✅ Multiple model comparison
    - ✅ Technical + Sentiment analysis
    - ✅ Trading signals
    - ✅ Mobile-optimized (PWA)
    
    #### 👨‍🎓 Research by
    **Sewmini Kangara**  
    BSc Computing (Hons)  
    Coventry University / NIBM  
    February 2026
    
    #### 📚 Repository
    [GitHub](https://github.com/Sewminikangara/hybrid-sentiment-technical-transformer-stock-forecasting)
    
    ---
    
    **⚠️ Disclaimer**: This is a research project. Not financial advice.  
    Always do your own research before making investment decisions.
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
