# 🧠 MarketMind

**Creating Future Trends with AI**

Progressive Web App for stock price prediction using Hybrid Sentiment-Technical Transformer models.

## 🚀 Quick Start

### Run Locally

```bash
# Navigate to app directory
cd app

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

### 📱 Install on iPhone

1. Open the app URL in Safari
2. Tap the Share button (square with arrow)
3. Scroll down and tap "Add to Home Screen"
4. Name it "MarketMind" and tap "Add"
5. The 🧠 MarketMind icon will appear on your home screen
6. Launch from home screen for full-screen experience

### 🤖 Install on Android

1. Open the app URL in Chrome
2. Tap the menu (⋮) in top right
3. Tap "Install app" or "Add to Home Screen"
4. Tap "Install"

## ✨ Features

- 📊 **Real-time Predictions** - Generate predictions for 1-30 days
- 🎯 **Trading Signals** - AI-powered BUY/SELL/HOLD recommendations
- 📈 **Interactive Charts** - Plotly visualizations with zoom and pan
- ⚖️ **Model Comparison** - Compare all 4 model architectures
- 💬 **Sentiment Analysis** - View latest news sentiment scores
- 📉 **Technical Indicators** - RSI, MACD, and more
- 📱 **Mobile Optimized** - Works great on phones and tablets
- 🌐 **Works Offline** - Cache predictions for offline access (coming soon)

## 🎯 Supported Stocks

### US Markets (NASDAQ)
- 🍎 AAPL - Apple Inc.
- 🔍 GOOGL - Alphabet Inc.
- 🚗 TSLA - Tesla Inc.
- 📦 AMZN - Amazon.com Inc.
- 💻 MSFT - Microsoft Corp.

### Indian Markets (NSE)
- 🇮🇳 RELIANCE.NS - Reliance Industries
- 🇮🇳 TCS.NS - Tata Consultancy Services
- 🇮🇳 INFY.NS - Infosys Ltd.

### Sri Lankan Markets (CSE)
- 🇱🇰 CSEALL - CSE All Share Index

## 🤖 AI Models

### 1. Early Fusion Transformer
- Combines sentiment and technical features at input layer
- **Best for**: Short-term predictions
- **Complexity**: Medium

### 2. Late Fusion Transformer
- Processes features separately, combines at decision layer
- **Best for**: Balanced predictions
- **Complexity**: Medium

### 3. Attention Fusion Transformer
- Dynamic weighting of sentiment vs technical signals
- **Best for**: Volatile markets
- **Complexity**: High

### 4. LSTM Baseline
- Traditional recurrent neural network
- **Best for**: Long-term trends
- **Complexity**: Low

## 📊 Performance Metrics

All models display:
- **MAPE** - Mean Absolute Percentage Error (lower is better)
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **Directional Accuracy** - Price direction prediction accuracy

## 🛠️ Technology Stack

- **Backend**: Python 3.9+
- **Web Framework**: Streamlit
- **ML Framework**: PyTorch
- **Charts**: Plotly
- **Models**: Transformer architecture with attention mechanisms

## 📁 Project Structure

```
app/
├── streamlit_app.py         # Main app file
├── utils/
│   ├── data_loader.py       # Load stock data
│   ├── model_loader.py      # Load trained models
│   ├── predictor.py         # Generate predictions
│   └── visualizer.py        # Create charts
└── requirements.txt         # App dependencies
```

## 🌐 Deployment

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app/streamlit_app.py` as main file
5. Deploy!

Your app will be live at `https://your-app.streamlit.app`

### Alternative: Heroku

```bash
# Create Procfile
echo "web: streamlit run app/streamlit_app.py --server.port=\$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 🔒 Security & Privacy

- ✅ No data collection
- ✅ No user tracking
- ✅ Models run locally
- ✅ Open source code

## ⚠️ Disclaimer

**This is a research project for educational purposes.**

- Not financial advice
- Past performance doesn't guarantee future results
- Always do your own research before investing
- Models are for demonstration and learning

## 👨‍🎓 Research

**Author**: Sewmini Kangara  
**Institution**: Coventry University / NIBM  
**Program**: BSc Computing (Hons)  
**Date**: February 2026

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@thesis{kangara2026hybrid,
  title={Hybrid Sentiment-Technical Transformer Models for Stock Price Prediction},
  author={Kangara, Sewmini},
  year={2026},
  school={Coventry University}
}
```

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📧 Contact

GitHub: [@Sewminikangara](https://github.com/Sewminikangara)

---

**🧠 MarketMind** - Creating Future Trends with AI  
Made with ❤️ using Streamlit and PyTorch
