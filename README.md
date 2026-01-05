# Hybrid Sentiment-Technical Transformer for Stock Price Forecasting

**BSc (Hons) Computing - Final Year Project**  
**Coventry University / NIBM Sri Lanka**

## Overview

This project investigates the integration of technical indicators and sentiment analysis using Transformer-based architectures for stock price prediction. The research compares multiple fusion strategies (early, late, and attention-based) across developed and emerging markets.

## Project Structure

```
├── models/              # Model architectures (ARIMA, LSTM, Transformers)
├── scripts/             # Data processing, training, and evaluation
├── data_raw/           # Raw stock price and sentiment data
├── data_processed/     # Processed technical indicators
├── results/            # Training results and metrics
├── graphs/             # Prediction visualizations
├── notebooks/          # Exploratory data analysis
└── dissertation/       # Research documentation and diagrams
```

## Models Implemented

### Baseline Models
- **ARIMA(5,1,0)** - Traditional statistical baseline
- **LSTM** - Deep learning baseline (2 layers, 128 units)

### Transformer Models
- **Technical-only Transformer** - Uses only price-based indicators
- **Early Fusion** - Concatenates technical and sentiment features at input
- **Late Fusion** - Separate encoders, merged at prediction layer
- **Attention Fusion** - Cross-attention between technical and sentiment streams

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.9+
- PyTorch 2.0+
- TensorFlow 2.13+
- pandas, numpy, scikit-learn
- statsmodels (for ARIMA)
- transformers (for FinBERT)

## Usage

### Quick Test (Single Stock)
```bash
python quick_test.py
```
Trains LSTM and Transformer on AAPL with 20 epochs (~5-10 minutes)

### Full Training (All Stocks)
```bash
python train_all_models.py
```
Trains both models on all 8 stocks with 50 epochs (~2-3 hours)

### Data Collection

**Stock Prices:**
```bash
python scripts/collect_stock_data.py
```

**Calculate Technical Indicators:**
```bash
python scripts/calculate_technical_indicators.py
```

**Sentiment Data** (requires API keys):
```bash
python scripts/collect_reddit_data.py
python scripts/collect_news_data.py
```

### Evaluation
```bash
python scripts/evaluate.py
```
Generates comparison plots and performance metrics

## Dataset

### Stocks Analyzed
- **US Market:** AAPL, GOOGL, TSLA, AMZN, MSFT
- **Indian Market:** RELIANCE.NS, TCS.NS, INFY.NS

### Time Period
2021-01-01 to 2024-12-31 (daily data)

### Features
- **Technical Indicators:** RSI, MACD, Bollinger Bands, Moving Averages (20/50/200), Momentum, Volatility
- **Sentiment Features:** FinBERT polarity scores from news and social media (planned)

## Evaluation Metrics

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (%)
- Simulated Trading Profit

## Results

Results are saved in `results/` as CSV files with timestamp.  
Trained models are checkpointed in `models/` as `.pt` files.  
Prediction plots are saved in `graphs/` at 300 DPI.

## Research Objectives

1. Design Transformer-based baseline using technical indicators
2. Extract sentiment features from financial news and social media
3. Integrate technical and sentiment data using hybrid fusion strategies
4. Evaluate performance across multiple metrics and markets
5. Develop practical software tool for individual investors

## Notes

- Data preprocessing uses 70/15/15 train/val/test split
- Sequence length: 60 days
- Optimizer: Adam with learning rate 0.001
- Loss function: MSE
- Early stopping based on validation loss

## Author

K.M.S.S Kangara  
BSc (Hons) Computing  
Coventry University / NIBM  
2024/2025

## License

This project is submitted as part of academic requirements and is not licensed for commercial use.
