"""
Diagnostic tool to investigate model prediction patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.transformer_early_fusion import EarlyFusionPredictor
from models.transformer_late_fusion import LateFusionPredictor
from models.transformer_attention_fusion import AttentionFusionPredictor

def diagnose_model(stock, model_type):
    """Check what a model is actually predicting"""
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {stock} - {model_type}")
    print('='*80)
    
    # Load data
    hybrid_file = 'data_processed/hybrid/hybrid_data_all_stocks_20260207_093427.csv'
    df = pd.read_csv(hybrid_file)
    stock_data = df[df['Stock'] == stock].copy()
    stock_data = stock_data.sort_values('Date')
    
    # Features
    technical_cols = [c for c in stock_data.columns if c not in 
                     ['Date', 'Stock', 'stock', 'date', 'source', 'Close',
                      'sentiment_score', 'sentiment_label', 'confidence',
                      'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
                      'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']]
    
    sentiment_cols = ['sentiment_score', 'sentiment_positive', 'sentiment_negative', 
                     'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']
    
    # Prepare sequences
    SEQUENCE_LENGTH = 60
    technical_data = stock_data[technical_cols].values
    sentiment_data = stock_data[sentiment_cols].values
    prices = stock_data['Close'].values
    dates = pd.to_datetime(stock_data['Date']).values
    
    X_tech_seq = []
    X_sent_seq = []
    y = []
    test_dates = []
    
    for i in range(SEQUENCE_LENGTH, len(stock_data)):
        X_tech_seq.append(technical_data[i-SEQUENCE_LENGTH:i])
        X_sent_seq.append(sentiment_data[i-SEQUENCE_LENGTH:i])
        y.append(prices[i])
        test_dates.append(dates[i])
    
    X_tech = np.array(X_tech_seq)
    X_sent = np.array(X_sent_seq)
    y_actual = np.array(y)
    
    # Test set
    test_size = int(len(X_tech) * 0.15)
    X_tech_test = X_tech[-test_size:]
    X_sent_test = X_sent[-test_size:]
    y_test = y_actual[-test_size:]
    test_dates = test_dates[-test_size:]
    
    # Load model
    model_path = f"results/{stock}_{model_type}.pt"
    
    if model_type == 'early_fusion':
        predictor = EarlyFusionPredictor(len(technical_cols), len(sentiment_cols))
    elif model_type == 'late_fusion':
        predictor = LateFusionPredictor(len(technical_cols), len(sentiment_cols))
    else:
        predictor = AttentionFusionPredictor(len(technical_cols), len(sentiment_cols))
    
    predictor.load_model(model_path)
    y_pred = predictor.predict(X_tech_test, X_sent_test).flatten()
    
    # Analysis
    print(f"\nActual Prices:")
    print(f"  Min: {y_test.min():.2f}, Max: {y_test.max():.2f}, Mean: {y_test.mean():.2f}")
    print(f"\nPredicted Prices:")
    print(f"  Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")
    
    # Direction accuracy
    actual_direction = np.diff(y_test) > 0
    pred_direction = np.diff(y_pred) > 0
    dir_acc = np.mean(actual_direction == pred_direction) * 100
    
    print(f"\nDirection Accuracy: {dir_acc:.2f}%")
    
    # Price differences
    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        predicted = y_pred[i]
        diff_pct = ((predicted - actual) / actual) * 100
        print(f"  Day {i+1}: Actual=${actual:.2f}, Pred=${predicted:.2f}, Diff={diff_pct:+.2f}%")
    
    # Expected returns
    expected_returns = []
    for i in range(1, len(y_pred)):
        current = y_test[i-1]
        predicted_next = y_pred[i]
        exp_return = (predicted_next - current) / current
        expected_returns.append(exp_return)
    
    expected_returns = np.array(expected_returns)
    
    print(f"\nExpected Returns Distribution:")
    print(f"  Positive expectations: {np.sum(expected_returns > 0)} / {len(expected_returns)}")
    print(f"  Negative expectations: {np.sum(expected_returns < 0)} / {len(expected_returns)}")
    print(f"  Mean expected return: {np.mean(expected_returns)*100:.4f}%")
    print(f"  Median expected return: {np.median(expected_returns)*100:.4f}%")
    
    # Signals at different thresholds
    for threshold in [0.001, 0.002, 0.005, 0.01]:
        buy_signals = np.sum(expected_returns > threshold)
        sell_signals = np.sum(expected_returns < -threshold)
        hold_signals = len(expected_returns) - buy_signals - sell_signals
        print(f"\n  Threshold {threshold*100:.1f}%:")
        print(f"    BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")

# Test a few models
diagnose_model('AAPL', 'early_fusion')
diagnose_model('GOOGL', 'attention_fusion')
diagnose_model('TSLA', 'early_fusion')
diagnose_model('MSFT', 'late_fusion')
