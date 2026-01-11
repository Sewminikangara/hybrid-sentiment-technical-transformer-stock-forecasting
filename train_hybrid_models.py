"""
Train All Hybrid Models
Early Fusion + Late Fusion + Attention Fusion
"""

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.data_preprocessing import StockDataPreprocessor
from models.transformer_early_fusion import EarlyFusionPredictor
from models.transformer_late_fusion import LateFusionPredictor
from models.transformer_attention_fusion import AttentionFusionPredictor

# Configuration
HYBRID_DATA_FILE = 'data_processed/hybrid/hybrid_data_all_stocks_20260107_144619.csv'
SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32

# Stock symbols
STOCKS = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'CSEALL']

# Feature counts
TECHNICAL_FEATURES = 35
SENTIMENT_FEATURES = 7

def prepare_hybrid_data(stock_symbol):
    """Prepare hybrid data for training"""
    print(f"\n  Loading hybrid data for {stock_symbol}...")
    
    # Load hybrid dataset
    df = pd.read_csv(HYBRID_DATA_FILE)
    stock_data = df[df['Stock'] == stock_symbol].copy()
    stock_data = stock_data.sort_values('Date')
    
    print(f"    Records: {len(stock_data)}")
    print(f"    Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
    
    # Separate technical and sentiment features
    technical_cols = [c for c in stock_data.columns if c not in 
                     ['Date', 'Stock', 'stock', 'date', 'source', 'Close',
                      'sentiment_score', 'sentiment_label', 'confidence',
                      'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
                      'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']]
    
    sentiment_cols = ['sentiment_score', 'sentiment_positive', 'sentiment_negative', 
                     'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']
    
    print(f"    Technical features: {len(technical_cols)}")
    print(f"    Sentiment features: {len(sentiment_cols)}")
    
    # Prepare features
    technical_data = stock_data[technical_cols].values
    sentiment_data = stock_data[sentiment_cols].values
    
    # Create sequences
    X_tech_seq = []
    X_sent_seq = []
    y = []
    
    for i in range(SEQUENCE_LENGTH, len(stock_data)):
        X_tech_seq.append(technical_data[i-SEQUENCE_LENGTH:i])
        X_sent_seq.append(sentiment_data[i-SEQUENCE_LENGTH:i])
        y.append(stock_data.iloc[i]['Close'])  # Predict next close price
    
    X_tech = np.array(X_tech_seq)
    X_sent = np.array(X_sent_seq)
    y = np.array(y).reshape(-1, 1)
    
    # Train/val/test split (70/15/15)
    train_size = int(len(X_tech) * 0.7)
    val_size = int(len(X_tech) * 0.15)
    
    # Return numpy arrays (predictor classes handle conversion to tensors)
    return {
        'X_tech_train': X_tech[:train_size],
        'X_tech_val': X_tech[train_size:train_size+val_size],
        'X_tech_test': X_tech[train_size+val_size:],
        'X_sent_train': X_sent[:train_size],
        'X_sent_val': X_sent[train_size:train_size+val_size],
        'X_sent_test': X_sent[train_size+val_size:],
        'y_train': y[:train_size],
        'y_val': y[train_size:train_size+val_size],
        'y_test': y[train_size+val_size:]
    }

def train_early_fusion(stock, data):
    """Train Early Fusion Transformer"""
    print(f"\n  Training Early Fusion Transformer...")
    # Instantiate predictor (trainer wrapper)
    predictor = EarlyFusionPredictor(
        technical_size=TECHNICAL_FEATURES,
        sentiment_size=SENTIMENT_FEATURES,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dropout=0.1,
        batch_size=BATCH_SIZE
    )

    # Train (predictor expects numpy arrays)
    predictor.train(
        data['X_tech_train'], data['X_sent_train'], data['y_train'],
        X_technical_val=data['X_tech_val'], X_sentiment_val=data['X_sent_val'], y_val=data['y_val'],
        epochs=EPOCHS
    )

    # Predict on test set and evaluate
    y_pred = predictor.predict(data['X_tech_test'], data['X_sent_test'])
    metrics = predictor.evaluate(data['y_test'], y_pred)

    # Save model
    model_path = f"results/{stock}_early_fusion.pt"
    predictor.save_model(model_path)

    return metrics

def train_late_fusion(stock, data):
    """Train Late Fusion Transformer"""
    print(f"\n  Training Late Fusion Transformer...")
    predictor = LateFusionPredictor(
        technical_size=TECHNICAL_FEATURES,
        sentiment_size=SENTIMENT_FEATURES,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dropout=0.1,
        batch_size=BATCH_SIZE
    )

    predictor.train(
        data['X_tech_train'], data['X_sent_train'], data['y_train'],
        X_technical_val=data['X_tech_val'], X_sentiment_val=data['X_sent_val'], y_val=data['y_val'],
        epochs=EPOCHS
    )

    y_pred = predictor.predict(data['X_tech_test'], data['X_sent_test'])
    metrics = predictor.evaluate(data['y_test'], y_pred)

    model_path = f"results/{stock}_late_fusion.pt"
    predictor.save_model(model_path)

    return metrics

def train_attention_fusion(stock, data):
    """Train Attention Fusion Transformer"""
    print(f"\n  Training Attention Fusion Transformer...")
    predictor = AttentionFusionPredictor(
        technical_size=TECHNICAL_FEATURES,
        sentiment_size=SENTIMENT_FEATURES,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dropout=0.1,
        batch_size=BATCH_SIZE
    )

    predictor.train(
        data['X_tech_train'], data['X_sent_train'], data['y_train'],
        X_technical_val=data['X_tech_val'], X_sentiment_val=data['X_sent_val'], y_val=data['y_val'],
        epochs=EPOCHS
    )

    y_pred = predictor.predict(data['X_tech_test'], data['X_sent_test'])
    metrics = predictor.evaluate(data['y_test'], y_pred)

    model_path = f"results/{stock}_attention_fusion.pt"
    predictor.save_model(model_path)

    return metrics

def main():
    print("=" * 80)
    print("HYBRID MODEL TRAINING - ALL 3 FUSION STRATEGIES")
    print("=" * 80)
    print(f"\nModels: Early Fusion | Late Fusion | Attention Fusion")
    print(f"Stocks: {len(STOCKS)}")
    print(f"Epochs: {EPOCHS} per model")
    print(f"Features: {TECHNICAL_FEATURES} technical + {SENTIMENT_FEATURES} sentiment")
    
    all_results = []
    
    for i, stock in enumerate(STOCKS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(STOCKS)}] TRAINING {stock}")
        print('='*80)
        
        try:
            # Prepare data
            data = prepare_hybrid_data(stock)
            
            # Train all 3 models
            early_metrics = train_early_fusion(stock, data)
            early_metrics['Stock'] = stock
            early_metrics['Model'] = 'Early_Fusion'
            all_results.append(early_metrics)
            
            late_metrics = train_late_fusion(stock, data)
            late_metrics['Stock'] = stock
            late_metrics['Model'] = 'Late_Fusion'
            all_results.append(late_metrics)
            
            attention_metrics = train_attention_fusion(stock, data)
            attention_metrics['Stock'] = stock
            attention_metrics['Model'] = 'Attention_Fusion'
            all_results.append(attention_metrics)
            
            print(f"\n  ✓ {stock} complete - 3 models trained")
            
        except Exception as e:
            print(f"\n  ✗ Error training {stock}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/hybrid_training_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"\nTotal models trained: {len(results_df)}")
    print(f"Expected: {len(STOCKS) * 3} (9 stocks × 3 models)")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print("\nAverage Performance by Model:")
    print(results_df.groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean())
    
    print("\n" + "=" * 80)
    print("HYBRID MODEL TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
