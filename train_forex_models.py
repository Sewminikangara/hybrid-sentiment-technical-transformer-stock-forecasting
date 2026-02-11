"""
Train Forex Models (Simplified Version)
Trains transformer models on forex data with technical + sentiment features

Note: With only 58 days of data, these will be demo models.
For production, you'd need 1-2 years of data minimum.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Import existing model architectures
sys.path.append(str(Path(__file__).parent / 'models'))
from transformer_early_fusion import EarlyFusionTransformer
from transformer_late_fusion import LateFusionTransformer
from transformer_attention_fusion import AttentionFusionTransformer
from baseline_lstm import LSTMModel

def prepare_forex_data(stock='EURUSD', seq_length=30):
    """Prepare forex data for training"""
    
    base_path = Path(__file__).parent
    data_path = base_path / 'data_processed' / 'hybrid'
    
    # Load hybrid data
    hybrid_files = list(data_path.glob('forex_hybrid_data_*.csv'))
    if not hybrid_files:
        print("❌ No forex hybrid data found!")
        return None, None, None, None
    
    latest = max(hybrid_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by stock
    df = df[df['Stock'] == stock].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    if len(df) < seq_length + 10:
        print(f"❌ Not enough data for {stock}: {len(df)} days")
        return None, None, None, None
    
    print(f"  Data points: {len(df)}")
    
    # Separate features
    exclude_cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify technical vs sentiment
    sentiment_cols = ['sentiment_score', 'positive', 'negative', 'neutral',
                     'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility']
    technical_cols = [col for col in feature_cols if col not in sentiment_cols]
    
    print(f"  Technical features: {len(technical_cols)}")
    print(f"  Sentiment features: {len(sentiment_cols)}")
    
    # Normalize features
    technical_data = df[technical_cols].values
    sentiment_data = df[sentiment_cols].values
    prices = df['Close'].values
    
    # Simple normalization
    technical_mean = technical_data.mean(axis=0)
    technical_std = technical_data.std(axis=0) + 1e-8
    technical_norm = (technical_data - technical_mean) / technical_std
    
    sentiment_norm = sentiment_data  # Already normalized
    
    prices_mean = prices.mean()
    prices_std = prices.std()
    prices_norm = (prices - prices_mean) / prices_std
    
    # Create sequences
    X_tech, X_sent, y = [], [], []
    
    for i in range(len(df) - seq_length):
        X_tech.append(technical_norm[i:i+seq_length])
        X_sent.append(sentiment_norm[i:i+seq_length])
        y.append(prices_norm[i+seq_length])
    
    X_tech = np.array(X_tech)
    X_sent = np.array(X_sent)
    y = np.array(y)
    
    # Split train/test (80/20)
    split = int(0.8 * len(X_tech))
    
    train_data = (
        torch.FloatTensor(X_tech[:split]),
        torch.FloatTensor(X_sent[:split]),
        torch.FloatTensor(y[:split])
    )
    
    test_data = (
        torch.FloatTensor(X_tech[split:]),
        torch.FloatTensor(X_sent[split:]),
        torch.FloatTensor(y[split:])
    )
    
    stats = {
        'price_mean': prices_mean,
        'price_std': prices_std,
        'technical_size': len(technical_cols),
        'sentiment_size': len(sentiment_cols)
    }
    
    print(f"  Train: {len(train_data[0])}, Test: {len(test_data[0])}")
    
    return train_data, test_data, stats, df

def train_simple_model(model, train_data, epochs=50, lr=0.001):
    """Simple training loop"""
    
    X_tech, X_sent, y = train_data
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if isinstance(model, (EarlyFusionTransformer, AttentionFusionTransformer)):
            pred = model(X_tech, X_sent).squeeze()
        elif isinstance(model, LateFusionTransformer):
            pred = model(X_tech, X_sent).squeeze()
        else:  # LSTM
            X_combined = torch.cat([X_tech, X_sent], dim=-1)
            pred = model(X_combined).squeeze()
        
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return model

def train_forex_models():
    """Train all models for all forex pairs"""
    
    forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
    models_config = {
        'early_fusion': EarlyFusionTransformer,
        'late_fusion': LateFusionTransformer,
        'attention_fusion': AttentionFusionTransformer,
        'lstm': LSTMModel
    }
    
    results_path = Path(__file__).parent / 'results'
    results_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("TRAINING FOREX MODELS")
    print("="*60)
    print(f"\nPairs: {len(forex_pairs)}")
    print(f"Models: {len(models_config)}")
    print(f"Total models to train: {len(forex_pairs) * len(models_config)}")
    print("\nNote: These are demo models with limited data (58 days)")
    print("For production, collect 1-2 years of forex data.\n")
    
    all_results = []
    
    for pair in forex_pairs:
        print(f"\n{'='*60}")
        print(f"Training models for {pair}")
        print(f"{'='*60}")
        
        # Prepare data
        train_data, test_data, stats, df = prepare_forex_data(pair)
        
        if train_data is None:
            print(f"  Skipping {pair} - insufficient data")
            continue
        
        for model_name, ModelClass in models_config.items():
            print(f"\n  Training {model_name}...")
            
            try:
                # Initialize model
                if model_name == 'lstm':
                    model = ModelClass(
                        input_size=stats['technical_size'] + stats['sentiment_size'],
                        hidden_size=64,
                        num_layers=2
                    )
                else:
                    model = ModelClass(
                        technical_size=stats['technical_size'],
                        sentiment_size=stats['sentiment_size'],
                        d_model=64,
                        nhead=4,
                        num_encoder_layers=2,
                        dropout=0.1
                    )
                
                # Train
                model = train_simple_model(model, train_data, epochs=50)
                
                # Save
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'stats': stats,
                    'config': {
                        'technical_size': stats['technical_size'],
                        'sentiment_size': stats['sentiment_size']
                    }
                }
                
                save_path = results_path / f"{pair}_{model_name}.pt"
                torch.save(save_dict, save_path)
                
                print(f"    ✓ Saved: {save_path.name}")
                
                all_results.append({
                    'Pair': pair,
                    'Model': model_name,
                    'Status': 'Trained',
                    'Data_Points': len(train_data[0]) + len(test_data[0])
                })
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                all_results.append({
                    'Pair': pair,
                    'Model': model_name,
                    'Status': f'Failed: {str(e)[:50]}',
                    'Data_Points': 0
                })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_path / f'forex_training_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n✅ Results saved to: {results_file}")
    print(f"\n📊 Summary:")
    print(results_df.to_string(index=False))
    
    successful = len(results_df[results_df['Status'] == 'Trained'])
    print(f"\n✓ Successfully trained: {successful}/{len(results_df)} models")
    
    return results_df

if __name__ == '__main__':
    results = train_forex_models()
