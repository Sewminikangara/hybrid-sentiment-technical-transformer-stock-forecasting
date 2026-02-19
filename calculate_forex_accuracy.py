"""
Calculate accuracy metrics for forex models
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.transformer_early_fusion import EarlyFusionTransformer
from models.transformer_late_fusion import LateFusionTransformer
from models.transformer_attention_fusion import AttentionFusionTransformer
from models.baseline_lstm import BaselineLSTM

def calculate_metrics(y_true, y_pred):
    """Calculate MAPE and directional accuracy"""
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional accuracy
    y_true_direction = np.diff(y_true) > 0
    y_pred_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return mape, directional_accuracy, rmse

def evaluate_forex_models():
    """Evaluate all forex models and calculate accuracy"""
    
    # Load forex data
    data_path = Path('data_processed/hybrid')
    forex_file = list(data_path.glob('forex_hybrid_data_*.csv'))
    
    if not forex_file:
        print("❌ No forex data found!")
        return
    
    df = pd.read_csv(max(forex_file, key=lambda p: p.stat().st_mtime))
    
    forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
    models = {
        'early_fusion': 'Early_Fusion',
        'late_fusion': 'Late_Fusion', 
        'attention_fusion': 'Attention_Fusion',
        'lstm': 'LSTM'
    }
    
    results = []
    
    sequence_length = 60
    
    for pair in forex_pairs:
        print(f"\n📊 Evaluating {pair}...")
        
        # Get data for this pair
        pair_data = df[df['Stock'] == pair].copy().sort_values('Date').reset_index(drop=True)
        
        if len(pair_data) < sequence_length + 10:
            print(f"  ⚠️ Not enough data for {pair}")
            continue
        
        # Define features (matching training)
        sentiment_cols = ['sentiment_score', 'positive', 'negative', 'neutral',
                         'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility']
        exclude_cols = ['Date', 'Stock', 'Close'] + sentiment_cols
        
        technical_cols = [col for col in pair_data.columns if col not in exclude_cols]
        sentiment_cols_actual = [col for col in sentiment_cols if col in pair_data.columns]
        
        # Get Close prices
        close_prices = pair_data['Close'].values
        
        # Split into train/test (last 10% for testing)
        split_idx = int(len(pair_data) * 0.9)
        test_data = pair_data.iloc[split_idx - sequence_length:].reset_index(drop=True)
        
        # Normalize features
        scaler_tech = StandardScaler()
        scaler_sent = StandardScaler()
        scaler_close = StandardScaler()
        
        # Fit on training data
        scaler_tech.fit(pair_data[technical_cols].iloc[:split_idx])
        if sentiment_cols_actual:
            scaler_sent.fit(pair_data[sentiment_cols_actual].iloc[:split_idx])
        scaler_close.fit(close_prices[:split_idx].reshape(-1, 1))
        
        # Transform test data
        technical_features = scaler_tech.transform(test_data[technical_cols])
        if sentiment_cols_actual:
            sentiment_features = scaler_sent.transform(test_data[sentiment_cols_actual])
        else:
            sentiment_features = np.zeros((len(test_data), 7))
        
        close_normalized = scaler_close.transform(close_prices[split_idx - sequence_length:].reshape(-1, 1)).flatten()
        
        for model_key, model_name in models.items():
            model_file = Path(f'results/{pair}_{model_key}.pt')
            
            if not model_file.exists():
                print(f"  ⚠️ Model not found: {model_file}")
                continue
            
            try:
                # Load model
                checkpoint = torch.load(model_file, map_location='cpu')
                
                technical_dim = len(technical_cols)
                sentiment_dim = len(sentiment_cols_actual) if sentiment_cols_actual else 7
                
                if model_key == 'early_fusion':
                    model = EarlyFusionTransformer(technical_dim, sentiment_dim)
                elif model_key == 'late_fusion':
                    model = LateFusionTransformer(technical_dim, sentiment_dim)
                elif model_key == 'attention_fusion':
                    model = AttentionFusionTransformer(technical_dim, sentiment_dim)
                else:  # lstm
                    model = BaselineLSTM(technical_dim + sentiment_dim)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                
                # Make predictions
                predictions = []
                actuals = []
                
                with torch.no_grad():
                    for i in range(sequence_length, len(test_data)):
                        # Get sequence
                        tech_seq = technical_features[i-sequence_length:i]
                        sent_seq = sentiment_features[i-sequence_length:i]
                        
                        # Convert to tensors
                        tech_tensor = torch.FloatTensor(tech_seq).unsqueeze(0)
                        sent_tensor = torch.FloatTensor(sent_seq).unsqueeze(0)
                        
                        # Predict
                        if model_key == 'lstm':
                            combined = np.concatenate([tech_seq, sent_seq], axis=1)
                            input_tensor = torch.FloatTensor(combined).unsqueeze(0)
                            pred = model(input_tensor).item()
                        else:
                            pred = model(tech_tensor, sent_tensor).item()
                        
                        predictions.append(pred)
                        actuals.append(close_normalized[i])
                
                # Denormalize
                predictions = scaler_close.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                actuals = scaler_close.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
                
                # Calculate metrics
                mape, dir_acc, rmse = calculate_metrics(actuals, predictions)
                
                print(f"  ✅ {model_name}: MAPE={mape:.2f}%, Dir_Acc={dir_acc:.1f}%, RMSE={rmse:.4f}")
                
                results.append({
                    'Pair': pair,
                    'Model': model_name,
                    'MAPE': round(mape, 2),
                    'Directional_Accuracy': round(dir_acc, 1),
                    'RMSE': round(rmse, 4),
                    'Test_Points': len(predictions)
                })
                
            except Exception as e:
                print(f"  ❌ Error with {model_name}: {str(e)}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = f'results/forex_accuracy_metrics_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n📊 Summary:")
    print(results_df.to_string())
    
    return results_df

if __name__ == '__main__':
    print("🔍 Calculating Forex Model Accuracy...\n")
    evaluate_forex_models()
