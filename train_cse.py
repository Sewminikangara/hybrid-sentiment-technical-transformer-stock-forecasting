"""
Train CSE All-Share Index (Sri Lankan Market)
"""

import os
import pandas as pd
from datetime import datetime
from scripts.train import ModelTrainer

def main():
    print("="*70)
    print("TRAINING CSE ALL-SHARE INDEX (SRI LANKA)")
    print("="*70)
    
    # Configuration
    DATA_PATH = "data_processed/technical/technical_indicators_all_stocks_20260104_232310.csv"
    STOCK = "CSEALL"
    SEQUENCE_LENGTH = 60
    EPOCHS = 50
    
    print(f"\nConfiguration:")
    print(f"   Stock: {STOCK} (Colombo Stock Exchange All-Share Index)")
    print(f"   Data: Sri Lankan market (2021-2025)")
    print(f"   Sequence Length: {SEQUENCE_LENGTH}")
    print(f"   Epochs: {EPOCHS}")
    
    # Initialize trainer
    trainer = ModelTrainer(DATA_PATH)
    
    # Prepare data
    print(f"\nPreparing data for {STOCK}...")
    data, preprocessor = trainer.prepare_data(STOCK, sequence_length=SEQUENCE_LENGTH)
    
    print(f"   Train samples: {data['X_train'].shape[0]}")
    print(f"   Test samples:  {data['X_test'].shape[0]}")
    print(f"   Features: {data['X_train'].shape[2]}")
    
    # Train LSTM
    print(f"\nTraining LSTM for {STOCK} ({EPOCHS} epochs)...")
    lstm_result = trainer.train_lstm(STOCK, data, epochs=EPOCHS)
    
    lstm_metrics = lstm_result['metrics']
    print(f"\n{STOCK} - LSTM Results:")
    print(f"   RMSE: {lstm_metrics['RMSE']:.6f}")
    print(f"   MAE: {lstm_metrics['MAE']:.6f}")
    print(f"   MAPE: {lstm_metrics['MAPE']:.2f}%")
    print(f"   Directional Accuracy: {lstm_metrics['Directional_Accuracy']:.2f}%")
    
    # Train Technical Transformer
    print(f"\nTraining Technical Transformer for {STOCK} ({EPOCHS} epochs)...")
    transformer_result = trainer.train_technical_transformer(STOCK, data, epochs=EPOCHS)
    
    transformer_metrics = transformer_result['metrics']
    print(f"\n{STOCK} - Technical Transformer Results:")
    print(f"   RMSE: {transformer_metrics['RMSE']:.6f}")
    print(f"   MAE: {transformer_metrics['MAE']:.6f}")
    print(f"   MAPE: {transformer_metrics['MAPE']:.2f}%")
    print(f"   Directional Accuracy: {transformer_metrics['Directional_Accuracy']:.2f}%")
    
    # Save combined results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    results = [{
        'stock': STOCK,
        'market': 'Sri_Lanka',
        'model_type': 'LSTM',
        'RMSE': lstm_metrics['RMSE'],
        'MAE': lstm_metrics['MAE'],
        'MAPE': lstm_metrics['MAPE'],
        'Directional_Accuracy': lstm_metrics['Directional_Accuracy']
    }, {
        'stock': STOCK,
        'market': 'Sri_Lanka',
        'model_type': 'Technical_Transformer',
        'RMSE': transformer_metrics['RMSE'],
        'MAE': transformer_metrics['MAE'],
        'MAPE': transformer_metrics['MAPE'],
        'Directional_Accuracy': transformer_metrics['Directional_Accuracy']
    }]
    
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/cse_training_results_{timestamp}.csv"
    
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    print(f"\nResults saved to: {results_path}")
    print(f"\n{'='*70}")
    print("CSE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nModels saved:")
    print(f"   results/{STOCK}_lstm.pt")
    print(f"   results/{STOCK}_technical_transformer.pt")
    print(f"\n✓ Sri Lankan market analysis ready for dissertation!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
