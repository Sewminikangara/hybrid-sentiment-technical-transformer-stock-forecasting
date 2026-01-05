"""
Training script for all stock models
Runs LSTM and Transformer training across multiple stocks
"""

import os
import json
import pandas as pd
from datetime import datetime
from scripts.train import ModelTrainer

def main():
    print("="*70)
    print("STOCK PRICE FORECASTING - FULL TRAINING")
    print("="*70)
    
    # Configuration
    DATA_PATH = "data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv"
    SEQUENCE_LENGTH = 60
    EPOCHS = 50
    
    # Load stock data
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    stocks = df['Ticker'].unique()
    
    print(f"Found {len(stocks)} stocks: {', '.join(stocks)}")
    print(f"\nTraining Configuration:")
    print(f"   Sequence Length: {SEQUENCE_LENGTH}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Models: LSTM, Technical Transformer")
    
    # Initialize trainer
    trainer = ModelTrainer(DATA_PATH)
    
    # Results storage
    all_results = []
    
    # Train each stock
    for i, stock in enumerate(stocks, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(stocks)}] Training: {stock}")
        print(f"{'='*70}")
        
        try:
            # Prepare data
            print(f"\nPreparing data for {stock}...")
            data, preprocessor = trainer.prepare_data(stock, sequence_length=SEQUENCE_LENGTH)
            
            print(f"   Train samples: {data['X_train'].shape[0]}")
            print(f"   Test samples:  {data['X_test'].shape[0]}")
            print(f"   Features: {data['X_train'].shape[2]}")
            
            # Train LSTM
            print(f"\nTraining LSTM for {stock} ({EPOCHS} epochs)...")
            lstm_result = trainer.train_lstm(stock, data, epochs=EPOCHS)
            
            lstm_summary = {
                'stock': stock,
                'model_type': 'LSTM',
                'RMSE': lstm_result['metrics']['RMSE'],
                'MAE': lstm_result['metrics']['MAE'],
                'MAPE': lstm_result['metrics']['MAPE'],
                'Directional_Accuracy': lstm_result['metrics']['Directional_Accuracy']
            }
            all_results.append(lstm_summary)
            
            print(f"   LSTM - RMSE: {lstm_summary['RMSE']:.4f}, "
                  f"Directional Accuracy: {lstm_summary['Directional_Accuracy']:.2f}%")
            
            # Train Technical Transformer
            print(f"\nTraining Technical Transformer for {stock} ({EPOCHS} epochs)...")
            transformer_result = trainer.train_technical_transformer(stock, data, epochs=EPOCHS)
            
            transformer_summary = {
                'stock': stock,
                'model_type': 'Technical_Transformer',
                'RMSE': transformer_result['metrics']['RMSE'],
                'MAE': transformer_result['metrics']['MAE'],
                'MAPE': transformer_result['metrics']['MAPE'],
                'Directional_Accuracy': transformer_result['metrics']['Directional_Accuracy']
            }
            all_results.append(transformer_summary)
            
            print(f"   Transformer - RMSE: {transformer_summary['RMSE']:.4f}, "
                  f"Directional Accuracy: {transformer_summary['Directional_Accuracy']:.2f}%")
            
        except Exception as e:
            print(f"   Error training {stock}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING TRAINING RESULTS")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/training_results_{timestamp}.csv"
    
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("AVERAGE PERFORMANCE BY MODEL")
    print(f"{'='*70}")
    
    if len(all_results) > 0:
        summary = results_df.groupby('model_type').agg({
            'RMSE': 'mean',
            'MAE': 'mean',
            'MAPE': 'mean',
            'Directional_Accuracy': 'mean'
        }).round(4)
        
        print(summary)
        
        print(f"\n{'='*70}")
        print(f"Training completed for {len(stocks)} stocks")
        print(f"Models saved in: models/")
        print(f"Results saved in: results/")
        print(f"Graphs saved in: graphs/")
        print(f"{'='*70}\n")
    else:
        print("\nNo results to display - all training failed")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
