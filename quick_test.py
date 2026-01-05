#!/usr/bin/env python3

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.train import ModelTrainer

def main():
    print("="*70)
    print("STOCK PRICE PREDICTION - QUICK START")
    print("="*70)
    
    # Configuration
    data_path = 'data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv'
    ticker = 'AAPL'
    epochs = 20  # Use 20 for quick test, 50-100 for production
    
    print(f"\nConfiguration:")
    print(f"  Stock: {ticker}")
    print(f"  Epochs: {epochs}")
    print(f"  Data: {data_path}")
    print(f"\n{'='*70}\n")
    
    # Initialize trainer
    trainer = ModelTrainer(data_path)
    
    # Prepare data
    print(f"Step 1: Preparing data for {ticker}...")
    data, preprocessor = trainer.prepare_data(ticker, sequence_length=60)
    
    # Train LSTM baseline
    print(f"\nStep 2: Training LSTM baseline...")
    print(f"This will take ~2-5 minutes with {epochs} epochs")
    lstm_results = trainer.train_lstm(ticker, data, epochs=epochs)
    
    # Train Technical Transformer
    print(f"\nStep 3: Training Technical Transformer...")
    print(f"This will take ~3-7 minutes with {epochs} epochs")
    transformer_results = trainer.train_technical_transformer(ticker, data, epochs=epochs)
    
    # Save results
    print(f"\nStep 4: Saving results...")
    trainer.save_results()
    
    # Display summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{ticker} - LSTM Baseline:")
    for metric, value in lstm_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\n{ticker} - Technical Transformer:")
    for metric, value in transformer_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\n{'='*70}")
    print("FILES CREATED:")
    print(f"{'='*70}")
    print(f"  Models: models/{ticker}_lstm.pt")
    print(f"  Models: models/{ticker}_technical_transformer.pt")
    print(f"  Results: results/training_results_*.json")
    print(f"\n✓ Training complete! Check the results/ and models/ directories")
    print(f"\nNext steps:")
    print(f"  1. View results in results/training_results_*.json")
    print(f"  2. Train on more stocks by editing this file")
    print(f"  3. Increase epochs to 50-100 for production training")
    print(f"  4. Run scripts/evaluate.py to generate visualizations")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
