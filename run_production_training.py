#!/usr/bin/env python3


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.train import train_all_models

def main():
    print("="*70)
    print("STOCK PRICE PREDICTION - PRODUCTION TRAINING")
    print("="*70)
    
    # Configuration
    data_path = 'data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv'
    
    # Choose stocks to train on
    # US stocks
    us_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    # India stocks  
    india_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    
    # Choose which market to train
    print("\nSelect market to train:")
    print("  1. US stocks (5 stocks)")
    print("  2. India stocks (3 stocks)")
    print("  3. All stocks (8 stocks)")
    
    choice = input("\nEnter choice (1-3) or press Enter for option 1: ").strip()
    
    if choice == '2':
        tickers = india_tickers
        market = "India"
    elif choice == '3':
        tickers = us_tickers + india_tickers
        market = "All Markets"
    else:
        tickers = us_tickers
        market = "US"
    
    epochs = 50  # Production epochs
    
    print(f"\nConfiguration:")
    print(f"  Market: {market}")
    print(f"  Stocks: {', '.join(tickers)}")
    print(f"  Epochs: {epochs}")
    print(f"  Data: {data_path}")
    print(f"\n⚠️  This will take 1-3 hours depending on your hardware")
    
    confirm = input("\nProceed? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("Training cancelled")
        return
    
    print(f"\n{'='*70}\n")
    print("Starting training...")
    print("You can monitor progress in real-time below\n")
    
    # Train all models
    trainer = train_all_models(tickers, data_path, epochs=epochs)
    
    print(f"\n{'='*70}")
    print("✓ ALL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nFiles created:")
    print(f"  Models: models/*.pt ({len(tickers) * 2} model files)")
    print(f"  Results: results/training_results_*.json")
    print(f"  Summary: results/results_summary.csv")
    print(f"\nNext steps:")
    print(f"  1. View results: cat results/results_summary.csv")
    print(f"  2. Generate visualizations: python scripts/evaluate.py")
    print(f"  3. Use models for predictions")
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
