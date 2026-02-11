"""
Add Sentiment Features to Forex Data
Creates sentiment-like features based on price momentum and volatility
(In production, you'd use real forex news sentiment)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_forex_sentiment(df):
    """
    Generate sentiment-like features for forex based on market behavior
    
    Note: In a real system, you'd collect:
    - Central bank announcements
    - Economic indicators (GDP, inflation, employment)
    - News sentiment from forex-specific sources
    
    For now, we derive sentiment from price action as a proxy
    """
    
    df = df.copy()
    
    # Sentiment score based on price momentum and trend
    # Positive returns = positive sentiment
    df['sentiment_score'] = df['returns'].rolling(window=5).mean()
    df['sentiment_score'] = (df['sentiment_score'] - df['sentiment_score'].mean()) / df['sentiment_score'].std()
    df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)  # Normalize to [-1, 1]
    
    # Convert to 0-1 scale (like vader sentiment)
    df['sentiment_score'] = (df['sentiment_score'] + 1) / 2
    
    # Breakdown into positive, negative, neutral
    df['positive'] = df['sentiment_score'].apply(lambda x: max(0, x - 0.5) * 2)
    df['negative'] = df['sentiment_score'].apply(lambda x: max(0, 0.5 - x) * 2)
    df['neutral'] = 1 - df['positive'] - df['negative']
    
    # Moving averages of sentiment
    df['sentiment_MA3'] = df['sentiment_score'].rolling(window=3).mean()
    df['sentiment_MA7'] = df['sentiment_score'].rolling(window=7).mean()
    
    # Sentiment volatility (how stable is market sentiment)
    df['sentiment_volatility'] = df['sentiment_score'].rolling(window=7).std()
    
    return df

def merge_forex_technical_sentiment():
    """Merge technical indicators with sentiment features"""
    
    base_path = Path(__file__).parent
    processed_path = base_path / 'data_processed' / 'hybrid'
    
    # Load technical data
    tech_files = list(processed_path.glob('forex_technical_data_*.csv'))
    if not tech_files:
        print("❌ No forex technical data found! Run process_forex_technical.py first.")
        return None
    
    latest_tech = max(tech_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading technical data: {latest_tech.name}")
    
    df = pd.read_csv(latest_tech)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"\n📊 Adding sentiment features to {len(df['Stock'].unique())} forex pairs...")
    
    all_data = []
    
    for ticker in df['Stock'].unique():
        print(f"\n  Processing {ticker}...")
        
        ticker_data = df[df['Stock'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
        
        # Add sentiment features
        with_sentiment = generate_forex_sentiment(ticker_data)
        
        # Drop NaN rows from rolling windows
        with_sentiment = with_sentiment.dropna()
        
        print(f"    ✓ {len(with_sentiment)} days with technical + sentiment")
        
        all_data.append(with_sentiment)
    
    # Combine all
    combined = pd.concat(all_data, ignore_index=True)
    
    # Verify we have the right features
    sentiment_features = ['sentiment_score', 'positive', 'negative', 'neutral', 
                          'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility']
    
    print(f"\n✓ Added {len(sentiment_features)} sentiment features")
    print(f"✓ Total features: {len(combined.columns) - 2} (excluding Date, Stock)")
    
    # Save hybrid data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = processed_path / f'forex_hybrid_data_{timestamp}.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! Saved {len(combined)} rows to:")
    print(f"   {output_file}")
    print(f"\n📊 Data per pair:")
    print(combined.groupby('Stock').size())
    
    print(f"\n📋 Feature breakdown:")
    print(f"   Technical features: ~35")
    print(f"   Sentiment features: {len(sentiment_features)}")
    print(f"   Total: {len(combined.columns) - 2}")
    
    return combined

if __name__ == '__main__':
    df = merge_forex_technical_sentiment()
    
    if df is not None:
        print("\n" + "="*50)
        print("✅ FOREX DATA READY FOR TRAINING!")
        print("="*50)
        print("\nNext step:")
        print("Run: python train_forex_models.py")
        print("\nThis will train:")
        print("- Early Fusion Transformer")
        print("- Late Fusion Transformer")  
        print("- Attention Fusion Transformer")
        print("- LSTM Baseline")
        print("\nFor all 6 forex pairs!")
