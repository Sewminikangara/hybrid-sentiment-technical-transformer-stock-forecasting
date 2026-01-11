"""
Merge Sentiment Data with Technical Indicators
Prepare dataset for hybrid model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_and_merge_data():
    """Merge sentiment scores with technical indicators"""
    
    print("=" * 80)
    print("MERGING SENTIMENT + TECHNICAL DATA FOR HYBRID MODELS")
    print("=" * 80)
    
    # Load technical indicators
    print("\n[1/4] Loading technical indicators...")
    tech_file = 'data_processed/technical/technical_indicators_all_stocks_20260104_232310.csv'
    tech_df = pd.read_csv(tech_file, index_col=0)
    tech_df.index = pd.to_datetime(tech_df.index)
    tech_df = tech_df.reset_index()
    tech_df.columns = ['Date'] + list(tech_df.columns[1:])
    
    # Rename Ticker to Stock for consistency
    if 'Ticker' in tech_df.columns:
        tech_df = tech_df.rename(columns={'Ticker': 'Stock'})
    
    print(f"  ✓ Loaded {len(tech_df)} technical indicator records")
    print(f"  Columns: {tech_df.columns.tolist()[:5]}...")
    
    # Load sentiment data
    print("\n[2/4] Loading sentiment data...")
    sent_file = 'data_raw/sentiment/sentiment_all_stocks_20260107_133019.csv'
    sent_df = pd.read_csv(sent_file)
    sent_df['date'] = pd.to_datetime(sent_df['date'])
    print(f"  ✓ Loaded {len(sent_df)} sentiment records")
    
    # Merge data
    print("\n[3/4] Merging datasets...")
    merged_data = []
    
    for stock in tech_df['Stock'].unique():
        print(f"\n  Processing {stock}...")
        
        # Get stock-specific data
        stock_tech = tech_df[tech_df['Stock'] == stock].copy()
        stock_sent = sent_df[sent_df['stock'] == stock].copy()
        
        print(f"    Technical records: {len(stock_tech)}")
        print(f"    Sentiment records: {len(stock_sent)}")
        
        # Merge on date
        merged = pd.merge(
            stock_tech,
            stock_sent,
            left_on='Date',
            right_on='date',
            how='left'
        )
        
        # Fill missing sentiment with neutral (0.0)
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0.0)
        merged['sentiment_label'] = merged['sentiment_label'].fillna('neutral')
        merged['confidence'] = merged['confidence'].fillna(0.5)
        
        # Add sentiment features
        merged['sentiment_positive'] = (merged['sentiment_label'] == 'positive').astype(int)
        merged['sentiment_negative'] = (merged['sentiment_label'] == 'negative').astype(int)
        merged['sentiment_neutral'] = (merged['sentiment_label'] == 'neutral').astype(int)
        
        # Rolling sentiment features (3-day, 7-day averages)
        merged = merged.sort_values('Date')
        merged['sentiment_ma3'] = merged['sentiment_score'].rolling(window=3, min_periods=1).mean()
        merged['sentiment_ma7'] = merged['sentiment_score'].rolling(window=7, min_periods=1).mean()
        merged['sentiment_volatility'] = merged['sentiment_score'].rolling(window=7, min_periods=1).std().fillna(0)
        
        print(f"    Merged records: {len(merged)}")
        print(f"    Features: Technical({len([c for c in stock_tech.columns if c not in ['Date', 'Stock']])}) + Sentiment(7) = {len([c for c in merged.columns if c not in ['Date', 'Stock', 'stock', 'date', 'source']])}")
        
        merged_data.append(merged)
    
    # Combine all stocks
    final_df = pd.concat(merged_data, ignore_index=True)
    
    # Save
    print("\n[4/4] Saving merged dataset...")
    output_dir = Path('data_processed/hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'hybrid_data_all_stocks_{timestamp}.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n  ✓ Saved to: {output_file}")
    print(f"  ✓ Total records: {len(final_df)}")
    print(f"  ✓ Total features: {len(final_df.columns)}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    
    print("\nRecords per stock:")
    print(final_df.groupby('Stock').size())
    
    print("\nSentiment distribution:")
    print(final_df['sentiment_label'].value_counts())
    
    print("\nFeature categories:")
    tech_features = [c for c in final_df.columns if c not in ['Date', 'Stock', 'stock', 'date', 'source', 'sentiment_score', 'sentiment_label', 'confidence', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']]
    sent_features = ['sentiment_score', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']
    
    print(f"  Technical indicators: {len(tech_features)}")
    print(f"  Sentiment features: {len(sent_features)}")
    print(f"  Total predictive features: {len(tech_features) + len(sent_features)}")
    
    print("\n" + "=" * 80)
    print("HYBRID DATA MERGE COMPLETE")
    print("=" * 80)
    
    return output_file

if __name__ == "__main__":
    load_and_merge_data()
