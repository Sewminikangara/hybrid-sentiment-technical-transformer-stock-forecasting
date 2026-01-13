"""
Merge Sentiment Data with Technical Indicators
Prepare dataset for hybrid model training with proper normalization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_and_merge_data():
    """Merge sentiment scores with technical indicators"""
    
    print("=" * 80)
    print("MERGING SENTIMENT + TECHNICAL DATA FOR HYBRID MODELS")
    print("=" * 80)
    
    # Load technical indicators
    print("\n[1/5] Loading technical indicators...")
    tech_file = 'data_processed/technical/technical_indicators_all_stocks_with_cse_20260104_232250.csv'
    tech_df = pd.read_csv(tech_file)
    
    # Check if Date is in index or column
    if 'Date' not in tech_df.columns:
        tech_df = tech_df.reset_index()
        if 'index' in tech_df.columns:
            tech_df = tech_df.rename(columns={'index': 'Date'})
    
    # Convert dates and remove timezone for consistency
    tech_df['Date'] = pd.to_datetime(tech_df['Date'], format='mixed', utc=True).dt.tz_localize(None)
    
    # Rename Ticker to Stock for consistency
    if 'Ticker' in tech_df.columns:
        tech_df = tech_df.rename(columns={'Ticker': 'Stock'})
    
    print(f"  ✓ Loaded {len(tech_df)} technical indicator records")
    print(f"  ✓ Date range: {tech_df['Date'].min().date()} to {tech_df['Date'].max().date()}")
    
    # Load sentiment data
    print("\n[2/5] Loading sentiment data...")
    sent_file = 'data_raw/sentiment/sentiment_all_stocks_20260107_133019.csv'
    sent_df = pd.read_csv(sent_file)
    sent_df['date'] = pd.to_datetime(sent_df['date']).dt.tz_localize(None)
    print(f"  ✓ Loaded {len(sent_df)} sentiment records")
    print(f"  ✓ Date range: {sent_df['date'].min().date()} to {sent_df['date'].max().date()}")
    
    # Merge data
    print("\n[3/5] Merging datasets...")
    merged_data = []
    
    for stock in tech_df['Stock'].unique():
        print(f"\n  Processing {stock}...")
        
        # Get stock-specific data
        stock_tech = tech_df[tech_df['Stock'] == stock].copy()
        stock_sent = sent_df[sent_df['stock'] == stock].copy()
        
        print(f"    Technical records: {len(stock_tech)}")
        print(f"    Sentiment records: {len(stock_sent)}")
        
        # Merge on date (convert both to date only for matching)
        stock_tech['date_only'] = stock_tech['Date'].dt.date
        stock_sent['date_only'] = stock_sent['date'].dt.date
        
        merged = pd.merge(
            stock_tech,
            stock_sent,
            left_on='date_only',
            right_on='date_only',
            how='left'
        )
        
        # Keep original Date column, drop helper columns
        merged = merged.drop(columns=['date_only', 'date'], errors='ignore')
        
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
        
        # Remove any rows with NaN or inf values in technical features
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged = merged.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with forward fill, then backward fill, then 0
        for col in numeric_cols:
            if col not in ['Date']:
                merged[col] = merged[col].ffill().bfill().fillna(0)
        
        print(f"    Merged records: {len(merged)}")
        print(f"    Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
        
        merged_data.append(merged)
    
    # Combine all stocks
    final_df = pd.concat(merged_data, ignore_index=True)
    
    # Normalize technical and sentiment features (keep Date, Stock, and categorical columns)
    print("\n[4/5] Normalizing features...")
    
    exclude_cols = ['Date', 'Stock', 'stock', 'source', 'sentiment_label', 'confidence']
    feature_cols = [c for c in final_df.columns if c not in exclude_cols and final_df[c].dtype in ['float64', 'int64']]
    
    # Simple normalization using min-max scaling per feature
    for col in feature_cols:
        col_min = final_df[col].min()
        col_max = final_df[col].max()
        if col_max > col_min:
            final_df[col] = (final_df[col] - col_min) / (col_max - col_min)
        else:
            final_df[col] = 0.0
    
    print(f"  ✓ Normalized {len(feature_cols)} features")
    
    # Save
    print("\n[5/5] Saving merged dataset...")
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
