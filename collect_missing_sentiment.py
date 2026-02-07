"""
Collect missing sentiment data for Indian stocks
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
sys.path.append('scripts')

from collect_real_sentiment_quick import QuickSentimentCollector

# Missing stocks
MISSING_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services'
}

def main():
    print("Collecting missing sentiment data...")
    
    collector = QuickSentimentCollector()
    
    all_data = []
    
    for stock, company in MISSING_STOCKS.items():
        data = collector.collect_for_stock(stock, company)
        all_data.extend(data)
    
    if all_data:
        # Load existing data
        existing_file = 'data_raw/sentiment/real_sentiment_all_stocks_20260205_215434.csv'
        existing_df = pd.read_csv(existing_file)
        
        # Combine
        new_df = pd.DataFrame(all_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.sort_values('date')
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'data_raw/sentiment/real_sentiment_all_stocks_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Updated file saved: {output_file}")
        print(f"✓ Total records: {len(combined_df)}")
        print("\nRecords per stock:")
        print(combined_df['stock'].value_counts())

if __name__ == "__main__":
    main()
