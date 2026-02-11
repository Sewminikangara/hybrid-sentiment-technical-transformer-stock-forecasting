"""
Collect Forex (Currency Pairs) Data
Downloads historical forex data for major currency pairs
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

# Major Forex Pairs - Try different formats
FOREX_PAIRS = {
    'EUR=X': 'EUR/USD',
    'GBP=X': 'GBP/USD', 
    'JPY=X': 'USD/JPY',
    'AUD=X': 'AUD/USD',
    'CAD=X': 'USD/CAD',
    'CHF=X': 'USD/CHF',
}

def collect_forex_data(start_date='2021-01-01', end_date=None):
    """Download forex data from Yahoo Finance"""
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Collecting Forex data from {start_date} to {end_date}...")
    
    all_data = []
    
    for symbol, name in FOREX_PAIRS.items():
        print(f"\nDownloading {name} ({symbol})...")
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"  ⚠️  No data found for {name}")
                continue
            
            # Add ticker column
            df['Ticker'] = symbol
            df.reset_index(inplace=True)
            
            print(f"  ✓ Downloaded {len(df)} days of data")
            all_data.append(df)
            
        except Exception as e:
            print(f"  ✗ Error downloading {name}: {e}")
    
    if not all_data:
        print("\n❌ No forex data collected!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data_raw/stock_prices/forex_data_{timestamp}.csv'
    combined_df.to_csv(filename, index=False)
    
    print(f"\n✅ Saved {len(combined_df)} rows to: {filename}")
    print(f"   Pairs: {', '.join(FOREX_PAIRS.values())}")
    
    return combined_df

if __name__ == '__main__':
    # Collect data from 2021 to now (same as your stock data)
    df = collect_forex_data(start_date='2021-01-01')
    
    if df is not None:
        print("\n📊 Data Summary:")
        print(df.groupby('Ticker').size())
        print("\nLatest prices:")
        for ticker in df['Ticker'].unique():
            latest = df[df['Ticker'] == ticker].iloc[-1]
            print(f"  {ticker}: {latest['Close']:.4f}")
