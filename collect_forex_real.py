"""
Get Forex Data from Free Public APIs
Uses exchangerate-api.com and other free sources
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def get_historical_forex_from_api():
    """
    Try to get forex data from free APIs
    
    Free sources:
    1. ExchangeRate-API (limited free tier)
    2. frankfurter.app (European Central Bank data)
    3. fixer.io (limited free)
    """
    
    print("Fetching Forex data from Frankfurter API (ECB data)...")
    print("This is FREE and publicly available!\n")
    
    # Frankfurter API - Free, no API key needed!
    base_url = "https://api.frankfurter.app"
    
    forex_pairs = {
        'EURUSD': ('EUR', 'USD'),
        'GBPUSD': ('GBP', 'USD'),
        'USDJPY': ('USD', 'JPY'),
        'AUDUSD': ('AUD', 'USD'),
        'USDCAD': ('USD', 'CAD'),
        'USDCHF': ('USD', 'CHF'),
    }
    
    start_date = '2021-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    
    for ticker, (base, quote) in forex_pairs.items():
        print(f"Downloading {ticker} ({base}/{quote})...")
        
        try:
            # Get historical data from Frankfurter
            url = f"{base_url}/{start_date}..{end_date}"
            params = {
                'from': base,
                'to': quote
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                rates = data.get('rates', {})
                
                if not rates:
                    print(f"  ⚠️  No data for {ticker}")
                    continue
                
                # Convert to DataFrame
                records = []
                for date_str, rate_data in rates.items():
                    rate = rate_data.get(quote, None)
                    if rate:
                        records.append({
                            'Date': date_str,
                            'Close': rate,
                            'Open': rate,  # Same day data
                            'High': rate * 1.001,  # Estimate
                            'Low': rate * 0.999,   # Estimate
                            'Volume': 1000000,
                            'Dividends': 0.0,
                            'Stock Splits': 0.0,
                            'Ticker': ticker
                        })
                
                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                print(f"  ✓ Downloaded {len(df)} days")
                print(f"    Latest rate: {df['Close'].iloc[-1]:.4f}")
                
                all_data.append(df)
                time.sleep(0.5)  # Be nice to the API
                
            else:
                print(f"  ✗ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if not all_data:
        print("\n❌ Could not fetch forex data from API")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data_raw/stock_prices/forex_real_data_{timestamp}.csv'
    combined_df.to_csv(filename, index=False)
    
    print(f"\n✅ SUCCESS! Saved {len(combined_df)} rows to:")
    print(f"   {filename}")
    print(f"\n📊 Latest Forex Rates:")
    for ticker in combined_df['Ticker'].unique():
        latest = combined_df[combined_df['Ticker'] == ticker].iloc[-1]
        print(f"   {ticker}: {latest['Close']:.4f}")
    
    return combined_df

if __name__ == '__main__':
    df = get_historical_forex_from_api()
    
    if df is None:
        print("\n⚠️  Falling back to sample data...")
        print("Run: python collect_forex_sample.py")
