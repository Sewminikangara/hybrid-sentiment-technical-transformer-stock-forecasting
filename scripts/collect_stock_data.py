"""
Stock Price Data Collection Script
Collects historical stock price data from Yahoo Finance for US, India, and Sri Lanka markets
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Define stock tickers for each market
TICKERS = {
    'US': ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT'],
    'India': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
    'Sri_Lanka': ['JKH.N0000', 'COMB.N0000', 'DIAL.N0000']
}

# Data collection parameters
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

def create_data_directory():
    """Create directory structure for raw stock data"""
    base_path = Path(__file__).parent.parent / 'data_raw' / 'stock_prices'
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def collect_stock_data(ticker, start_date, end_date):
    """
    Collect stock data for a single ticker
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: Stock price data
    """
    try:
        print(f"Collecting data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"⚠️  No data found for {ticker}")
            return None
        
        # Add ticker column for identification
        data['Ticker'] = ticker
        
        print(f"✓ Successfully collected {len(data)} records for {ticker}")
        return data
    
    except Exception as e:
        print(f"✗ Error collecting data for {ticker}: {str(e)}")
        return None

def collect_all_markets():
    """Collect data for all markets and save to CSV files"""
    
    base_path = create_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_data = []
    
    print("\n" + "="*60)
    print("STOCK DATA COLLECTION STARTED")
    print("="*60)
    print(f"Period: {START_DATE} to {END_DATE}\n")
    
    for market, tickers in TICKERS.items():
        print(f"\n--- {market} Market ---")
        market_data = []
        
        for ticker in tickers:
            data = collect_stock_data(ticker, START_DATE, END_DATE)
            if data is not None:
                market_data.append(data)
                all_data.append(data)
        
        # Save market-specific file
        if market_data:
            market_df = pd.concat(market_data)
            filename = base_path / f'{market.lower()}_stocks_{timestamp}.csv'
            market_df.to_csv(filename)
            print(f"\n✓ Saved {market} data to {filename}")
    
    # Save combined file
    if all_data:
        combined_df = pd.concat(all_data)
        combined_filename = base_path / f'all_stocks_{timestamp}.csv'
        combined_df.to_csv(combined_filename)
        
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        print(f"Total records collected: {len(combined_df)}")
        print(f"Total tickers: {len(all_data)}")
        print(f"Combined data saved to: {combined_filename}")
        print("="*60 + "\n")
        
        # Display sample statistics
        print("\nSample Statistics:")
        print(combined_df.groupby('Ticker').agg({
            'Close': ['count', 'mean', 'min', 'max']
        }).round(2))
        
        return combined_df
    else:
        print("\n✗ No data collected!")
        return None

def collect_daily_update(ticker_list=None):
    """
    Collect latest data for today (for daily updates)
    
    Args:
        ticker_list (list): Optional list of specific tickers to update
    """
    if ticker_list is None:
        ticker_list = [ticker for market in TICKERS.values() for ticker in market]
    
    today = datetime.now().strftime("%Y-%m-%d")
    base_path = create_data_directory()
    
    print(f"\nCollecting daily update for {today}...")
    
    daily_data = []
    for ticker in ticker_list:
        data = collect_stock_data(ticker, today, today)
        if data is not None:
            daily_data.append(data)
    
    if daily_data:
        daily_df = pd.concat(daily_data)
        filename = base_path / f'daily_update_{today}.csv'
        daily_df.to_csv(filename)
        print(f"✓ Daily update saved to {filename}")
        return daily_df
    
    return None

def verify_data_quality(df):
    """
    Verify the quality of collected data
    
    Args:
        df (pd.DataFrame): Stock data to verify
    """
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing Values:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found ✓")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    # Check date range
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    
    # Check for anomalies (negative prices, extreme values)
    print("\nPrice Range Check:")
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        min_price = ticker_data['Close'].min()
        max_price = ticker_data['Close'].max()
        
        if min_price <= 0:
            print(f"⚠️  {ticker}: Negative or zero price detected!")
        else:
            print(f"✓ {ticker}: ${min_price:.2f} - ${max_price:.2f}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Stock Price Data Collection - Yahoo Finance            ║
    ║  Research: Hybrid Sentiment-Technical Transformers      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Collect all historical data
    data = collect_all_markets()
    
    if data is not None:
        # Verify data quality
        verify_data_quality(data)
        
        print("\n✓ Data collection completed successfully!")
        print("\nNext steps:")
        print("1. Review the collected data in data_raw/stock_prices/")
        print("2. Add Sri Lankan data (manual process):")
        print("   python scripts/process_srilanka_data.py")
        print("3. Run calculate_technical_indicators.py to compute indicators")
        print("4. Start collecting sentiment data from Reddit and News")
    else:
        print("\n✗ Data collection failed. Please check your internet connection and try again.")
