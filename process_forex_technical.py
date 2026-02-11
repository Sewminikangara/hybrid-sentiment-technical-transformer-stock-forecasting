"""
Process Forex Data with Technical Indicators
Adds technical indicators to forex data (similar to stock processing)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def calculate_technical_indicators(df):
    """Calculate technical indicators for forex data"""
    
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume-based (use dummy volume for forex since it's estimated)
    df['volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA']
    
    # Price position relative to high/low
    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Momentum
    df['momentum'] = df['Close'] - df['Close'].shift(4)
    df['ROC'] = df['Close'].pct_change(periods=12)
    
    return df

def process_forex_data():
    """Process all forex data with technical indicators"""
    
    base_path = Path(__file__).parent
    raw_path = base_path / 'data_raw' / 'stock_prices'
    output_path = base_path / 'data_processed' / 'hybrid'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find latest forex data file
    forex_files = list(raw_path.glob('forex_*_data_*.csv'))
    if not forex_files:
        print("❌ No forex data found! Run collect_forex_real.py first.")
        return None
    
    latest_forex = max(forex_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest_forex.name}")
    
    df = pd.read_csv(latest_forex)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    print(f"\n📊 Processing {len(df['Ticker'].unique())} forex pairs...")
    
    all_processed = []
    
    for ticker in df['Ticker'].unique():
        print(f"\n  Processing {ticker}...")
        
        ticker_data = df[df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
        
        # Calculate technical indicators
        processed = calculate_technical_indicators(ticker_data)
        
        # Add ticker column
        processed['Stock'] = ticker
        
        # Drop rows with NaN (from rolling windows)
        processed = processed.dropna()
        
        print(f"    ✓ {len(processed)} days with indicators")
        
        all_processed.append(processed)
    
    # Combine all data
    combined = pd.concat(all_processed, ignore_index=True)
    
    # Select relevant columns (35 technical features)
    feature_cols = [
        'Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume',
        'returns', 'log_returns',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
        'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'Stoch_K', 'Stoch_D', 'ATR', 'volatility',
        'volume_MA', 'volume_ratio', 'price_position',
        'momentum', 'ROC'
    ]
    
    result = combined[feature_cols].copy()
    
    # Save processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'forex_technical_data_{timestamp}.csv'
    result.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! Saved {len(result)} rows to:")
    print(f"   {output_file}")
    print(f"\n📊 Summary:")
    print(result.groupby('Stock').size())
    
    return result

if __name__ == '__main__':
    df = process_forex_data()
    
    if df is not None:
        print("\n✓ Technical indicators calculated for all forex pairs")
        print(f"✓ Total features: {len(df.columns) - 2} (Date, Stock excluded)")
        print("\nNext steps:")
        print("1. Collect sentiment data for forex news")
        print("2. Merge technical + sentiment data")
        print("3. Train transformer models")
