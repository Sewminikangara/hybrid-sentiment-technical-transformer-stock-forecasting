
import pandas as pd
import numpy as np
from pathlib import Path
import ta
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator

def create_output_directory():
    """Create directory for processed technical data"""
    base_path = Path(__file__).parent.parent / 'data_processed' / 'technical'
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def load_stock_data(filepath):
    """Load stock price data from CSV"""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"✓ Loaded data: {len(df)} records")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None

def calculate_moving_averages(df):
    """Calculate Simple and Exponential Moving Averages"""
    print("  Calculating Moving Averages...")
    
    # Simple Moving Averages
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    
    # Exponential Moving Averages
    df['EMA_12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
    
    return df

def calculate_rsi(df):
    """Calculate Relative Strength Index"""
    print("  Calculating RSI...")
    
    rsi_indicator = RSIIndicator(df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    return df

def calculate_macd(df):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    print("  Calculating MACD...")
    
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    return df

def calculate_bollinger_bands(df):
    """Calculate Bollinger Bands"""
    print("  Calculating Bollinger Bands...")
    
    bollinger = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Width'] = bollinger.bollinger_wband()
    
    return df

def calculate_stochastic(df):
    """Calculate Stochastic Oscillator"""
    print("  Calculating Stochastic Oscillator...")
    
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    return df

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    print("  Calculating OBV...")
    
    obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    
    return df

def calculate_price_features(df):
    """Calculate additional price-based features"""
    print("  Calculating Price Features...")
    
    # Daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Log returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Price momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Volatility (rolling standard deviation)
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volatility_30'] = df['Returns'].rolling(window=30).std()
    
    # Price range
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Close_Open_Range'] = df['Close'] - df['Open']
    
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df

def calculate_trend_features(df):
    """Calculate trend indicators"""
    print("  Calculating Trend Features...")
    
    # Price position relative to moving averages
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # Golden Cross / Death Cross signals
    df['SMA_Cross'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
    
    # Trend strength (ADX-like)
    df['Trend_Strength'] = abs(df['Price_vs_SMA20'])
    
    return df

def calculate_all_indicators(df):
    """Calculate all technical indicators"""
    print("\nCalculating Technical Indicators...")
    
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_stochastic(df)
    df = calculate_obv(df)
    df = calculate_price_features(df)
    df = calculate_trend_features(df)
    
    print("✓ All indicators calculated")
    
    return df

def process_stock_file(input_filepath, output_path):
    """Process a single stock data file and calculate all indicators"""
    print(f"\nProcessing: {input_filepath}")
    
    # Load data
    df = load_stock_data(input_filepath)
    if df is None:
        return None
    
    # Calculate indicators for each ticker
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].unique()
        processed_dfs = []
        
        for ticker in tickers:
            print(f"\n--- Processing {ticker} ---")
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = calculate_all_indicators(ticker_df)
            processed_dfs.append(ticker_df)
        
        processed_df = pd.concat(processed_dfs)
    else:
        processed_df = calculate_all_indicators(df)
    
    # Save processed data
    output_filename = output_path / f"technical_indicators_{Path(input_filepath).stem}.csv"
    processed_df.to_csv(output_filename)
    
    print(f"\n✓ Saved processed data to: {output_filename}")
    
    # Display summary
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS SUMMARY")
    print("="*60)
    print(f"Total records: {len(processed_df)}")
    print(f"Date range: {processed_df.index.min()} to {processed_df.index.max()}")
    print(f"\nIndicators calculated:")
    
    indicators = [col for col in processed_df.columns if col not in 
                 ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
    
    for indicator in indicators:
        non_null = processed_df[indicator].notna().sum()
        print(f"  {indicator}: {non_null} values")
    
    print("="*60 + "\n")
    
    return processed_df

def display_indicator_info():
    """Display information about each indicator"""
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS GUIDE")
    print("="*60)
    
    indicators_info = {
        "Moving Averages (SMA, EMA)": "Identify trends and support/resistance levels",
        "RSI (Relative Strength Index)": "Momentum oscillator (0-100), >70 overbought, <30 oversold",
        "MACD": "Trend-following momentum indicator",
        "Bollinger Bands": "Volatility bands around price",
        "Stochastic Oscillator": "Momentum indicator comparing closing price to price range",
        "OBV (On-Balance Volume)": "Volume-based indicator predicting price movements",
        "Price Features": "Returns, volatility, momentum",
        "Trend Features": "Trend direction and strength"
    }
    
    for indicator, description in indicators_info.items():
        print(f"\n{indicator}:")
        print(f"  {description}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Technical Indicators Calculator                        ║
    ║  Research: Hybrid Sentiment-Technical Transformers      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Display indicator information
    display_indicator_info()
    
    # Get input file
    data_raw_path = Path(__file__).parent.parent / 'data_raw' / 'stock_prices'
    output_path = create_output_directory()
    
    # Find most recent stock data file
    stock_files = list(data_raw_path.glob('all_stocks_*.csv'))
    
    if not stock_files:
        print("✗ No stock data files found in data_raw/stock_prices/")
        print("\nPlease run collect_stock_data.py first to collect stock data.")
    else:
        # Use the most recent file
        latest_file = max(stock_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest file: {latest_file.name}\n")
        
        # Process the file
        processed_data = process_stock_file(latest_file, output_path)
        
        if processed_data is not None:
            print("\n✓ Technical indicators calculation completed!")
            print("\nNext steps:")
            print("1. Review the processed data in data_processed/technical/")
            print("2. Use these indicators as features for your models")
            print("3. Combine with sentiment data for hybrid models")
