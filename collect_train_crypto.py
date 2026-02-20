"""
Cryptocurrency Data Collection, Processing, and Model Training Pipeline
Downloads historical crypto data for major pairs, processes technical indicators,
adds sentiment features, and trains all 4 transformer model architectures.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'models'))
from transformer_early_fusion import EarlyFusionTransformer
from transformer_late_fusion import LateFusionTransformer
from transformer_attention_fusion import AttentionFusionTransformer
from baseline_lstm import LSTMModel

# Major Cryptocurrency Pairs
CRYPTO_PAIRS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'SOL-USD': 'Solana',
    'XRP-USD': 'XRP',
    'ADA-USD': 'Cardano',
}

# Internal keys used for filenames (no hyphens)
CRYPTO_KEYS = {
    'BTC-USD': 'BTCUSD',
    'ETH-USD': 'ETHUSD',
    'BNB-USD': 'BNBUSD',
    'SOL-USD': 'SOLUSD',
    'XRP-USD': 'XRPUSD',
    'ADA-USD': 'ADAUSD',
}


def collect_crypto_data(start_date='2022-01-01', end_date=None):
    """Download cryptocurrency data from Yahoo Finance"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Collecting Crypto data from {start_date} to {end_date}...")
    all_data = []

    for symbol, name in CRYPTO_PAIRS.items():
        print(f"\n  Downloading {name} ({symbol})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                print(f"    No data found for {name}")
                continue
            df['Ticker'] = symbol
            df.reset_index(inplace=True)
            print(f"    Downloaded {len(df)} days of data")
            all_data.append(df)
        except Exception as e:
            print(f"    Error downloading {name}: {e}")

    if not all_data:
        print("\nNo crypto data collected!")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Save raw data
    raw_path = Path(__file__).parent / 'data_raw' / 'stock_prices'
    raw_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = raw_path / f'crypto_data_{timestamp}.csv'
    combined_df.to_csv(filename, index=False)
    print(f"\nSaved {len(combined_df)} rows to: {filename}")

    return combined_df


def calculate_technical_indicators(df):
    """Calculate technical indicators for crypto data"""
    df = df.copy()

    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    for window in [5, 10, 20, 50, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA']
    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['momentum'] = df['Close'] - df['Close'].shift(4)
    df['ROC'] = df['Close'].pct_change(periods=12)

    return df


def generate_crypto_sentiment(df):
    """Generate sentiment features based on price momentum (proxy)"""
    df = df.copy()

    df['sentiment_score'] = df['returns'].rolling(window=5).mean()
    df['sentiment_score'] = (df['sentiment_score'] - df['sentiment_score'].mean()) / (df['sentiment_score'].std() + 1e-8)
    df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
    df['sentiment_score'] = (df['sentiment_score'] + 1) / 2

    df['positive'] = df['sentiment_score'].apply(lambda x: max(0, x - 0.5) * 2)
    df['negative'] = df['sentiment_score'].apply(lambda x: max(0, 0.5 - x) * 2)
    df['neutral'] = 1 - df['positive'] - df['negative']

    df['sentiment_MA3'] = df['sentiment_score'].rolling(window=3).mean()
    df['sentiment_MA7'] = df['sentiment_score'].rolling(window=7).mean()
    df['sentiment_volatility'] = df['sentiment_score'].rolling(window=7).std()

    return df


def process_crypto_data(raw_df):
    """Process raw crypto data: technical indicators + sentiment"""
    output_path = Path(__file__).parent / 'data_processed' / 'hybrid'
    output_path.mkdir(parents=True, exist_ok=True)

    if 'Date' not in raw_df.columns and raw_df.index.name == 'Date':
        raw_df = raw_df.reset_index()
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df = raw_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    print(f"\nProcessing {len(raw_df['Ticker'].unique())} crypto pairs...")
    all_processed = []

    for ticker in raw_df['Ticker'].unique():
        key = CRYPTO_KEYS.get(ticker, ticker.replace('-', ''))
        print(f"\n  Processing {ticker} -> {key}...")

        ticker_data = raw_df[raw_df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)

        processed = calculate_technical_indicators(ticker_data)
        processed = generate_crypto_sentiment(processed)
        processed['Stock'] = key
        processed = processed.dropna()
        print(f"    {len(processed)} days with indicators + sentiment")
        all_processed.append(processed)

    combined = pd.concat(all_processed, ignore_index=True)

    feature_cols = [
        'Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume',
        'returns', 'log_returns',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
        'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'Stoch_K', 'Stoch_D', 'ATR', 'volatility',
        'volume_MA', 'volume_ratio', 'price_position',
        'momentum', 'ROC',
        'sentiment_score', 'positive', 'negative', 'neutral',
        'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility'
    ]

    existing_cols = [c for c in feature_cols if c in combined.columns]
    result = combined[existing_cols].copy()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'crypto_hybrid_data_{timestamp}.csv'
    result.to_csv(output_file, index=False)
    print(f"\nSaved hybrid data: {output_file}")
    print(f"Shape: {result.shape}")
    print(result.groupby('Stock').size())

    return result


def prepare_crypto_data(df, stock='BTCUSD', seq_length=30):
    """Prepare crypto data for training"""
    stock_df = df[df['Stock'] == stock].copy()
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)

    if len(stock_df) < seq_length + 10:
        print(f"  Not enough data for {stock}: {len(stock_df)} days")
        return None, None, None, None

    print(f"  Data points: {len(stock_df)}")

    exclude_cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in stock_df.columns if col not in exclude_cols]

    sentiment_cols = ['sentiment_score', 'positive', 'negative', 'neutral',
                      'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility']
    technical_cols = [col for col in feature_cols if col not in sentiment_cols]

    print(f"  Technical features: {len(technical_cols)}")
    print(f"  Sentiment features: {len(sentiment_cols)}")

    technical_data = stock_df[technical_cols].values
    sentiment_data = stock_df[sentiment_cols].values
    prices = stock_df['Close'].values

    technical_mean = technical_data.mean(axis=0)
    technical_std = technical_data.std(axis=0) + 1e-8
    technical_norm = (technical_data - technical_mean) / technical_std
    sentiment_norm = sentiment_data
    prices_mean = prices.mean()
    prices_std = prices.std() + 1e-8
    prices_norm = (prices - prices_mean) / prices_std

    X_tech, X_sent, y = [], [], []
    for i in range(len(stock_df) - seq_length):
        X_tech.append(technical_norm[i:i + seq_length])
        X_sent.append(sentiment_norm[i:i + seq_length])
        y.append(prices_norm[i + seq_length])

    X_tech = np.array(X_tech)
    X_sent = np.array(X_sent)
    y = np.array(y)

    split = int(0.8 * len(X_tech))
    train_data = (
        torch.FloatTensor(X_tech[:split]),
        torch.FloatTensor(X_sent[:split]),
        torch.FloatTensor(y[:split])
    )
    test_data = (
        torch.FloatTensor(X_tech[split:]),
        torch.FloatTensor(X_sent[split:]),
        torch.FloatTensor(y[split:])
    )

    stats = {
        'price_mean': prices_mean,
        'price_std': prices_std,
        'technical_size': len(technical_cols),
        'sentiment_size': len(sentiment_cols)
    }

    print(f"  Train: {len(train_data[0])}, Test: {len(test_data[0])}")
    return train_data, test_data, stats, stock_df


def train_model(model, train_data, epochs=100, lr=0.001):
    """Training loop"""
    X_tech, X_sent, y = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    model.train()

    best_loss = float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, LSTMModel):
            X_combined = torch.cat([X_tech, X_sent], dim=-1)
            pred = model(X_combined).squeeze()
        else:
            pred = model(X_tech, X_sent).squeeze()

        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    return model, best_loss


def evaluate_model(model, test_data, stats):
    """Evaluate model on test data"""
    X_tech, X_sent, y = test_data
    model.eval()
    with torch.no_grad():
        if isinstance(model, LSTMModel):
            X_combined = torch.cat([X_tech, X_sent], dim=-1)
            pred = model(X_combined).squeeze()
        else:
            pred = model(X_tech, X_sent).squeeze()

    pred_np = pred.numpy()
    y_np = y.numpy()

    pred_prices = pred_np * stats['price_std'] + stats['price_mean']
    actual_prices = y_np * stats['price_std'] + stats['price_mean']

    mse = np.mean((pred_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_prices - actual_prices))
    mape = np.mean(np.abs((actual_prices - pred_prices) / (actual_prices + 1e-8))) * 100

    pred_direction = np.sign(np.diff(np.concatenate([[actual_prices[0]], pred_prices])))
    actual_direction = np.sign(np.diff(np.concatenate([[actual_prices[0]], actual_prices])))
    dir_accuracy = np.mean(pred_direction == actual_direction) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': dir_accuracy
    }


def train_all_crypto_models(hybrid_df):
    """Train all models for all crypto pairs"""
    crypto_keys = list(CRYPTO_KEYS.values())
    models_config = {
        'early_fusion': EarlyFusionTransformer,
        'late_fusion': LateFusionTransformer,
        'attention_fusion': AttentionFusionTransformer,
        'lstm': LSTMModel
    }

    results_path = Path(__file__).parent / 'results'
    results_path.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING CRYPTO MODELS")
    print("=" * 60)
    print(f"Pairs: {len(crypto_keys)}")
    print(f"Models: {len(models_config)}")
    print(f"Total models to train: {len(crypto_keys) * len(models_config)}")

    all_results = []

    for pair in crypto_keys:
        print(f"\n{'=' * 60}")
        print(f"Training models for {pair}")
        print(f"{'=' * 60}")

        result = prepare_crypto_data(hybrid_df, pair)
        if result[0] is None:
            print(f"  Skipping {pair} - insufficient data")
            continue

        train_data, test_data, stats, _ = result

        for model_name, ModelClass in models_config.items():
            print(f"\n  Training {model_name}...")
            try:
                if model_name == 'lstm':
                    model = ModelClass(
                        input_size=stats['technical_size'] + stats['sentiment_size'],
                        hidden_size=64,
                        num_layers=2
                    )
                else:
                    model = ModelClass(
                        technical_size=stats['technical_size'],
                        sentiment_size=stats['sentiment_size'],
                        d_model=64,
                        nhead=4,
                        num_encoder_layers=2,
                        dropout=0.1
                    )

                model, best_loss = train_model(model, train_data, epochs=100)

                metrics = evaluate_model(model, test_data, stats)

                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'stats': stats,
                    'config': {
                        'technical_size': stats['technical_size'],
                        'sentiment_size': stats['sentiment_size'],
                        'asset_type': 'crypto'
                    }
                }

                save_path = results_path / f"{pair}_{model_name}.pt"
                torch.save(save_dict, save_path)
                print(f"    Saved: {save_path.name}")
                print(f"    MAPE: {metrics['MAPE']:.2f}%, Dir Acc: {metrics['Directional_Accuracy']:.1f}%")

                all_results.append({
                    'Pair': pair,
                    'Model': model_name.replace('_', ' ').title().replace(' ', '_'),
                    'RMSE': round(metrics['RMSE'], 4),
                    'MAE': round(metrics['MAE'], 4),
                    'MAPE': round(metrics['MAPE'], 2),
                    'Directional_Accuracy': round(metrics['Directional_Accuracy'], 1),
                    'Best_Loss': round(best_loss, 6),
                    'Status': 'Trained'
                })

            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'Pair': pair,
                    'Model': model_name,
                    'RMSE': 0, 'MAE': 0, 'MAPE': 0,
                    'Directional_Accuracy': 0,
                    'Best_Loss': 0,
                    'Status': f'Failed: {str(e)[:50]}'
                })

    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_path / f'crypto_training_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nResults saved to: {results_file}")
    print(f"\nSummary:")
    print(results_df.to_string(index=False))

    successful = len(results_df[results_df['Status'] == 'Trained'])
    print(f"\nSuccessfully trained: {successful}/{len(results_df)} models")

    return results_df


if __name__ == '__main__':
    print("=" * 60)
    print("CRYPTOCURRENCY PIPELINE")
    print("=" * 60)

    # Step 1: Collect data
    print("\n[STEP 1/3] Collecting crypto data from Yahoo Finance...")
    raw_df = collect_crypto_data(start_date='2022-01-01')

    if raw_df is None:
        print("Failed to collect data. Exiting.")
        sys.exit(1)

    # Step 2: Process technical + sentiment
    print("\n[STEP 2/3] Processing technical indicators + sentiment...")
    hybrid_df = process_crypto_data(raw_df)

    if hybrid_df is None:
        print("Failed to process data. Exiting.")
        sys.exit(1)

    # Step 3: Train models
    print("\n[STEP 3/3] Training models...")
    results = train_all_crypto_models(hybrid_df)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
