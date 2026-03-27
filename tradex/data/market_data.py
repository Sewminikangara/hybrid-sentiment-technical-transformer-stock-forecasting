"""
TradeXY - Market Data Provider

"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger("tradex.market_data")

# Project root for loading cached data
PROJECT_ROOT = Path(__file__).parent.parent.parent

class MarketDataProvider:
    """
    Unified market data provider for stocks, forex, and crypto.

    Priority:
        1. Live API data (yfinance, exchange APIs)
        2. Cached/local CSV data from the research project
        3. Mock data (for testing)

    Usage:
        provider = MarketDataProvider()
        df_4h = provider.get_ohlcv("BTCUSDT", timeframe="4h", bars=300)
        df_15m = provider.get_ohlcv("BTCUSDT", timeframe="15m", bars=500)
    """

    # Symbol type mappings
    CRYPTO_SYMBOLS = {"BTCUSDT", "ETHUSDT", "BNBUSD", "SOLUSD", "XRPUSD", "ADAUSD"}
    FOREX_SYMBOLS = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"}

    # yfinance ticker mappings
    YFINANCE_MAP = {
        "AAPL": "AAPL", "GOOGL": "GOOGL", "TSLA": "TSLA",
        "AMZN": "AMZN", "MSFT": "MSFT",
        "RELIANCE.NS": "RELIANCE.NS", "TCS.NS": "TCS.NS",
        "INFY.NS": "INFY.NS", "CSEALL": "CSEALL.CM",
        "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD",
        "BNBUSD": "BNB-USD", "SOLUSD": "SOL-USD",
        "XRPUSD": "XRP-USD", "ADAUSD": "ADA-USD",
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X",
        "USDCAD": "CAD=X", "USDCHF": "CHF=X",
    }

    # Timeframe to yfinance interval mapping
    TF_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "60m", "4h": "60m",  # yfinance doesn't have 4h - we'll resample
        "1d": "1d", "1w": "1wk",
    }

    # How much history to request for each timeframe
    TF_PERIOD_MAP = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "730d", "4h": "730d", "1d": "2y", "1w": "5y",
    }

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_ohlcv(self, symbol: str, timeframe: str = "4h",
                  bars: int = 300) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol at a specific timeframe.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
            timeframe: "1m", "5m", "15m", "30m", "1h", "4h", "1d"
            bars: Number of bars to return

        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
            or None if data unavailable
        """
        cache_key = f"{symbol}_{timeframe}"

        # Check cache
        if self.use_cache and cache_key in self._cache:
            df = self._cache[cache_key]
            return df.tail(bars).copy()

        # Try live data first
        df = self._fetch_live(symbol, timeframe)

        # Fallback to local CSV
        if df is None or df.empty:
            df = self._load_local(symbol, timeframe)

        # Last resort: generate mock data
        if df is None or df.empty:
            logger.warning(f"No data available for {symbol}/{timeframe}. Using mock data.")
            df = self._generate_mock(symbol, bars)

        if df is not None and not df.empty:
            self._cache[cache_key] = df
            return df.tail(bars).copy()

        return None

    def get_multi_timeframe(self, symbol: str,
                            trend_tf: str = "4h",
                            entry_tf: str = "15m",
                            trend_bars: int = 300,
                            entry_bars: int = 500
                            ) -> Dict[str, pd.DataFrame]:
        """
        Fetch both trend and entry timeframe data for a symbol.

        Returns:
            {"trend": df_trend, "entry": df_entry}
        """
        trend_df = self.get_ohlcv(symbol, trend_tf, trend_bars)
        entry_df = self.get_ohlcv(symbol, entry_tf, entry_bars)

        result = {}
        if trend_df is not None:
            result["trend"] = trend_df
        if entry_df is not None:
            result["entry"] = entry_df

        return result

    def _fetch_live(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch live data using yfinance."""
        try:
            import yfinance as yf

            ticker = self.YFINANCE_MAP.get(symbol, symbol)
            interval = self.TF_MAP.get(timeframe, "1d")
            period = self.TF_PERIOD_MAP.get(timeframe, "1y")

            logger.info(f"Fetching live data: {ticker} ({interval}, {period})")

            data = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)

            if data.empty:
                return None

            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure standard column names
            df = data.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })

            # Resample if needed (e.g., 4h from 1h data)
            if timeframe == "4h" and interval == "60m":
                df = self._resample_ohlcv(df, "4h")

            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        except ImportError:
            logger.warning("yfinance not installed. Falling back to local data.")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch live data for {symbol}: {e}")
            return None

    def _load_local(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from local CSV files (research project data)."""
        # Map to local file names
        local_map = {
            "BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD",
        }
        local_symbol = local_map.get(symbol, symbol)

        # Search for data files
        data_dirs = [
            PROJECT_ROOT / "data_processed" / "technical",
            PROJECT_ROOT / "data_processed",
            PROJECT_ROOT / "data_raw",
        ]

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue

            # Try various naming patterns
            patterns = [
                f"{local_symbol}_technical_indicators.csv",
                f"{local_symbol}_daily.csv",
                f"{local_symbol}.csv",
                f"{local_symbol}_data.csv",
            ]

            for pattern in patterns:
                filepath = data_dir / pattern
                if filepath.exists():
                    try:
                        df = pd.read_csv(filepath)

                        # Find date column
                        date_col = None
                        for col in ['Date', 'date', 'Datetime', 'datetime', 'timestamp']:
                            if col in df.columns:
                                date_col = col
                                break

                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col])
                            df.set_index(date_col, inplace=True)

                        # Ensure required columns
                        required = ['Open', 'High', 'Low', 'Close']
                        if all(col in df.columns for col in required):
                            if 'Volume' not in df.columns:
                                df['Volume'] = 0

                            logger.info(f"Loaded local data: {filepath.name} ({len(df)} rows)")
                            return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

                    except Exception as e:
                        logger.error(f"Failed to load {filepath}: {e}")

        return None

    def _resample_ohlcv(self, df: pd.DataFrame,
                        timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to a larger timeframe."""
        tf_map = {"4h": "4h", "1d": "1D", "1w": "1W"}
        rule = tf_map.get(timeframe, timeframe)

        resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return resampled

    def _generate_mock(self, symbol: str, bars: int) -> pd.DataFrame:
        """Generate realistic mock OHLCV data for testing."""
        np.random.seed(hash(symbol) % 2**32)

        # Base prices by asset type
        base_prices = {
            "BTCUSDT": 65000, "ETHUSDT": 3500, "BNBUSD": 300,
            "SOLUSD": 150, "XRPUSD": 0.6, "ADAUSD": 0.5,
            "EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150,
            "AUDUSD": 0.65, "USDCAD": 1.36, "USDCHF": 0.88,
            "AAPL": 180, "GOOGL": 140, "TSLA": 250,
            "AMZN": 175, "MSFT": 400,
            "RELIANCE.NS": 2500, "TCS.NS": 3800,
            "INFY.NS": 1500, "CSEALL": 12000,
        }

        base_price = base_prices.get(symbol, 100)
        volatility = base_price * 0.015  # 1.5% daily vol

        t = np.arange(bars)
        trend = base_price + (base_price * 0.0002) * t  # Slight uptrend
        cycles = volatility * 3 * np.sin(2 * np.pi * t / 60)
        noise = np.random.normal(0, volatility, bars)

        close = trend + cycles + noise
        high = close + np.abs(np.random.normal(volatility * 0.5, volatility * 0.2, bars))
        low = close - np.abs(np.random.normal(volatility * 0.5, volatility * 0.2, bars))
        open_price = close + np.random.normal(0, volatility * 0.3, bars)

        dates = pd.date_range(end=datetime.utcnow(), periods=bars, freq='4h')

        return pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(int(1e5), int(1e7), bars)
        }, index=dates)

    def get_available_symbols(self) -> Dict[str, List[str]]:
        """List all available symbols by category."""
        return {
            "stocks": ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT",
                       "RELIANCE.NS", "TCS.NS", "INFY.NS", "CSEALL"],
            "crypto": list(self.CRYPTO_SYMBOLS),
            "forex": list(self.FOREX_SYMBOLS),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("""
        TradeXY - Market Data Provider
        Live + Local + Mock Data Sources
    """)

    provider = MarketDataProvider()

    # Test with a few symbols
    for symbol in ["AAPL", "BTCUSDT", "EURUSD"]:
        data = provider.get_multi_timeframe(symbol)

        for tf_name, df in data.items():
            print(f"  {symbol} ({tf_name}): {len(df)} bars, "
                  f"range: {df['Close'].min():.2f} – {df['Close'].max():.2f}")

    logger.info("\n  Available symbols: {provider.get_available_symbols()}")
    logger.info("\nMarket Data Provider test complete.")
