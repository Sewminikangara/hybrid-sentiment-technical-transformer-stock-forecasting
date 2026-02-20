"""
Data Loader for Stock Price Prediction App
Loads historical stock data and training results
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Load stock data and results"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / 'data_processed' / 'hybrid'
        self.raw_data_path = self.base_path / 'data_raw' / 'stock_prices'
        self.results_path = self.base_path / 'results'
        
    def load_stock_data(self, stock='AAPL', is_forex=False, is_crypto=False):
        """Load hybrid data with real prices for a stock, forex pair, or crypto pair"""
        try:
            if is_crypto:
                crypto_files = list(self.data_path.glob('crypto_hybrid_data_*.csv'))
                if not crypto_files:
                    return None

                latest_crypto = max(crypto_files, key=lambda p: p.stat().st_mtime)
                df_crypto = pd.read_csv(latest_crypto)

                crypto_data = df_crypto[df_crypto['Stock'] == stock].copy()
                crypto_data = crypto_data.sort_values('Date').reset_index(drop=True)

                if len(crypto_data) > 0:
                    return crypto_data

                return None

            if is_forex:
                forex_files = list(self.data_path.glob('forex_hybrid_data_*.csv'))
                if not forex_files:
                    return None
                
                latest_forex = max(forex_files, key=lambda p: p.stat().st_mtime)
                df_forex = pd.read_csv(latest_forex)
                
                forex_data = df_forex[df_forex['Stock'] == stock].copy()
                forex_data = forex_data.sort_values('Date').reset_index(drop=True)
                
                if len(forex_data) > 0:
                    return forex_data
                    
                return None
            
            # Load processed hybrid data (normalized features) for stocks
            hybrid_files = list(self.data_path.glob('hybrid_data_all_stocks_*.csv'))
            
            if not hybrid_files:
                return None
            
            latest_hybrid = max(hybrid_files, key=lambda p: p.stat().st_mtime)
            df_hybrid = pd.read_csv(latest_hybrid)
            
            # Filter by stock
            stock_data = df_hybrid[df_hybrid['Stock'] == stock].copy()
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            
            # Load raw prices to get actual Close prices
            raw_files = list(self.raw_data_path.glob('all_stocks_with_cse_*.csv'))
            if raw_files:
                latest_raw = max(raw_files, key=lambda p: p.stat().st_mtime)
                df_raw = pd.read_csv(latest_raw)
                
                # Filter by stock (use Ticker column)
                raw_stock = df_raw[df_raw['Ticker'] == stock][['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].copy()
                
                if len(raw_stock) > 0:
                    # Use index-based replacement (assuming same order after sorting)
                    raw_stock = raw_stock.sort_values('Date').reset_index(drop=True)
                    
                    # Take the minimum length to avoid index errors
                    min_len = min(len(stock_data), len(raw_stock))
                    
                    # Replace Close price with real price
                    stock_data = stock_data.iloc[:min_len].copy()
                    stock_data['Close'] = raw_stock['Close'].iloc[:min_len].values
                    stock_data['Open'] = raw_stock['Open'].iloc[:min_len].values
                    stock_data['High'] = raw_stock['High'].iloc[:min_len].values
                    stock_data['Low'] = raw_stock['Low'].iloc[:min_len].values
                    stock_data['Volume'] = raw_stock['Volume'].iloc[:min_len].values
                
            return stock_data
            
        except Exception as e:
            return None
    
    def load_training_results(self):
        """Load latest training results"""
        try:
            results_files = list(self.results_path.glob('hybrid_training_results_*.csv'))
            
            if not results_files:
                return None
            
            latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
            results = pd.read_csv(latest_file)
            
            return results
            
        except Exception as e:
            return None
    
    def get_stock_info(self, stock):
        """Get stock information"""
        stock_info = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'market': 'NASDAQ'},
            'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'market': 'NASDAQ'},
            'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive', 'market': 'NASDAQ'},
            'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'E-commerce', 'market': 'NASDAQ'},
            'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Technology', 'market': 'NASDAQ'},
            'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Conglomerate', 'market': 'NSE'},
            'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT Services', 'market': 'NSE'},
            'INFY.NS': {'name': 'Infosys Ltd.', 'sector': 'IT Services', 'market': 'NSE'},
            'CSEALL': {'name': 'CSE All Share Index', 'sector': 'Index', 'market': 'CSE'},
            'BTCUSD': {'name': 'Bitcoin', 'sector': 'Cryptocurrency', 'market': 'Crypto'},
            'ETHUSD': {'name': 'Ethereum', 'sector': 'Cryptocurrency', 'market': 'Crypto'},
            'BNBUSD': {'name': 'Binance Coin', 'sector': 'Cryptocurrency', 'market': 'Crypto'},
            'SOLUSD': {'name': 'Solana', 'sector': 'Cryptocurrency', 'market': 'Crypto'},
            'XRPUSD': {'name': 'XRP', 'sector': 'Cryptocurrency', 'market': 'Crypto'},
            'ADAUSD': {'name': 'Cardano', 'sector': 'Cryptocurrency', 'market': 'Crypto'},
        }
        
        return stock_info.get(stock, {'name': stock, 'sector': 'Unknown', 'market': 'Unknown'})
    
    def get_latest_price(self, stock):
        """Get latest price for a stock"""
        data = self.load_stock_data(stock)
        
        if data is None or len(data) == 0:
            return None
        
        return {
            'price': data['Close'].iloc[-1],
            'date': data['Date'].iloc[-1],
            'change': data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        }
    
    def get_available_stocks(self):
        """Get list of available stocks"""
        try:
            hybrid_files = list(self.data_path.glob('hybrid_data_all_stocks_*.csv'))
            
            if not hybrid_files:
                return []
            
            latest_file = max(hybrid_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            return df['Stock'].unique().tolist()
            
        except:
            return ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'MSFT', 
                    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'CSEALL']
