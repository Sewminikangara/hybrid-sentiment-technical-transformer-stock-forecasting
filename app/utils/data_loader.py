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
        self.results_path = self.base_path / 'results'
        
    def load_stock_data(self, stock='AAPL'):
        """Load processed hybrid data for a stock"""
        try:
            # Find latest hybrid data file
            hybrid_files = list(self.data_path.glob('hybrid_data_all_stocks_*.csv'))
            
            if not hybrid_files:
                return None
            
            latest_file = max(hybrid_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Filter by stock
            stock_data = df[df['Stock'] == stock].copy()
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            
            return stock_data
            
        except Exception as e:
            print(f"Error loading stock data: {e}")
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
            print(f"Error loading results: {e}")
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
            'CSEALL': {'name': 'CSE All Share Index', 'sector': 'Index', 'market': 'CSE'}
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
