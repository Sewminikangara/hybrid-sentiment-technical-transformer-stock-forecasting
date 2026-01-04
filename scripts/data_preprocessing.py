"""
Data Preprocessing Pipeline for Stock Price Prediction
Handles data loading, cleaning, feature engineering, and train/test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class StockDataPreprocessor:
    """
    Preprocessing pipeline for stock market data
    """
    
    def __init__(self, sequence_length: int = 60, train_ratio: float = 0.8):
        """
        Args:
            sequence_length: Number of time steps for sequence generation
            train_ratio: Ratio of training data to total data
        """
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare stock data"""
        df = pd.read_csv(data_path)
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variable
        
        Args:
            df: Input dataframe
            target_col: Target column name (price to predict)
            
        Returns:
            X: Feature array
            y: Target array
        """
        # Select feature columns (exclude date and target)
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'date', 'Ticker', 'ticker', 'Symbol', target_col]]
        
        # Separate features and target
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        return X, y
    
    def scale_data(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features and target
        
        Args:
            X: Feature array
            y: Target array
            fit: Whether to fit the scalers
            
        Returns:
            Scaled X and y
        """
        if fit:
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.price_scaler.fit_transform(y)
        else:
            X_scaled = self.feature_scaler.transform(X)
            y_scaled = self.price_scaler.transform(y)
            
        return X_scaled, y_scaled
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            X_seq: Sequences of features [samples, sequence_length, features]
            y_seq: Corresponding targets [samples, 1]
        """
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split_idx = int(len(X) * self.train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_stock_data(self, data_path: str, ticker: str = None) -> Dict:
        """
        Complete preprocessing pipeline
        
        Args:
            data_path: Path to data file
            ticker: Optional ticker to filter
            
        Returns:
            Dictionary with all prepared data
        """
        # Load data
        df = self.load_data(data_path)
        
        # Filter by ticker if specified
        if ticker and 'Ticker' in df.columns:
            df = df[df['Ticker'] == ticker].copy()
        elif ticker and 'ticker' in df.columns:
            df = df[df['ticker'] == ticker].copy()
            
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Scale data
        X_scaled, y_scaled = self.scale_data(X, y, fit=True)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X_seq, y_seq)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler,
            'n_features': X_train.shape[2],
            'sequence_length': self.sequence_length
        }
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Original scale predictions
        """
        return self.price_scaler.inverse_transform(predictions.reshape(-1, 1))


def prepare_multi_stock_data(data_path: str, tickers: List[str], 
                             sequence_length: int = 60) -> Dict[str, Dict]:
    """
    Prepare data for multiple stocks
    
    Args:
        data_path: Path to combined data file
        tickers: List of stock tickers
        sequence_length: Sequence length for models
        
    Returns:
        Dictionary mapping ticker to prepared data
    """
    preprocessor = StockDataPreprocessor(sequence_length=sequence_length)
    results = {}
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        results[ticker] = preprocessor.prepare_stock_data(data_path, ticker)
        
    return results


if __name__ == "__main__":
    # Example usage
    preprocessor = StockDataPreprocessor(sequence_length=60, train_ratio=0.8)
    
    # Prepare technical indicators data
    data = preprocessor.prepare_stock_data(
        'data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv',
        ticker='AAPL'
    )
    
    print(f"Training samples: {data['X_train'].shape[0]}")
    print(f"Test samples: {data['X_test'].shape[0]}")
    print(f"Sequence length: {data['sequence_length']}")
    print(f"Number of features: {data['n_features']}")
