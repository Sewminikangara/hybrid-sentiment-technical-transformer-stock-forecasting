"""
ARIMA Baseline Model for Stock Price Prediction
Traditional time series forecasting approach
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ARIMAPredictor:
    """
    ARIMA model for stock price prediction (baseline)
    """
    
    def __init__(self, order=(5, 1, 0)):
        """
        Args:
            order: ARIMA order (p, d, q)
                p: autoregressive order
                d: differencing order
                q: moving average order
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def fit(self, train_data: np.ndarray):
        """
        Fit ARIMA model on training data
        
        Args:
            train_data: Training time series
        """
        self.model = ARIMA(train_data, order=self.order)
        self.fitted_model = self.model.fit()
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }


def train_arima_baseline(train_data: np.ndarray, test_data: np.ndarray, 
                        order=(5, 1, 0)) -> dict:
    """
    Train and evaluate ARIMA baseline
    
    Args:
        train_data: Training prices
        test_data: Test prices
        order: ARIMA order
        
    Returns:
        Results dictionary
    """
    print(f"Training ARIMA{order} model...")
    
    # Initialize and train
    model = ARIMAPredictor(order=order)
    model.fit(train_data)
    
    # Make predictions
    predictions = model.predict(steps=len(test_data))
    
    # Evaluate
    metrics = model.evaluate(test_data, predictions)
    
    print("\nARIMA Baseline Results:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
    
    return {
        'model': model,
        'predictions': predictions,
        'metrics': metrics
    }


if __name__ == "__main__":
    # Example usage
    from scripts.data_preprocessing import StockDataPreprocessor
    
    # Load and prepare data
    preprocessor = StockDataPreprocessor(sequence_length=60)
    data_path = 'data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv'
    
    df = preprocessor.load_data(data_path)
    df_aapl = df[df['Ticker'] == 'AAPL'].copy()
    
    # Get closing prices
    prices = df_aapl['Close'].values
    
    # Split data
    train_size = int(len(prices) * 0.8)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    
    # Train ARIMA
    results = train_arima_baseline(train_prices, test_prices, order=(5, 1, 0))
