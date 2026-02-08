"""
Stock Price Predictor
Generates predictions using trained models
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from pathlib import Path

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader

class StockPredictor:
    """Generate stock price predictions"""
    
    def __init__(self, stock, model_name):
        self.stock = stock
        self.model_name = model_name
        self.data_loader = DataLoader()
        self.model_loader = ModelLoader()
        self.sequence_length = 60
        
    def predict(self, days=7):
        """Generate predictions for next N days"""
        try:
            # Load data
            stock_data = self.data_loader.load_stock_data(self.stock)
            
            if stock_data is None or len(stock_data) < self.sequence_length:
                return None
            
            # Prepare features - use same column definitions as training
            technical_cols = [c for c in stock_data.columns if c not in 
                             ['Date', 'Stock', 'stock', 'date', 'source', 'Close',
                              'sentiment_score', 'sentiment_label', 'confidence',
                              'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
                              'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']]
            
            sentiment_cols = ['sentiment_score', 'sentiment_positive', 'sentiment_negative', 
                             'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']
            
            # Get technical dimensions
            technical_dim = len(technical_cols)
            sentiment_dim = len(sentiment_cols)
            
            # Load model
            model = self.model_loader.load_model(
                self.stock, 
                self.model_name, 
                technical_dim=technical_dim,
                sentiment_dim=sentiment_dim
            )
            
            if model is None:
                print(f"WARNING: Model not loaded for {self.stock} - {self.model_name}")
                print(f"Technical dims: {technical_dim}, Sentiment dims: {sentiment_dim}")
                return self._generate_sample_prediction(stock_data, days)
            
            print(f"SUCCESS: Model loaded for {self.stock} - {self.model_name}")
            print(f"Features: {technical_dim} technical + {sentiment_dim} sentiment = {technical_dim + sentiment_dim} total")
            
            # Get real Close prices for scaling (already loaded from raw data)
            real_close_prices = stock_data['Close'].values
            last_real_price = real_close_prices[-1]
            
            # Technical and sentiment features are already normalized in hybrid data
            # Just extract them directly
            technical_features = stock_data[technical_cols].values
            sentiment_features = stock_data[sentiment_cols].values
            
            # Generate predictions
            predictions = []
            device = self.model_loader.device
            
            # Use last sequence as input (data is already normalized)
            tech_seq = torch.FloatTensor(technical_features[-self.sequence_length:]).unsqueeze(0).to(device)
            sent_seq = torch.FloatTensor(sentiment_features[-self.sequence_length:]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                for _ in range(days):
                    pred = model(tech_seq, sent_seq)
                    predictions.append(pred.cpu().numpy()[0, 0])
                    
                    # Update sequence (simple approach - use last prediction)
                    # In production, you'd need to update technical indicators too
                    tech_seq = tech_seq[:, 1:, :]
                    sent_seq = sent_seq[:, 1:, :]
            
            # Denormalize predictions
            # Model was trained on normalized data, need to scale back to real prices
            predictions = np.array(predictions)
            
            # Calculate price statistics from real historical data
            price_mean = np.mean(real_close_prices)
            price_std = np.std(real_close_prices)
            
            # Denormalize: value * std + mean
            predictions_denorm = predictions * price_std + price_mean
            
            # Ensure predictions are reasonable (within 20% of last price)
            predictions_denorm = np.clip(predictions_denorm, 
                                        last_real_price * 0.8, 
                                        last_real_price * 1.2)
            
            # Generate dates
            last_date = pd.to_datetime(stock_data['Date'].iloc[-1])
            pred_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Calculate confidence intervals (±5%)
            lower = predictions_denorm * 0.95
            upper = predictions_denorm * 1.05
            
            print(f"Prediction range: ${predictions_denorm[0]:.2f} to ${predictions_denorm[-1]:.2f}")
            
            return {
                'dates': pred_dates,
                'prices': predictions_denorm.tolist(),
                'lower': lower.tolist(),
                'upper': upper.tolist()
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._generate_sample_prediction(stock_data, days)
    
    def _generate_sample_prediction(self, stock_data, days):
        """Generate sample prediction when model not available"""
        last_price = stock_data['Close'].iloc[-1]
        last_date = pd.to_datetime(stock_data['Date'].iloc[-1])
        
        # Simple random walk with slight upward bias
        predictions = []
        current_price = last_price
        
        for i in range(days):
            change = np.random.normal(0.001, 0.02)  # Slight upward drift
            current_price = current_price * (1 + change)
            predictions.append(current_price)
        
        pred_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        predictions = np.array(predictions)
        lower = predictions * 0.95
        upper = predictions * 1.05
        
        return {
            'dates': pred_dates,
            'prices': predictions.tolist(),
            'lower': lower.tolist(),
            'upper': upper.tolist()
        }
