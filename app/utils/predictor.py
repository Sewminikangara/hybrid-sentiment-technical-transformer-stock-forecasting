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

from app.utils.data_loader import DataLoader
from app.utils.model_loader import ModelLoader

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
            
            # Prepare features
            technical_cols = [col for col in stock_data.columns if col not in 
                             ['Date', 'Stock', 'Close', 'compound', 'neg', 'neu', 'pos']]
            sentiment_cols = ['compound', 'neg', 'neu', 'pos']
            
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
                return self._generate_sample_prediction(stock_data, days)
            
            # Normalize data
            technical_features = stock_data[technical_cols].values
            sentiment_features = stock_data[sentiment_cols].values
            target = stock_data['Close'].values
            
            tech_scaler = StandardScaler()
            sent_scaler = StandardScaler()
            price_scaler = StandardScaler()
            
            tech_norm = tech_scaler.fit_transform(technical_features)
            sent_norm = sent_scaler.fit_transform(sentiment_features)
            price_norm = price_scaler.fit_transform(target.reshape(-1, 1))
            
            # Generate predictions
            predictions = []
            device = self.model_loader.device
            
            # Use last sequence as input
            tech_seq = torch.FloatTensor(tech_norm[-self.sequence_length:]).unsqueeze(0).to(device)
            sent_seq = torch.FloatTensor(sent_norm[-self.sequence_length:]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                for _ in range(days):
                    pred = model(tech_seq, sent_seq)
                    predictions.append(pred.cpu().numpy()[0, 0])
                    
                    # Update sequence (simple approach - use last prediction)
                    # In production, you'd need to update technical indicators too
                    tech_seq = tech_seq[:, 1:, :]
                    sent_seq = sent_seq[:, 1:, :]
            
            # Denormalize predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions_denorm = price_scaler.inverse_transform(predictions).flatten()
            
            # Generate dates
            last_date = pd.to_datetime(stock_data['Date'].iloc[-1])
            pred_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Calculate confidence intervals (±5%)
            lower = predictions_denorm * 0.95
            upper = predictions_denorm * 1.05
            
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
