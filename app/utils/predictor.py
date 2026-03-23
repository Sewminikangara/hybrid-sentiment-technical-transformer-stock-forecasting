import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from pathlib import Path

from utils.data_loader import DataLoader
from utils.model_loader import ModelLoader


class StockPredictor:

    def __init__(self, stock, model_name, is_forex=False, is_crypto=False):
        self.stock = stock
        self.model_name = model_name
        self.is_forex = is_forex
        self.is_crypto = is_crypto
        self.data_loader = DataLoader()
        self.model_loader = ModelLoader()
        self.sequence_length = 60

    def predict(self, days=7):
        try:
            stock_data = self.data_loader.load_stock_data(
                self.stock, is_forex=self.is_forex, is_crypto=self.is_crypto
            )

            if stock_data is None or len(stock_data) < 20:
                return None

            self.sequence_length = min(self.sequence_length, len(stock_data))

            if self.is_forex or self.is_crypto:
                sentiment_cols = ['sentiment_score', 'positive', 'negative', 'neutral',
                                  'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility']
                exclude = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume'] + sentiment_cols
            else:
                sentiment_cols = ['sentiment_score', 'sentiment_positive', 'sentiment_negative',
                                  'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7',
                                  'sentiment_volatility']
                exclude = ['Date', 'Stock', 'stock', 'date', 'source', 'Close', 'Ticker',
                           'sentiment_score', 'sentiment_label', 'confidence'] + sentiment_cols

            actual_sentiment_cols = [c for c in sentiment_cols if c in stock_data.columns]
            technical_cols = [c for c in stock_data.columns if c not in exclude]

            technical_dim = len(technical_cols)
            sentiment_dim = len(actual_sentiment_cols)

            model = self.model_loader.load_model(
                self.stock, self.model_name,
                technical_dim=technical_dim,
                sentiment_dim=sentiment_dim
            )

            if model is None:
                return None

            real_close_prices = stock_data['Close'].values
            last_real_price   = float(real_close_prices[-1])

            technical_features = stock_data[technical_cols].values
            sentiment_features = stock_data[actual_sentiment_cols].values

            device   = self.model_loader.device
            tech_seq = torch.FloatTensor(technical_features[-self.sequence_length:]).unsqueeze(0).to(device)
            sent_seq = torch.FloatTensor(sentiment_features[-self.sequence_length:]).unsqueeze(0).to(device)

            from models.baseline_lstm import LSTMModel
            is_lstm = isinstance(model, LSTMModel)

            predictions_raw = []
            with torch.no_grad():
                for _ in range(days):
                    if is_lstm:
                        if model.lstm.input_size == technical_dim:
                            pred = model(tech_seq)
                        else:
                            pred = model(torch.cat([tech_seq, sent_seq], dim=-1))
                    else:
                        pred = model(tech_seq, sent_seq)

                    raw_val = float(pred.cpu().numpy()[0, 0])
                    predictions_raw.append(raw_val)

                    new_tech = tech_seq[:, -1:, :].clone()
                    new_tech[0, 0, 0] = raw_val
                    tech_seq = torch.cat([tech_seq[:, 1:, :], new_tech], dim=1)

                    new_sent = sent_seq[:, -1:, :].clone()
                    sent_seq = torch.cat([sent_seq[:, 1:, :], new_sent], dim=1)

            predictions_raw = np.array(predictions_raw)
            direction_bias  = float(np.mean(predictions_raw))

            recent_prices = real_close_prices[-90:] if len(real_close_prices) >= 90 else real_close_prices
            hist_returns  = np.diff(recent_prices) / recent_prices[:-1]
            hist_mean_ret = float(np.mean(hist_returns)) if len(hist_returns) > 0 else 0.0
            hist_std_ret  = float(np.std(hist_returns))  if len(hist_returns) > 0 else 0.01

            daily_bias = np.tanh(direction_bias) * 0.005

            np.random.seed(42)
            predictions_denorm = np.zeros(days)
            prev = last_real_price
            for i in range(days):
                daily_return = hist_mean_ret + daily_bias + np.random.normal(0, hist_std_ret * 0.4)
                prev = prev * (1 + daily_return)
                predictions_denorm[i] = prev

            predictions_denorm = np.clip(
                predictions_denorm,
                last_real_price * 0.85,
                last_real_price * 1.15
            )

            last_date  = pd.to_datetime(stock_data['Date'].iloc[-1])
            pred_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

            lower = np.array([predictions_denorm[i] * (1 - hist_std_ret * (i + 1)) for i in range(days)])
            upper = np.array([predictions_denorm[i] * (1 + hist_std_ret * (i + 1)) for i in range(days)])

            return {
                'dates':  pred_dates,
                'prices': predictions_denorm.tolist(),
                'lower':  lower.tolist(),
                'upper':  upper.tolist()
            }

        except Exception:
            return None
