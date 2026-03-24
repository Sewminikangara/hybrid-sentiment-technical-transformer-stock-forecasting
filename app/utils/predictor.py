import numpy as np
import pandas as pd
import torch
from datetime import timedelta

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
        """Generate an N-day price forecast for the configured asset and model."""
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
            technical_cols        = [c for c in stock_data.columns if c not in exclude]
            technical_dim         = len(technical_cols)
            sentiment_dim         = len(actual_sentiment_cols)

            # Retrieve training-time price scaler from checkpoint if available
            model_key  = self.model_loader.model_map.get(self.model_name, 'early_fusion')
            model_path = self.model_loader.results_path / f'{self.stock}_{model_key}.pt'
            if not model_path.exists():
                model_path = self.model_loader.best_models_path / f'best_{model_key}_transformer.pt'

            price_mean = None
            price_std  = None
            if model_path.exists():
                try:
                    ck = torch.load(model_path, map_location='cpu', weights_only=False)
                    if isinstance(ck, dict) and 'stats' in ck:
                        price_mean = ck['stats'].get('price_mean')
                        price_std  = ck['stats'].get('price_std')
                except Exception:
                    pass

            model = self.model_loader.load_model(
                self.stock, self.model_name,
                technical_dim=technical_dim,
                sentiment_dim=sentiment_dim
            )
            if model is None:
                return None

            real_close_prices  = stock_data['Close'].values
            last_real_price    = float(real_close_prices[-1])
            technical_features = stock_data[technical_cols].values
            sentiment_features = stock_data[actual_sentiment_cols].values

            device   = self.model_loader.device
            tech_seq = torch.FloatTensor(technical_features[-self.sequence_length:]).unsqueeze(0).to(device)
            sent_seq = torch.FloatTensor(sentiment_features[-self.sequence_length:]).unsqueeze(0).to(device)

            from models.baseline_lstm import LSTMModel
            is_lstm = isinstance(model, LSTMModel)

            raw_outputs = []
            with torch.no_grad():
                for _ in range(days):
                    if is_lstm:
                        pred = (model(tech_seq) if model.lstm.input_size == technical_dim
                                else model(torch.cat([tech_seq, sent_seq], dim=-1)))
                    else:
                        pred = model(tech_seq, sent_seq)

                    raw_val = float(pred.cpu().numpy()[0, 0])
                    raw_outputs.append(raw_val)

                    new_tech = tech_seq[:, -1:, :].clone()
                    new_tech[0, 0, 0] = raw_val
                    tech_seq = torch.cat([tech_seq[:, 1:, :], new_tech], dim=1)
                    new_sent = sent_seq[:, -1:, :].clone()
                    sent_seq = torch.cat([sent_seq[:, 1:, :], new_sent], dim=1)

            raw_outputs = np.array(raw_outputs)

            # Denormalise: use checkpoint scaler when available
            if price_mean is not None and price_std is not None and price_std > 0:
                predicted_level = float(np.mean(raw_outputs)) * price_std + price_mean
            else:
                window          = real_close_prices[-90:] if len(real_close_prices) >= 90 else real_close_prices
                p_mean          = float(np.mean(window))
                p_std           = float(np.std(window)) if np.std(window) > 0 else p_mean * 0.05
                predicted_level = float(np.mean(raw_outputs)) * p_std + p_mean

            # Derive directional bias: positive model output → bullish, negative → bearish
            mean_raw     = float(np.mean(raw_outputs))
            direction    = np.sign(mean_raw) if abs(mean_raw) > 1e-6 else 1.0
            bias_strength = min(abs(mean_raw) / 10.0, 0.005)  # cap at ±0.5 % per day

            # Historical daily return stats (used for per-day step size)
            recent       = real_close_prices[-90:] if len(real_close_prices) >= 90 else real_close_prices
            hist_returns = np.diff(recent) / recent[:-1] if len(recent) > 1 else np.array([0.01])
            hist_mean    = float(np.mean(hist_returns))
            hist_std     = float(np.std(hist_returns)) if np.std(hist_returns) > 0 else 0.01

            # Compound forecast: each day applies (historical mean + model directional bias)
            predictions_denorm = np.zeros(days)
            prev = last_real_price
            for i in range(days):
                daily_return       = hist_mean + direction * bias_strength
                prev               = prev * (1 + daily_return)
                predictions_denorm[i] = prev

            predictions_denorm = np.clip(
                predictions_denorm,
                last_real_price * 0.85,
                last_real_price * 1.15
            )

            last_date  = pd.to_datetime(stock_data['Date'].iloc[-1])
            pred_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

            lower = np.array([predictions_denorm[i] * (1 - hist_std * (i + 1)) for i in range(days)])
            upper = np.array([predictions_denorm[i] * (1 + hist_std * (i + 1)) for i in range(days)])

            return {
                'dates':  pred_dates,
                'prices': predictions_denorm.tolist(),
                'lower':  lower.tolist(),
                'upper':  upper.tolist()
            }

        except Exception:
            return None
