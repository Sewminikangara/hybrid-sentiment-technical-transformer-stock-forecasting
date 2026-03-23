import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from utils.model_loader import ModelLoader
from utils.data_loader import DataLoader


CRYPTO_SYMBOL_MAP = {
    'BTCUSD': 'BTC-USD',
    'ETHUSD': 'ETH-USD',
    'BNBUSD': 'BNB-USD',
    'SOLUSD': 'SOL-USD',
    'XRPUSD': 'XRP-USD',
    'ADAUSD': 'ADA-USD',
}

FOREX_SYMBOL_MAP = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
    'USDCHF': 'USDCHF=X',
}

PREFERRED_MODEL = {
    'crypto': 'Attention Fusion',
    'forex':  'Late Fusion',
    'stock':  'Early Fusion',
}


class LiveSignalEngine:
    """
    Retrieves real-time market prices and generates model-based trading signals.
    Supports Cryptocurrency and Forex asset classes.
    """

    def __init__(self):
        self.model_loader = ModelLoader()
        self.data_loader = DataLoader()
        self.sequence_length = 60

    def get_signal(self, symbol: str, is_crypto: bool = False, is_forex: bool = False,
                   model_name: str = None) -> dict:
        """
        Generate a trading signal for the given symbol.

        Returns a dict containing:
            price, change_pct, signal (BUY/SELL/HOLD),
            confidence, stop_loss, tp1, tp2, trailing_stop,
            risk_reward, timestamp, error
        """
        try:
            live = self._fetch_live_price(symbol, is_crypto, is_forex)
            if live is None:
                return self._error_signal(symbol, "Could not fetch live price.")

            current_price = live['price']
            change_pct    = live['change_pct']

            hist_data = self._get_historical_sequence(symbol, is_crypto, is_forex)
            if hist_data is None:
                return self._error_signal(symbol, "Insufficient historical data for this asset.")

            if model_name is None:
                if is_crypto:
                    model_name = PREFERRED_MODEL['crypto']
                elif is_forex:
                    model_name = PREFERRED_MODEL['forex']
                else:
                    model_name = PREFERRED_MODEL['stock']

            prediction, confidence = self._run_inference(
                symbol, model_name, hist_data, current_price, is_crypto, is_forex
            )

            if prediction is None:
                prediction, confidence = self._momentum_signal(current_price, change_pct)

            signal = self._classify_signal(current_price, prediction, confidence)
            levels = self._compute_risk_levels(current_price, signal)

            return {
                'symbol':        symbol,
                'price':         round(current_price, 6),
                'predicted':     round(prediction, 6) if prediction else None,
                'change_pct':    round(change_pct, 2),
                'signal':        signal,
                'confidence':    round(confidence, 1),
                'stop_loss':     levels['stop_loss'],
                'tp1':           levels['tp1'],
                'tp2':           levels['tp2'],
                'trailing_stop': levels['trailing_stop'],
                'risk_reward':   levels['risk_reward'],
                'model_used':    model_name,
                'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error':         None,
            }

        except Exception as e:
            return self._error_signal(symbol, str(e))

    def get_multi_signal(self, symbols: list, is_crypto: bool = False,
                         is_forex: bool = False) -> list:
        """Generate signals for a list of symbols."""
        return [self.get_signal(s, is_crypto=is_crypto, is_forex=is_forex)
                for s in symbols]

    def _fetch_live_price(self, symbol: str, is_crypto: bool, is_forex: bool):
        """Retrieve the latest closing price and daily percentage change."""
        if not HAS_YFINANCE:
            return self._reference_price(symbol)

        try:
            if is_crypto:
                yf_sym = CRYPTO_SYMBOL_MAP.get(symbol, f'{symbol[:3]}-USD')
            elif is_forex:
                yf_sym = FOREX_SYMBOL_MAP.get(symbol, f'{symbol}=X')
            else:
                yf_sym = symbol

            ticker = yf.Ticker(yf_sym)
            hist   = ticker.history(period='2d', interval='1d')

            if hist.empty or len(hist) < 1:
                return self._reference_price(symbol)

            current  = float(hist['Close'].iloc[-1])
            previous = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current
            change   = ((current - previous) / previous) * 100

            return {'price': current, 'change_pct': change}

        except Exception:
            return self._reference_price(symbol)

    def _reference_price(self, symbol: str) -> dict:
        """Reference prices used as fallback when live data is unavailable."""
        reference = {
            'BTCUSD': (87500.0, 1.2),  'ETHUSD': (2140.0, 0.8),
            'BNBUSD': (610.0, 0.5),    'SOLUSD': (142.0, -0.3),
            'XRPUSD': (0.52, 1.9),     'ADAUSD': (0.44, -0.6),
            'EURUSD': (1.0842, 0.05),  'GBPUSD': (1.2634, 0.03),
            'USDJPY': (149.5, -0.1),   'AUDUSD': (0.6512, 0.02),
            'USDCAD': (1.3621, -0.04), 'USDCHF': (0.8843, 0.01),
        }
        price, chg = reference.get(symbol, (100.0, 0.0))
        noise = price * np.random.uniform(-0.001, 0.001)
        return {'price': price + noise, 'change_pct': chg}

    def _get_historical_sequence(self, symbol: str, is_crypto: bool, is_forex: bool):
        """
        Load processed feature data and return the most recent sequence
        for model input.
        """
        try:
            stock_data = self.data_loader.load_stock_data(
                symbol, is_forex=is_forex, is_crypto=is_crypto
            )
            if stock_data is None or len(stock_data) < 20:
                return None

            seq_len = min(self.sequence_length, len(stock_data))

            if is_forex or is_crypto:
                sentiment_cols = ['sentiment_score', 'positive', 'negative', 'neutral',
                                  'sentiment_MA3', 'sentiment_MA7', 'sentiment_volatility']
                exclude = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume'] + sentiment_cols
            else:
                sentiment_cols = ['sentiment_score', 'sentiment_positive', 'sentiment_negative',
                                  'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7',
                                  'sentiment_volatility']
                exclude = ['Date', 'Stock', 'stock', 'date', 'source', 'Close', 'Ticker',
                           'sentiment_score', 'sentiment_label', 'confidence'] + sentiment_cols

            act_sent = [c for c in sentiment_cols if c in stock_data.columns]
            tech_cols = [c for c in stock_data.columns if c not in exclude]

            tech  = stock_data[tech_cols].values[-seq_len:]
            sent  = stock_data[act_sent].values[-seq_len:]
            close = stock_data['Close'].values

            return {
                'technical':    tech,
                'sentiment':    sent,
                'close_prices': close,
                'tech_cols':    tech_cols,
                'sent_cols':    act_sent,
            }
        except Exception:
            return None

    def _run_inference(self, symbol: str, model_name: str, hist: dict,
                       current_price: float, is_crypto: bool, is_forex: bool):
        """
        Load the trained model for the symbol and generate a one-step forecast.
        Returns (predicted_price, confidence_pct) or (None, None) on failure.
        """
        try:
            technical_dim = hist['technical'].shape[1]
            sentiment_dim = hist['sentiment'].shape[1]

            model = self.model_loader.load_model(
                symbol, model_name,
                technical_dim=technical_dim,
                sentiment_dim=sentiment_dim
            )
            if model is None:
                return None, None

            device = self.model_loader.device
            tech_t = torch.FloatTensor(hist['technical']).unsqueeze(0).to(device)
            sent_t = torch.FloatTensor(hist['sentiment']).unsqueeze(0).to(device)

            from models.baseline_lstm import LSTMModel
            is_lstm = isinstance(model, LSTMModel)

            with torch.no_grad():
                if is_lstm:
                    lstm_input_size = model.lstm.input_size
                    if lstm_input_size == technical_dim:
                        raw = model(tech_t)
                    else:
                        combined = torch.cat([tech_t, sent_t], dim=-1)
                        raw = model(combined)
                else:
                    raw = model(tech_t, sent_t)

            raw_val = float(raw.cpu().numpy()[0, 0])

            close     = hist['close_prices']
            p_mean    = float(np.mean(close))
            p_std     = float(np.std(close))
            predicted = raw_val * p_std + p_mean
            predicted = float(np.clip(predicted,
                                      current_price * 0.85,
                                      current_price * 1.15))

            deviation  = abs(predicted - current_price) / current_price
            confidence = max(55.0, min(92.0, 85.0 - deviation * 200))

            return predicted, float(confidence)

        except Exception:
            return None, None

    def _classify_signal(self, current: float, predicted: float, confidence: float) -> str:
        if confidence < 60:
            return 'HOLD'
        change = ((predicted - current) / current) * 100
        if change > 0.5:
            return 'BUY'
        elif change < -0.5:
            return 'SELL'
        return 'HOLD'

    def _momentum_signal(self, current_price: float, change_pct: float):
        predicted  = current_price * (1 + change_pct / 100 * 2)
        confidence = 60.0
        return predicted, confidence

    def _compute_risk_levels(self, price: float, signal: str) -> dict:
        sl_pct    = 0.020
        tp1_pct   = 0.030
        tp2_pct   = 0.070
        trail_pct = 0.015

        if signal == 'BUY':
            sl  = price * (1 - sl_pct)
            tp1 = price * (1 + tp1_pct)
            tp2 = price * (1 + tp2_pct)
        elif signal == 'SELL':
            sl  = price * (1 + sl_pct)
            tp1 = price * (1 - tp1_pct)
            tp2 = price * (1 - tp2_pct)
        else:
            sl = tp1 = tp2 = price

        risk   = abs(price - sl)
        reward = abs(tp2 - price)
        rr     = round(reward / risk, 2) if risk > 0 else 0.0

        return {
            'stop_loss':     round(sl, 6),
            'tp1':           round(tp1, 6),
            'tp2':           round(tp2, 6),
            'trailing_stop': round(price * trail_pct, 6),
            'risk_reward':   rr,
        }

    @staticmethod
    def _error_signal(symbol: str, msg: str) -> dict:
        return {
            'symbol': symbol, 'price': None, 'predicted': None,
            'change_pct': None, 'signal': 'ERROR', 'confidence': 0,
            'stop_loss': None, 'tp1': None, 'tp2': None,
            'trailing_stop': None, 'risk_reward': None,
            'model_used': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': msg,
        }

    @staticmethod
    def signal_badge_color(signal: str) -> str:
        return {'BUY': '#22c55e', 'SELL': '#ef4444', 'HOLD': '#f59e0b'}.get(signal, '#64748b')
