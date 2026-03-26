import json
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False
    class BinanceAPIException(Exception): pass

logger = logging.getLogger(__name__)


class AutoTrader:
    """Automated trading bot using Binance Testnet"""

    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """Initialize trading bot

        Args:
            api_key: Binance API key (optional, for testnet)
            api_secret: Binance API secret (optional, for testnet)
            testnet: Use testnet (True) or live trading (False)
        """
        self.testnet = testnet
        self.trades_file = Path(__file__).parent.parent / 'trading_history.json'
        self.config_file = Path(__file__).parent.parent / 'trading_config.json'

        # Initialize Binance client (testnet or live)
        if api_key and api_secret and HAS_BINANCE:
            try:
                if testnet:
                    # Binance Testnet
                    self.client = Client(api_key, api_secret, testnet=True)
                    self.client.API_URL = 'https://testnet.binance.vision/api'
                else:
                    # Live Binance (use with caution!)
                    self.client = Client(api_key, api_secret)
                self.connected = True
            except Exception as e:
                logger.error("Failed to connect to Binance: %s", e)
                self.client = None
                self.connected = False
        else:
            self.client = None
            self.connected = False
            if api_key and api_secret and not HAS_BINANCE:
                logger.warning("API keys provided but python-binance is not installed. Using paper trading mode.")

        # Load or initialize trading history
        self.load_trading_history()

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load trading configuration"""
        default_config = {
            'risk_per_trade': 0.02,  # 2% of portfolio per trade
            'max_trades_per_day': 10,
            'stop_loss_pct': 0.02,  # 2% initial stop loss
            'trailing_stop_pct': 0.015, # 1.5% trailing stop
            'tp_target1_pct': 0.03,  # 3% take profit (close 50%)
            'tp_target2_pct': 0.07,  # 7% take profit (close 50%)
            'daily_loss_limit_pct': 0.05, # Stop trading if 5% lost today
            'min_confidence': 0.75,  # Increased confidence for "High Accuracy" signals
            'trading_pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'initial_balance': 10000.0,
        }

        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()

    def save_config(self):
        """Save trading configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_trading_history(self):
        """Load trading history from file"""
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                self.trades = json.load(f)
        else:
            self.trades = {
                'balance': self.config.get('initial_balance', 10000.0) if hasattr(self, 'config') else 10000.0,
                'positions': [],
                'closed_trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0
                }
            }
            self.save_trading_history()

    def save_trading_history(self):
        """Save trading history to file"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)

    def get_current_price(self, symbol):
        """Get current price for a symbol

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            float: Current price
        """
        if self.connected and self.client:
            try:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except BinanceAPIException as e:
                logger.error("Failed to fetch price for %s: %s", symbol, e)
                return None
        else:
            return self.get_simulated_price(symbol)

    def get_simulated_price(self, symbol):
        """Get simulated price for paper trading"""
        sim_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 400.0,
            'EURUSD': 1.0842,
            'GBPUSD': 1.2634,
        }
        return sim_prices.get(symbol, 100.0)

    def execute_trade(self, symbol, signal, confidence, current_price=None, quantity=None):
        """Execute a trade based on model signal

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: Model confidence (0-1)
            current_price: Current asset price (optional)
            quantity: Trade quantity (optional, calculated if None)

        Returns:
            dict: Trade result
        """
        if signal == 'HOLD' or confidence < self.config['min_confidence']:
            return {'status': 'skipped', 'reason': 'Low confidence or HOLD signal'}

        # Check if we've exceeded daily trade limit
        today_trades = sum(1 for t in self.trades['closed_trades']
                          if t['date'][:10] == datetime.now().strftime('%Y-%m-%d'))

        if today_trades >= self.config['max_trades_per_day']:
            return {'status': 'skipped', 'reason': 'Daily trade limit reached'}

        # Get current price
        if current_price is None:
            current_price = self.get_current_price(symbol)

        if current_price is None:
            return {'status': 'error', 'reason': 'Could not fetch price'}

        # Calculate position size based on risk
        if quantity is None:
            risk_amount = self.trades['balance'] * self.config['risk_per_trade']
            quantity = risk_amount / current_price

        # Calculate stop loss and multiple take profit targets
        if signal == 'BUY':
            stop_loss = current_price * (1 - self.config['stop_loss_pct'])
            tp1 = current_price * (1 + self.config['tp_target1_pct'])
            tp2 = current_price * (1 + self.config['tp_target2_pct'])
        else:  # SELL
            stop_loss = current_price * (1 + self.config['stop_loss_pct'])
            tp1 = current_price * (1 - self.config['tp_target1_pct'])
            tp2 = current_price * (1 - self.config['tp_target2_pct'])

        # Create trade record
        trade = {
            'id': len(self.trades['closed_trades']) + len(self.trades['positions']) + 1,
            'symbol': symbol,
            'signal': signal,
            'entry_price': current_price,
            'highest_price': current_price if signal == 'BUY' else 999999.0,
            'lowest_price': current_price if signal == 'SELL' else 0.0,
            'quantity': quantity,
            'remaining_quantity': quantity,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp1_hit': False,
            'confidence': confidence,
            'status': 'open',
            'entry_time': datetime.now().isoformat(),
            'paper_trading': not self.connected
        }

        # Execute on Binance (if connected) or paper trade
        if self.connected and self.client:
            try:
                # Place market order on Binance Testnet
                order = self.client.create_order(
                    symbol=symbol,
                    side='BUY' if signal == 'BUY' else 'SELL',
                    type='MARKET',
                    quantity=round(quantity, 6)
                )
                trade['order_id'] = order['orderId']
                trade['status'] = 'executed'
            except BinanceAPIException as e:
                return {'status': 'error', 'reason': str(e)}
        else:
            # Paper trading
            trade['status'] = 'paper_executed'

        # Add to open positions
        self.trades['positions'].append(trade)

        if not self.connected:
            if signal == 'BUY':
                self.trades['balance'] -= current_price * quantity

        self.save_trading_history()

        return {
            'status': 'success',
            'trade': trade,
            'message': f"{'Paper ' if not self.connected else ''}Executed {signal} {quantity:.6f} {symbol} @ ${current_price:.2f}"
        }

    def check_and_close_positions(self):
        closed_positions = []
        updates_made = False

        for position in self.trades['positions']:
            symbol = position['symbol']
            current_price = self.get_current_price(symbol)

            if current_price is None:
                continue

            should_close_all = False
            should_close_partial = False
            close_reason = None

            # Update Trailing Stop logic
            if position['signal'] == 'BUY':
                if current_price > position.get('highest_price', 0):
                    position['highest_price'] = current_price
                    # Move stop loss up
                    new_sl = current_price * (1 - self.config.get('trailing_stop_pct', 0.02))
                    if new_sl > position['stop_loss']:
                        position['stop_loss'] = new_sl
                        updates_made = True
            else: # SELL
                if current_price < position.get('lowest_price', 999999):
                    position['lowest_price'] = current_price
                    # Move stop loss down
                    new_sl = current_price * (1 + self.config.get('trailing_stop_pct', 0.02))
                    if new_sl < position['stop_loss']:
                        position['stop_loss'] = new_sl
                        updates_made = True

            # 1. Check stop loss
            if position['signal'] == 'BUY' and current_price <= position['stop_loss']:
                should_close_all = True
                close_reason = 'stop_loss'
            elif position['signal'] == 'SELL' and current_price >= position['stop_loss']:
                should_close_all = True
                close_reason = 'stop_loss'

            # 2. Check Take Profit 1 (Close 50%)
            if not position.get('tp1_hit', False):
                if (position['signal'] == 'BUY' and current_price >= position['tp1']) or \
                   (position['signal'] == 'SELL' and current_price <= position['tp1']):
                    should_close_partial = True
                    close_reason = 'tp1'

            # 3. Check Take Profit 2 (Close All)
            if (position['signal'] == 'BUY' and current_price >= position['tp2']) or \
               (position['signal'] == 'SELL' and current_price <= position['tp2']):
                should_close_all = True
                close_reason = 'tp2'

            if should_close_all:
                # Close entire position
                qty_to_close = position.get('remaining_quantity', position['quantity'])
                self._process_close(position, current_price, qty_to_close, close_reason, is_partial=False)
                closed_positions.append(position)
                updates_made = True
            elif should_close_partial:
                # Close half position
                qty_to_close = position['quantity'] * 0.5
                self._process_close(position, current_price, qty_to_close, close_reason, is_partial=True)
                updates_made = True

        # Remove closed positions
        self.trades['positions'] = [p for p in self.trades['positions'] if p['status'] != 'closed']

        if updates_made or closed_positions:
            self.save_trading_history()

        return closed_positions

    def _process_close(self, position, current_price, quantity, reason, is_partial=False):
        """Helper to process a partial or full close"""
        # Execute order on Binance if live
        if self.connected and self.client:
            try:
                self.client.create_order(
                    symbol=position['symbol'],
                    side='SELL' if position['signal'] == 'BUY' else 'BUY',
                    type='MARKET',
                    quantity=round(quantity, 6)
                )
            except Exception as e:
                logger.error("Failed to close Binance position: %s", e)

        # Calculate PnL for this chunk
        entry_price = position['entry_price']
        if position['signal'] == 'BUY':
            pnl = (current_price - entry_price) * quantity
        else:
            pnl = (entry_price - current_price) * quantity

        # Update balance
        if not self.connected:
            # Paper trading math
            self.trades['balance'] += (current_price * quantity) if position['signal'] == 'BUY' else (entry_price * quantity + pnl)

        if is_partial:
            position['tp1_hit'] = True
            position['remaining_quantity'] -= quantity
            # Realized PnL is tracked in closed trades
            partial_record = position.copy()
            partial_record['status'] = 'partially_closed'
            partial_record['exit_price'] = current_price
            partial_record['close_reason'] = reason
            partial_record['pnl'] = pnl
            partial_record['quantity'] = quantity
            self.trades['closed_trades'].append(partial_record)
        else:
            position['status'] = 'closed'
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now().isoformat()
            position['close_reason'] = reason
            position['pnl'] = pnl
            position['quantity'] = quantity # Final chunk size for record
            self.trades['closed_trades'].append(position)

        # Update performance
        self.trades['performance']['total_trades'] += 1
        self.trades['performance']['total_profit'] += pnl
        if pnl > 0: self.trades['performance']['winning_trades'] += 1
        else: self.trades['performance']['losing_trades'] += 1

    def get_performance_summary(self):
        """Get trading performance summary"""
        perf = self.trades['performance']

        # Calculate additional metrics
        if self.trades['closed_trades']:
            pnl_list = [t['pnl'] for t in self.trades['closed_trades']]

            # Sharpe Ratio (simplified)
            if len(pnl_list) > 1:
                returns_mean = np.mean(pnl_list)
                returns_std = np.std(pnl_list)
                sharpe = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0
                perf['sharpe_ratio'] = round(sharpe, 2)

            # Max drawdown
            cumulative = np.cumsum(pnl_list)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            perf['max_drawdown'] = round(float(np.min(drawdown)), 2)

            # Average profit per trade
            perf['avg_profit_per_trade'] = round(np.mean(pnl_list), 2)

        return perf

    def get_open_positions(self):
        """Get list of open positions"""
        return self.trades['positions']

    def get_closed_trades(self, limit=50):
        """Get recent closed trades

        Args:
            limit: Maximum number of trades to return

        Returns:
            list: Recent closed trades
        """
        return self.trades['closed_trades'][-limit:]

    def get_balance(self):
        """Get current account balance"""
        if self.connected and self.client:
            try:
                account = self.client.get_account()
                # Get USDT balance
                for balance in account['balances']:
                    if balance['asset'] == 'USDT':
                        return float(balance['free'])
            except BinanceAPIException as e:
                logger.error("Failed to retrieve balance: %s", e)

        # Return paper trading balance
        return self.trades['balance']

    def reset_paper_trading(self):
        """Reset paper trading history"""
        if not self.connected:
            self.trades = {
                'balance': self.config['initial_balance'],
                'positions': [],
                'closed_trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0
                }
            }
            self.save_trading_history()
            return True
        return False
