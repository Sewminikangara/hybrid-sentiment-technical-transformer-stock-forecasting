"""
Automated Trading  for Binance Testnet
Executes trades based on AI model predictions
"""

import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException


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
        if api_key and api_secret:
            if testnet:
                # Binance Testnet
                self.client = Client(api_key, api_secret, testnet=True)
                self.client.API_URL = 'https://testnet.binance.vision/api'
            else:
                # Live Binance (use with caution!)
                self.client = Client(api_key, api_secret)
            self.connected = True
        else:
            # Paper trading mode (simulated)
            self.client = None
            self.connected = False
        
        # Load or initialize trading history
        self.load_trading_history()
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load trading configuration"""
        default_config = {
            'risk_per_trade': 0.02,  # 2% of portfolio per trade
            'max_trades_per_day': 10,
            'stop_loss_pct': 0.02,  # 2% stop loss
            'take_profit_pct': 0.05,  # 5% take profit
            'min_confidence': 0.65,  # Minimum 65% confidence to trade
            'trading_pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'initial_balance': 10000.0,  # $10,000 initial balance for paper trading
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
                print(f"Error getting price for {symbol}: {e}")
                return None
        else:
            # Paper trading - use simulated/last known price
            # In production, you'd fetch from a free API
            return self.get_simulated_price(symbol)
    
    def get_simulated_price(self, symbol):
        """Get simulated price for paper trading"""
        # Simulated prices for demo (in production, fetch from API)
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
        
        # Calculate stop loss and take profit
        if signal == 'BUY':
            stop_loss = current_price * (1 - self.config['stop_loss_pct'])
            take_profit = current_price * (1 + self.config['take_profit_pct'])
        else:  # SELL
            stop_loss = current_price * (1 + self.config['stop_loss_pct'])
            take_profit = current_price * (1 - self.config['take_profit_pct'])
        
        # Create trade record
        trade = {
            'id': len(self.trades['closed_trades']) + len(self.trades['positions']) + 1,
            'symbol': symbol,
            'signal': signal,
            'entry_price': current_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
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
        
        # Update balance (deduct for BUY, add for SELL in paper trading)
        if not self.connected:
            if signal == 'BUY':
                self.trades['balance'] -= current_price * quantity
            # For SELL, we'd need to check if we have the position first
        
        self.save_trading_history()
        
        return {
            'status': 'success',
            'trade': trade,
            'message': f"{'Paper ' if not self.connected else ''}Executed {signal} {quantity:.6f} {symbol} @ ${current_price:.2f}"
        }
    
    def check_and_close_positions(self):
        """Check open positions for stop loss or take profit triggers"""
        closed_positions = []
        
        for position in self.trades['positions']:
            symbol = position['symbol']
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                continue
            
            should_close = False
            close_reason = None
            
            # Check stop loss
            if position['signal'] == 'BUY' and current_price <= position['stop_loss']:
                should_close = True
                close_reason = 'stop_loss'
            elif position['signal'] == 'SELL' and current_price >= position['stop_loss']:
                should_close = True
                close_reason = 'stop_loss'
            
            # Check take profit
            if position['signal'] == 'BUY' and current_price >= position['take_profit']:
                should_close = True
                close_reason = 'take_profit'
            elif position['signal'] == 'SELL' and current_price <= position['take_profit']:
                should_close = True
                close_reason = 'take_profit'
            
            if should_close:
                # Close position
                position['exit_price'] = current_price
                position['exit_time'] = datetime.now().isoformat()
                position['close_reason'] = close_reason
                position['status'] = 'closed'
                
                # Calculate profit/loss
                if position['signal'] == 'BUY':
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:  # SELL
                    pnl = (position['entry_price'] - current_price) * position['quantity']
                
                position['pnl'] = pnl
                position['pnl_pct'] = (pnl / (position['entry_price'] * position['quantity'])) * 100
                
                # Update balance
                if not self.connected:
                    if position['signal'] == 'BUY':
                        self.trades['balance'] += current_price * position['quantity']
                    else:
                        self.trades['balance'] += position['entry_price'] * position['quantity']
                    self.trades['balance'] += pnl
                
                # Update performance metrics
                self.trades['performance']['total_trades'] += 1
                self.trades['performance']['total_profit'] += pnl
                
                if pnl > 0:
                    self.trades['performance']['winning_trades'] += 1
                else:
                    self.trades['performance']['losing_trades'] += 1
                
                # Calculate win rate
                total = self.trades['performance']['total_trades']
                wins = self.trades['performance']['winning_trades']
                self.trades['performance']['win_rate'] = (wins / total * 100) if total > 0 else 0
                
                # Move to closed trades
                self.trades['closed_trades'].append(position)
                closed_positions.append(position)
        
        # Remove closed positions from open positions
        self.trades['positions'] = [p for p in self.trades['positions'] if p['status'] != 'closed']
        
        if closed_positions:
            self.save_trading_history()
        
        return closed_positions
    
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
                print(f"Error getting balance: {e}")
        
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
