"""
PROFESSIONAL Binance Trading Bot with REAL API Integration
This is production-grade code that actually executes trades on Binance Testnet
"""

import ccxt
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


class ProfessionalTrader:
    """
    Professional-grade automated trading bot
    - REAL Binance Testnet integration
    - REAL order execution
    - REAL balance tracking
    - Live price streaming
    """
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Initialize professional trading bot
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret  
            testnet: Use testnet (True) or live (False) - DEFAULT TESTNET FOR SAFETY
        """
        self.testnet = testnet
        self.trades_file = Path(__file__).parent.parent / 'live_trading_history.json'
        self.config_file = Path(__file__).parent.parent / 'live_trading_config.json'
        
        # Initialize CCXT Binance connection
        self.exchange = None
        self.connected = False
        
        if api_key and api_secret:
            try:
                if testnet:
                    # REAL Binance Testnet connection
                    self.exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'spot',
                            'adjustForTimeDifference': True,
                        }
                    })
                    # Set testnet URLs
                    self.exchange.set_sandbox_mode(True)
                else:
                    # Live Binance (DANGEROUS - disabled by default)
                    raise ValueError("Live trading is disabled for safety. Use testnet=True")
                
                # Test connection
                self.exchange.load_markets()
                self.connected = True
                print(f"✅ Connected to Binance {'Testnet' if testnet else 'Live'}")
                
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                self.exchange = None
                self.connected = False
        
        # Load configuration and history
        self.load_config()
        self.load_trading_history()
    
    def load_config(self):
        """Load trading configuration"""
        default_config = {
            'risk_per_trade': 0.02,  # 2%
            'stop_loss_pct': 0.02,  # 2%
            'take_profit_pct': 0.05,  # 5%
            'min_confidence': 0.65,  # 65%
            'max_trades_per_day': 10,
            'initial_balance': 10000.0,  # For paper trading
            'trading_pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_trading_history(self):
        """Load trading history"""
        if self.trades_file.exists():
            with open(self.trades_file, 'r') as f:
                self.trades = json.load(f)
        else:
            self.trades = {
                'paper_balance': self.config.get('initial_balance', 10000.0),
                'positions': [],
                'closed_trades': [],
                'performance': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'win_rate': 0.0,
                }
            }
            self.save_trading_history()
    
    def save_trading_history(self):
        """Save trading history"""
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def get_real_balance(self):
        """
        Get REAL balance from Binance account
        This actually queries the Binance API
        """
        if not self.connected or not self.exchange:
            return None
        
        try:
            balance = self.exchange.fetch_balance()
            
            # Get USDT balance
            usdt_balance = {
                'free': float(balance['USDT']['free']) if 'USDT' in balance else 0.0,
                'used': float(balance['USDT']['used']) if 'USDT' in balance else 0.0,
                'total': float(balance['USDT']['total']) if 'USDT' in balance else 0.0,
            }
            
            # Get all non-zero balances
            all_balances = {}
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and amounts.get('total', 0) > 0:
                    all_balances[currency] = {
                        'free': float(amounts['free']),
                        'used': float(amounts['used']),
                        'total': float(amounts['total'])
                    }
            
            return {
                'usdt': usdt_balance,
                'all': all_balances,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return None
    
    def get_live_price(self, symbol):
        """
        Get REAL live price from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            dict: Price data with bid, ask, last
        """
        if not self.connected or not self.exchange:
            return None
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'bid': float(ticker['bid']),  # Buy price
                'ask': float(ticker['ask']),  # Sell price
                'last': float(ticker['last']),  # Last trade price
                'high': float(ticker['high']),  # 24h high
                'low': float(ticker['low']),  # 24h low
                'volume': float(ticker['baseVolume']),  # 24h volume
                'timestamp': ticker['timestamp'],
                'datetime': ticker['datetime']
            }
            
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
    
    def execute_real_trade(self, symbol, side, quantity, signal_type='market'):
        """
        EXECUTE REAL TRADE on Binance Testnet
        This actually places an order on Binance!
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            quantity: Amount to trade
            signal_type: 'market' or 'limit'
            
        Returns:
            dict: Order result
        """
        if not self.connected or not self.exchange:
            return {'status': 'error', 'reason': 'Not connected to exchange'}
        
        try:
            # Place market order
            if signal_type == 'market':
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=quantity
                )
            else:
                # For limit orders, you'd specify price
                price = self.get_live_price(symbol)['last']
                order = self.exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=quantity,
                    price=price
                )
            
            return {
                'status': 'success',
                'order_id': order['id'],
                'symbol': order['symbol'],
                'side': order['side'],
                'type': order['type'],
                'quantity': float(order['amount']),
                'price': float(order.get('price', 0)),
                'filled': float(order.get('filled', 0)),
                'cost': float(order.get('cost', 0)),
                'timestamp': order['timestamp'],
                'datetime': order['datetime'],
                'raw_order': order
            }
            
        except ccxt.InsufficientFunds as e:
            return {'status': 'error', 'reason': f'Insufficient funds: {str(e)}'}
        except ccxt.InvalidOrder as e:
            return {'status': 'error', 'reason': f'Invalid order: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'reason': f'Order failed: {str(e)}'}
    
    def get_order_status(self, order_id, symbol):
        """
        Check status of an order
        
        Args:
            order_id: Order ID from Binance
            symbol: Trading pair
            
        Returns:
            dict: Order status
        """
        if not self.connected or not self.exchange:
            return None
        
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return {
                'id': order['id'],
                'status': order['status'],  # 'open', 'closed', 'canceled'
                'filled': float(order['filled']),
                'remaining': float(order['remaining']),
                'price': float(order.get('price', 0)),
                'average': float(order.get('average', 0)),
                'cost': float(order.get('cost', 0)),
            }
        except Exception as e:
            print(f"Error fetching order: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """
        Get all open orders
        
        Args:
            symbol: Trading pair (optional)
            
        Returns:
            list: Open orders
        """
        if not self.connected or not self.exchange:
            return []
        
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []
    
    def cancel_order(self, order_id, symbol):
        """
        Cancel an open order
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            bool: Success status
        """
        if not self.connected or not self.exchange:
            return False
        
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            print(f"Error canceling order: {e}")
            return False
    
    def get_historical_candles(self, symbol, timeframe='1h', limit=100):
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading pair
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            limit: Number of candles
            
        Returns:
            DataFrame: OHLCV data
        """
        if not self.connected or not self.exchange:
            return None
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return None
    
    def calculate_position_size(self, symbol, risk_pct=None):
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Trading pair
            risk_pct: Risk percentage (default from config)
            
        Returns:
            float: Position size in base currency
        """
        if risk_pct is None:
            risk_pct = self.config['risk_per_trade']
        
        balance = self.get_real_balance()
        if not balance:
            return 0.0
        
        usdt_balance = balance['usdt']['free']
        price_data = self.get_live_price(symbol)
        
        if not price_data:
            return 0.0
        
        current_price = price_data['last']
        
        # Calculate position size
        risk_amount = usdt_balance * risk_pct
        position_size = risk_amount / current_price
        
        return position_size
    
    def place_ai_trade(self, symbol, signal, confidence, ai_prediction=None):
        """
        Place trade based on AI signal with professional risk management
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: AI confidence (0-1)
            ai_prediction: Optional prediction details
            
        Returns:
            dict: Trade result
        """
        # Validate signal
        if signal == 'HOLD' or confidence < self.config['min_confidence']:
            return {
                'status': 'skipped',
                'reason': f"Signal: {signal}, Confidence: {confidence:.1%} < {self.config['min_confidence']:.1%}"
            }
        
        # Get current price
        price_data = self.get_live_price(symbol)
        if not price_data:
            return {'status': 'error', 'reason': 'Could not fetch price'}
        
        current_price = price_data['last']
        
        # Calculate position size
        position_size = self.calculate_position_size(symbol)
        
        if position_size == 0:
            return {'status': 'error', 'reason': 'Insufficient balance'}
        
        # Execute trade
        side = 'buy' if signal == 'BUY' else 'sell'
        
        result = self.execute_real_trade(
            symbol=symbol,
            side=side,
            quantity=position_size
        )
        
        if result['status'] == 'success':
            # Calculate stop loss and take profit
            if signal == 'BUY':
                stop_loss = current_price * (1 - self.config['stop_loss_pct'])
                take_profit = current_price * (1 + self.config['take_profit_pct'])
            else:
                stop_loss = current_price * (1 + self.config['stop_loss_pct'])
                take_profit = current_price * (1 - self.config['take_profit_pct'])
            
            # Record trade
            trade_record = {
                'id': len(self.trades['positions']) + len(self.trades['closed_trades']) + 1,
                'order_id': result['order_id'],
                'symbol': symbol,
                'signal': signal,
                'side': side,
                'entry_price': current_price,
                'quantity': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'entry_time': datetime.now().isoformat(),
                'status': 'open',
                'ai_prediction': ai_prediction,
                'exchange_order': result
            }
            
            self.trades['positions'].append(trade_record)
            self.save_trading_history()
            
            return {
                'status': 'success',
                'message': f"✅ REAL ORDER EXECUTED: {signal} {position_size:.6f} {symbol} @ ${current_price:.2f}",
                'trade': trade_record,
                'order': result
            }
        else:
            return result
    
    def get_account_summary(self):
        """
        Get comprehensive account summary
        
        Returns:
            dict: Account details
        """
        if not self.connected:
            return {
                'connected': False,
                'mode': 'offline',
                'balance': self.trades.get('paper_balance', 0)
            }
        
        balance = self.get_real_balance()
        
        return {
            'connected': True,
            'mode': 'testnet' if self.testnet else 'live',
            'balance': balance,
            'open_positions': len(self.trades['positions']),
            'total_trades': self.trades['performance']['total_trades'],
            'win_rate': self.trades['performance']['win_rate'],
        }
    
    def get_performance_metrics(self):
        """Get detailed performance metrics"""
        perf = self.trades['performance'].copy()
        
        if self.trades['closed_trades']:
            pnl_list = [t.get('pnl', 0) for t in self.trades['closed_trades']]
            
            if len(pnl_list) > 1:
                returns_mean = np.mean(pnl_list)
                returns_std = np.std(pnl_list)
                perf['sharpe_ratio'] = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0
                
                cumulative = np.cumsum(pnl_list)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = cumulative - running_max
                perf['max_drawdown'] = float(np.min(drawdown))
                perf['avg_profit'] = np.mean(pnl_list)
        
        return perf
