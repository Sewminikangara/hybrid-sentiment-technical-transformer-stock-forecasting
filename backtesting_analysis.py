"""
Backtesting Analysis for Stock Price Prediction Models
Simulates real trading using model predictions
Calculates portfolio performance, returns, and risk metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models.transformer_early_fusion import EarlyFusionPredictor
from models.transformer_late_fusion import LateFusionPredictor
from models.transformer_attention_fusion import AttentionFusionPredictor
from models.transformer_technical import TechnicalTransformerPredictor

print("=" * 80)
print("BACKTESTING ANALYSIS - TRADING SIMULATION")
print("=" * 80)
print("\nSimulating real trading with model predictions")
print("Calculating returns, risk metrics, and profitability")
print("=" * 80)

class TradingSimulator:
    """Simulates trading based on model predictions"""
    
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        """
        Args:
            initial_capital: Starting portfolio value ($)
            transaction_cost: Trading fee as fraction (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def generate_signals(self, actual_prices, predicted_prices, threshold=0.002):
        """
        Generate buy/sell signals from predictions
        
        Strategy:
        - BUY: If predicted price > current price (expect increase)
        - SELL: If predicted price < current price (expect decrease)
        - HOLD: Otherwise
        
        Args:
            threshold: Minimum expected return to generate signal (default 0.2%)
        
        Returns:
            signals: 1 (buy), -1 (sell), 0 (hold)
        """
        signals = np.zeros(len(predicted_prices))
        
        for i in range(len(predicted_prices)):
            if i == 0:
                signals[i] = 0  # No signal on first day
            else:
                current_price = actual_prices[i-1]
                predicted_next = predicted_prices[i]
                
                # Calculate expected return
                expected_return = (predicted_next - current_price) / current_price
                
                # Signal thresholds (adjustable)
                if expected_return > threshold:  # Expect gain
                    signals[i] = 1  # BUY
                elif expected_return < -threshold:  # Expect loss
                    signals[i] = -1  # SELL
                else:
                    signals[i] = 0  # HOLD
        
        return signals
    
    def simulate_trading(self, dates, actual_prices, signals):
        """
        Simulate trading execution
        
        Returns:
            portfolio_values: Portfolio value over time
            trades: List of executed trades
        """
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        shares = 0
        
        portfolio_values = []
        trades = []
        
        for i, (date, price, signal) in enumerate(zip(dates, actual_prices, signals)):
            # Execute trade based on signal
            if signal == 1 and cash > 0:  # BUY
                # Buy as many shares as possible
                shares_to_buy = cash / (price * (1 + self.transaction_cost))
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                
                shares += shares_to_buy
                cash -= cost
                
                trades.append({
                    'date': date,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'cost': cost
                })
                
            elif signal == -1 and shares > 0:  # SELL
                # Sell all shares
                revenue = shares * price * (1 - self.transaction_cost)
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares,
                    'revenue': revenue
                })
                
                cash += revenue
                shares = 0
            
            # Calculate current portfolio value
            portfolio_value = cash + (shares * price)
            portfolio_values.append(portfolio_value)
        
        return np.array(portfolio_values), trades
    
    def calculate_metrics(self, portfolio_values, dates):
        """
        Calculate performance and risk metrics
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        # Calculate returns
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Annualized return (assuming 252 trading days)
        days = len(portfolio_values)
        years = days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe Ratio (risk-free rate = 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Win Rate (days with positive returns)
        win_rate = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
        
        return {
            'Total_Return_%': total_return * 100,
            'Annualized_Return_%': annualized_return * 100,
            'Volatility_%': volatility * 100,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown_%': max_drawdown * 100,
            'Win_Rate_%': win_rate * 100,
            'Final_Value_$': portfolio_values[-1],
            'Profit_$': portfolio_values[-1] - self.initial_capital
        }
    
    def buy_and_hold_benchmark(self, actual_prices):
        """
        Calculate buy-and-hold strategy performance
        Buy on first day, sell on last day
        """
        shares = self.initial_capital / (actual_prices[0] * (1 + self.transaction_cost))
        final_value = shares * actual_prices[-1] * (1 - self.transaction_cost)
        
        # Portfolio values over time
        portfolio_values = shares * actual_prices * (1 - self.transaction_cost)
        
        return portfolio_values

def load_model_and_predict(stock, model_type):
    """Load trained model and generate predictions on test set"""
    
    print(f"\n  Loading {model_type} model for {stock}...")
    
    # Load hybrid data
    hybrid_file = 'data_processed/hybrid/hybrid_data_all_stocks_20260207_093427.csv'
    df = pd.read_csv(hybrid_file)
    stock_data = df[df['Stock'] == stock].copy()
    stock_data = stock_data.sort_values('Date')
    
    # Separate features
    technical_cols = [c for c in stock_data.columns if c not in 
                     ['Date', 'Stock', 'stock', 'date', 'source', 'Close',
                      'sentiment_score', 'sentiment_label', 'confidence',
                      'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
                      'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']]
    
    sentiment_cols = ['sentiment_score', 'sentiment_positive', 'sentiment_negative', 
                     'sentiment_neutral', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_volatility']
    
    # Prepare sequences
    SEQUENCE_LENGTH = 60
    technical_data = stock_data[technical_cols].values
    sentiment_data = stock_data[sentiment_cols].values
    prices = stock_data['Close'].values
    dates = pd.to_datetime(stock_data['Date']).values
    
    X_tech_seq = []
    X_sent_seq = []
    y = []
    test_dates = []
    
    for i in range(SEQUENCE_LENGTH, len(stock_data)):
        X_tech_seq.append(technical_data[i-SEQUENCE_LENGTH:i])
        X_sent_seq.append(sentiment_data[i-SEQUENCE_LENGTH:i])
        y.append(prices[i])
        test_dates.append(dates[i])
    
    X_tech = np.array(X_tech_seq)
    X_sent = np.array(X_sent_seq)
    y_actual = np.array(y)
    
    # Use only test set (last 15%)
    test_size = int(len(X_tech) * 0.15)
    X_tech_test = X_tech[-test_size:]
    X_sent_test = X_sent[-test_size:]
    y_test = y_actual[-test_size:]
    test_dates = test_dates[-test_size:]
    
    # Load model
    model_path = f"results/{stock}_{model_type}.pt"
    
    if model_type == 'early_fusion':
        predictor = EarlyFusionPredictor(len(technical_cols), len(sentiment_cols))
    elif model_type == 'late_fusion':
        predictor = LateFusionPredictor(len(technical_cols), len(sentiment_cols))
    elif model_type == 'attention_fusion':
        predictor = AttentionFusionPredictor(len(technical_cols), len(sentiment_cols))
    else:  # technical only
        predictor = TechnicalTransformerPredictor(len(technical_cols))
    
    predictor.load_model(model_path)
    
    # Generate predictions
    if model_type == 'technical_transformer':
        y_pred = predictor.predict(X_tech_test)
    else:
        y_pred = predictor.predict(X_tech_test, X_sent_test)
    
    return y_test, y_pred.flatten(), test_dates

def backtest_model(stock, model_type, threshold=0.002):
    """
    Backtest a specific model on a stock
    
    Args:
        stock: Stock ticker
        model_type: Model architecture
        threshold: Signal threshold for trading (default 0.2% = 0.002)
    """
    
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {stock} - {model_type.upper()} (Threshold: {threshold*100:.1f}%)")
    print('='*80)
    
    # Load predictions
    actual, predicted, dates = load_model_and_predict(stock, model_type)
    
    # Initialize simulator
    simulator = TradingSimulator(initial_capital=10000)
    
    # Generate trading signals
    print(f"  Generating trading signals (threshold: {threshold*100:.1f}%)...")
    signals = simulator.generate_signals(actual, predicted, threshold=threshold)
    
    # Simulate trading
    print("  Simulating trading...")
    portfolio_values, trades = simulator.simulate_trading(dates, actual, signals)
    
    # Calculate metrics
    metrics = simulator.calculate_metrics(portfolio_values, dates)
    
    # Buy-and-hold benchmark
    print("  Calculating buy-and-hold benchmark...")
    benchmark_values = simulator.buy_and_hold_benchmark(actual)
    benchmark_metrics = simulator.calculate_metrics(benchmark_values, dates)
    
    # Print results
    print(f"\n  MODEL STRATEGY:")
    for key, value in metrics.items():
        print(f"    {key:25s}: {value:10.2f}")
    
    print(f"\n  BUY-AND-HOLD BENCHMARK:")
    for key, value in benchmark_metrics.items():
        print(f"    {key:25s}: {value:10.2f}")
    
    print(f"\n  TRADES EXECUTED: {len(trades)}")
    print(f"  Signal Distribution: BUY: {np.sum(signals==1)}, SELL: {np.sum(signals==-1)}, HOLD: {np.sum(signals==0)}")
    
    return {
        'stock': stock,
        'model': model_type,
        'metrics': metrics,
        'benchmark_metrics': benchmark_metrics,
        'portfolio_values': portfolio_values,
        'benchmark_values': benchmark_values,
        'dates': dates,
        'trades': trades,
        'signals': signals
    }

def main():
    """Main backtesting execution"""
    
    print("\n[1/3] Running backtesting simulations...")
    
    # Stocks to test
    stocks = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
    
    # Models to test
    models = ['early_fusion', 'late_fusion', 'attention_fusion']
    
    all_results = []
    
    for stock in stocks:
        for model in models:
            try:
                result = backtest_model(stock, model)
                all_results.append(result)
            except Exception as e:
                print(f"\n  ✗ Error backtesting {stock} {model}: {e}")
    
    # Generate comparison plots
    print("\n[2/3] Generating performance plots...")
    
    # Create results directory
    output_dir = Path('graphs/backtesting')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Portfolio Value Over Time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, stock in enumerate(stocks):
        ax = axes[idx]
        stock_results = [r for r in all_results if r['stock'] == stock]
        
        # Plot benchmark first
        if stock_results:
            dates = stock_results[0]['dates']
            benchmark = stock_results[0]['benchmark_values']
            ax.plot(dates, benchmark, '--', label='Buy-and-Hold', linewidth=2, alpha=0.7)
        
        # Plot each model
        for result in stock_results:
            ax.plot(result['dates'], result['portfolio_values'], 
                   label=result['model'].replace('_', ' ').title(), linewidth=1.5)
        
        ax.set_title(f'{stock} - Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=10000, color='r', linestyle=':', alpha=0.5, label='Initial Capital')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'portfolio_performance.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'portfolio_performance.png'}")
    
    # Plot 2: Returns Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    returns_data = []
    for result in all_results:
        returns_data.append({
            'Stock': result['stock'],
            'Model': result['model'].replace('_', ' ').title(),
            'Return_%': result['metrics']['Total_Return_%']
        })
    
    returns_df = pd.DataFrame(returns_data)
    returns_pivot = returns_df.pivot(index='Stock', columns='Model', values='Return_%')
    returns_pivot.plot(kind='bar', ax=ax)
    ax.set_title('Total Returns by Model and Stock')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Stock')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'returns_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'returns_comparison.png'}")
    
    # Plot 3: Risk-Return Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for result in all_results:
        metrics = result['metrics']
        ax.scatter(metrics['Volatility_%'], metrics['Annualized_Return_%'],
                  s=100, alpha=0.6,
                  label=f"{result['stock']} - {result['model'].replace('_', ' ').title()}")
    
    ax.set_xlabel('Volatility (%) - Risk')
    ax.set_ylabel('Annualized Return (%)')
    ax.set_title('Risk-Return Profile')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_return.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'risk_return.png'}")
    
    # Save summary table
    print("\n[3/3] Saving summary results...")
    
    summary_data = []
    for result in all_results:
        row = {
            'Stock': result['stock'],
            'Model': result['model'],
            **result['metrics']
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f'backtesting_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ Saved: {summary_file}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("BACKTESTING SUMMARY")
    print("=" * 80)
    
    print("\nAverage Performance by Model:")
    avg_perf = summary_df.groupby('Model')[['Total_Return_%', 'Sharpe_Ratio', 'Max_Drawdown_%', 'Win_Rate_%']].mean()
    print(avg_perf.to_string())
    
    print("\nBest Performers:")
    best_return = summary_df.nlargest(3, 'Total_Return_%')[['Stock', 'Model', 'Total_Return_%', 'Sharpe_Ratio']]
    print("\nHighest Returns:")
    print(best_return.to_string(index=False))
    
    best_sharpe = summary_df.nlargest(3, 'Sharpe_Ratio')[['Stock', 'Model', 'Sharpe_Ratio', 'Total_Return_%']]
    print("\nBest Risk-Adjusted Returns (Sharpe Ratio):")
    print(best_sharpe.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("BACKTESTING ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in: {output_dir}/")
    print("\nKey Findings:")
    print("- Portfolio simulations completed for 4 stocks x 3 models")
    print("- Compared against buy-and-hold benchmark")
    print("- Calculated comprehensive risk and return metrics")
    print("- Ready for dissertation Chapter 4!")

if __name__ == "__main__":
    main()
