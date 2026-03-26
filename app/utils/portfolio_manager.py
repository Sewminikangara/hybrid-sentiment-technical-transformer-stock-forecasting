import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple


class PortfolioManager:
    """Enterprise-grade portfolio risk management"""

    def __init__(self):
        self.portfolio = {}
        self.predictions = {}

    def add_asset(self, symbol: str, quantity: float, current_price: float, prediction: Dict):
        """Add asset to portfolio with predictions"""
        self.portfolio[symbol] = {
            'quantity': quantity,
            'current_price': current_price,
            'value': quantity * current_price
        }
        self.predictions[symbol] = prediction

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return sum(asset['value'] for asset in self.portfolio.values())

    def calculate_var(self, confidence_level: float = 0.95, time_horizon: int = 1) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR)

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            (VaR in dollars, VaR in percentage)
        """
        if not self.predictions:
            return 0.0, 0.0

        # Get expected returns for each asset
        returns = []
        weights = []
        total_value = self.calculate_portfolio_value()

        for symbol, asset in self.portfolio.items():
            if symbol in self.predictions:
                pred = self.predictions[symbol]
                current_price = asset['current_price']
                predicted_price = pred.get('predicted_price', current_price)

                # Calculate expected return
                expected_return = (predicted_price - current_price) / current_price
                returns.append(expected_return)

                # Weight by portfolio allocation
                weight = asset['value'] / total_value
                weights.append(weight)

        if not returns:
            return 0.0, 0.0

        # Calculate portfolio return
        portfolio_return = np.dot(returns, weights)

        # Simple VaR calculation (assumes normal distribution)
        # In production, use historical simulation or Monte Carlo
        returns_std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence_level)

        var_pct = abs(z_score * returns_std * np.sqrt(time_horizon))
        var_dollars = var_pct * total_value

        return var_dollars, var_pct * 100

    def calculate_cvar(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)

        Returns:
            (CVaR in dollars, CVaR in percentage)
        """
        var_dollars, var_pct = self.calculate_var(confidence_level)

        # CVaR is typically 1.3-1.5x VaR for normal distribution
        cvar_dollars = var_dollars * 1.4
        cvar_pct = var_pct * 1.4

        return cvar_dollars, cvar_pct

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return)

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        if not self.predictions:
            return 0.0

        returns = []
        weights = []
        total_value = self.calculate_portfolio_value()

        for symbol, asset in self.portfolio.items():
            if symbol in self.predictions:
                pred = self.predictions[symbol]
                current_price = asset['current_price']
                predicted_price = pred.get('predicted_price', current_price)

                expected_return = (predicted_price - current_price) / current_price
                returns.append(expected_return)

                weight = asset['value'] / total_value
                weights.append(weight)

        if not returns:
            return 0.0

        # Portfolio return
        portfolio_return = np.dot(returns, weights)

        # Portfolio volatility (simplified)
        returns_std = np.std(returns)

        # Annualize (assuming daily returns)
        annual_return = portfolio_return * 252
        annual_std = returns_std * np.sqrt(252)

        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_std if annual_std > 0 else 0

        return sharpe

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02, target_return: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted return)

        Args:
            risk_free_rate: Annual risk-free rate
            target_return: Minimum acceptable return

        Returns:
            Sortino ratio
        """
        if not self.predictions:
            return 0.0

        returns = []
        weights = []
        total_value = self.calculate_portfolio_value()

        for symbol, asset in self.portfolio.items():
            if symbol in self.predictions:
                pred = self.predictions[symbol]
                current_price = asset['current_price']
                predicted_price = pred.get('predicted_price', current_price)

                expected_return = (predicted_price - current_price) / current_price
                returns.append(expected_return)

                weight = asset['value'] / total_value
                weights.append(weight)

        if not returns:
            return 0.0

        # Portfolio return
        portfolio_return = np.dot(returns, weights)

        # Downside deviation (only negative returns)
        downside_returns = [min(0, r - target_return) for r in returns]
        downside_std = np.sqrt(np.mean(np.square(downside_returns)))

        # Annualize
        annual_return = portfolio_return * 252
        annual_downside_std = downside_std * np.sqrt(252)

        # Sortino ratio
        sortino = (annual_return - risk_free_rate) / annual_downside_std if annual_downside_std > 0 else 0

        return sortino

    def calculate_diversification_score(self) -> float:
        """
        Calculate portfolio diversification (0-100)
        100 = perfectly diversified, 0 = single asset

        Returns:
            Diversification score (0-100)
        """
        if len(self.portfolio) <= 1:
            return 0.0

        # Calculate Herfindahl index
        total_value = self.calculate_portfolio_value()
        weights = [asset['value'] / total_value for asset in self.portfolio.values()]

        herfindahl = sum(w**2 for w in weights)

        # Convert to diversification score (inverse of concentration)
        # Perfect diversification = 1/n, max concentration = 1
        n = len(self.portfolio)
        min_herfindahl = 1 / n

        diversification_score = (1 - (herfindahl - min_herfindahl) / (1 - min_herfindahl)) * 100

        return diversification_score

    def get_asset_allocation(self) -> pd.DataFrame:
        """
        Get current asset allocation breakdown

        Returns:
            DataFrame with allocation details
        """
        total_value = self.calculate_portfolio_value()

        data = []
        for symbol, asset in self.portfolio.items():
            allocation_pct = (asset['value'] / total_value) * 100

            # Get prediction info
            pred_info = self.predictions.get(symbol, {})
            predicted_price = pred_info.get('predicted_price', asset['current_price'])
            confidence = pred_info.get('confidence', 0.5)

            expected_return = ((predicted_price - asset['current_price']) / asset['current_price']) * 100

            data.append({
                'Symbol': symbol,
                'Quantity': asset['quantity'],
                'Current Price': asset['current_price'],
                'Value': asset['value'],
                'Allocation (%)': allocation_pct,
                'Predicted Price': predicted_price,
                'Expected Return (%)': expected_return,
                'Confidence': confidence
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Allocation (%)', ascending=False)

        return df

    def recommend_rebalancing(self, target_allocation: Dict[str, float] = None) -> pd.DataFrame:
        """
        Recommend rebalancing actions

        Args:
            target_allocation: Target allocation percentages (e.g., {'AAPL': 25, 'GOOGL': 25, ...})
                              If None, equal weight allocation

        Returns:
            DataFrame with rebalancing recommendations
        """
        current_allocation = self.get_asset_allocation()
        total_value = self.calculate_portfolio_value()

        if target_allocation is None:
            # Equal weight
            n = len(self.portfolio)
            target_allocation = {symbol: 100/n for symbol in self.portfolio.keys()}

        recommendations = []

        for symbol in self.portfolio.keys():
            current_pct = current_allocation[current_allocation['Symbol'] == symbol]['Allocation (%)'].values[0]
            target_pct = target_allocation.get(symbol, 0)

            difference_pct = target_pct - current_pct
            difference_value = (difference_pct / 100) * total_value

            if abs(difference_pct) > 1:  # Only if difference > 1%
                action = "BUY" if difference_value > 0 else "SELL"

                recommendations.append({
                    'Symbol': symbol,
                    'Current (%)': current_pct,
                    'Target (%)': target_pct,
                    'Difference (%)': difference_pct,
                    'Action': action,
                    'Amount ($)': abs(difference_value)
                })

        return pd.DataFrame(recommendations)

    def calculate_maximum_drawdown(self, historical_returns: List[float]) -> float:
        """
        Calculate maximum drawdown from historical returns

        Args:
            historical_returns: List of historical returns

        Returns:
            Maximum drawdown percentage
        """
        if not historical_returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(historical_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = abs(np.min(drawdown)) * 100

        return max_drawdown

    def stress_test(self, scenarios: Dict[str, float]) -> Dict:
        """
        Run stress test scenarios

        Args:
            scenarios: Dict of scenario names to market shock percentages
                      e.g., {'Market Crash': -0.20, 'Bull Run': 0.30}

        Returns:
            Dict of scenario results
        """
        results = {}
        total_value = self.calculate_portfolio_value()

        for scenario_name, shock_pct in scenarios.items():
            # Apply shock to entire portfolio
            shocked_value = total_value * (1 + shock_pct)
            loss = total_value - shocked_value
            loss_pct = (loss / total_value) * 100

            results[scenario_name] = {
                'Shock (%)': shock_pct * 100,
                'Portfolio Value After': shocked_value,
                'Loss ($)': loss,
                'Loss (%)': loss_pct
            }

        return results

    def get_risk_report(self) -> Dict:
        """
        Generate comprehensive risk report

        Returns:
            Dict with all risk metrics
        """
        var_dollars, var_pct = self.calculate_var()
        cvar_dollars, cvar_pct = self.calculate_cvar()

        return {
            'portfolio_value': self.calculate_portfolio_value(),
            'num_assets': len(self.portfolio),
            'var_95_dollars': var_dollars,
            'var_95_percent': var_pct,
            'cvar_95_dollars': cvar_dollars,
            'cvar_95_percent': cvar_pct,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'diversification_score': self.calculate_diversification_score()
        }
