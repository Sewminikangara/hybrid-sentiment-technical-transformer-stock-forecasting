"""
Batch Prediction Engine
For Fintech Startups and Financial Institutions

Provides:
- Multi-asset batch prediction
- Parallel processing
- Confidence intervals
- Risk-adjusted forecasts
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import concurrent.futures
from .predictor import StockPredictor


class BatchPredictor:
    """Efficient batch prediction for multiple assets"""

    def __init__(self, model_name: str = 'early_fusion'):
        self.model_name = model_name

    def predict_single(self, symbol: str, days: int = 7, is_forex: bool = False) -> Dict:
        """
        Predict single asset with error handling

        Returns:
            Dict with predictions, confidence, and metadata
        """
        try:
            predictor = StockPredictor(symbol, self.model_name, is_forex)
            predictions = predictor.predict(days)

            if predictions is None or len(predictions) == 0:
                return {
                    'symbol': symbol,
                    'status': 'failed',
                    'error': 'No predictions generated',
                    'predictions': None
                }

            # Calculate confidence based on prediction stability
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)

            # Inverse of coefficient of variation as confidence
            confidence = 1 / (1 + abs(pred_std / pred_mean)) if pred_mean != 0 else 0.5

            return {
                'symbol': symbol,
                'status': 'success',
                'predictions': predictions,
                'predicted_price': predictions[-1],
                'current_price': predictions[0],
                'expected_return': ((predictions[-1] - predictions[0]) / predictions[0]) * 100,
                'confidence': confidence,
                'volatility': pred_std,
                'days': days,
                'model': self.model_name
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'predictions': None
            }

    def batch_predict(self, symbols: List[str], days: int = 7, is_forex: bool = False,
                     parallel: bool = True, max_workers: int = 4) -> pd.DataFrame:
        """
        Predict multiple assets efficiently

        Args:
            symbols: List of stock/forex symbols
            days: Number of days to predict
            is_forex: Whether symbols are forex pairs
            parallel: Use parallel processing
            max_workers: Max parallel workers

        Returns:
            DataFrame with all predictions
        """
        results = []

        if parallel and len(symbols) > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.predict_single, symbol, days, is_forex): symbol
                    for symbol in symbols
                }

                for future in concurrent.futures.as_completed(future_to_symbol):
                    result = future.result()
                    results.append(result)
        else:
            # Sequential execution
            for symbol in symbols:
                result = self.predict_single(symbol, days, is_forex)
                results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by expected return (best opportunities first)
        if 'expected_return' in df.columns:
            df = df.sort_values('expected_return', ascending=False)

        return df

    def predict_with_risk_metrics(self, symbols: List[str], days: int = 7,
                                  is_forex: bool = False) -> pd.DataFrame:
        """
        Batch predict with risk metrics (VaR, CVaR, Sharpe)

        Returns:
            DataFrame with predictions and risk metrics
        """
        # Get basic predictions
        df = self.batch_predict(symbols, days, is_forex)

        # Add risk metrics
        risk_metrics = []

        for _, row in df.iterrows():
            if row['status'] == 'success' and row['predictions'] is not None:
                predictions = row['predictions']

                # Calculate returns
                returns = np.diff(predictions) / predictions[:-1]

                # Value at Risk (95% confidence)
                var_95 = abs(np.percentile(returns, 5)) * 100

                # Conditional VaR
                cvar_95 = abs(np.mean(returns[returns <= np.percentile(returns, 5)])) * 100

                # Sharpe ratio (simplified)
                sharpe = (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0

                risk_metrics.append({
                    'VaR_95 (%)': var_95,
                    'CVaR_95 (%)': cvar_95,
                    'Sharpe_Ratio': sharpe
                })
            else:
                risk_metrics.append({
                    'VaR_95 (%)': None,
                    'CVaR_95 (%)': None,
                    'Sharpe_Ratio': None
                })

        # Add risk metrics to DataFrame
        risk_df = pd.DataFrame(risk_metrics)
        df = pd.concat([df, risk_df], axis=1)

        return df

    def rank_opportunities(self, symbols: List[str], days: int = 7,
                          is_forex: bool = False, min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Rank trading opportunities by risk-adjusted return

        Args:
            symbols: List of symbols
            days: Prediction horizon
            is_forex: Forex or stock
            min_confidence: Minimum confidence threshold

        Returns:
            DataFrame ranked by opportunity score
        """
        df = self.predict_with_risk_metrics(symbols, days, is_forex)

        # Filter by confidence
        df = df[df['confidence'] >= min_confidence]

        # Calculate opportunity score (return / risk)
        df['opportunity_score'] = (df['expected_return'] / (df['VaR_95 (%)'] + 1)) * df['confidence']

        # Rank
        df = df.sort_values('opportunity_score', ascending=False)

        # Add rank
        df['rank'] = range(1, len(df) + 1)

        return df

    def compare_models(self, symbol: str, days: int = 7, is_forex: bool = False) -> pd.DataFrame:
        """
        Compare predictions across multiple models

        Returns:
            DataFrame comparing all models
        """
        models = ['technical_transformer', 'early_fusion', 'late_fusion', 'attention_fusion']

        results = []

        for model in models:
            predictor = BatchPredictor(model)
            result = predictor.predict_single(symbol, days, is_forex)

            if result['status'] == 'success':
                results.append({
                    'Model': model,
                    'Predicted Price': result['predicted_price'],
                    'Expected Return (%)': result['expected_return'],
                    'Confidence': result['confidence'],
                    'Volatility': result['volatility']
                })

        df = pd.DataFrame(results)

        # Add ensemble prediction (weighted average by confidence)
        if len(df) > 0:
            total_confidence = df['Confidence'].sum()
            ensemble_price = (df['Predicted Price'] * df['Confidence']).sum() / total_confidence
            ensemble_return = (df['Expected Return (%)'] * df['Confidence']).sum() / total_confidence

            df = pd.concat([df, pd.DataFrame([{
                'Model': 'Ensemble (Weighted)',
                'Predicted Price': ensemble_price,
                'Expected Return (%)': ensemble_return,
                'Confidence': df['Confidence'].mean(),
                'Volatility': df['Volatility'].mean()
            }])], ignore_index=True)

        return df

    def portfolio_prediction(self, portfolio: Dict[str, float], days: int = 7,
                            is_forex: bool = False) -> Dict:
        """
        Predict portfolio performance

        Args:
            portfolio: Dict of {symbol: allocation_weight} (weights sum to 1.0)
            days: Prediction horizon
            is_forex: Forex or stock

        Returns:
            Dict with portfolio-level predictions
        """
        symbols = list(portfolio.keys())
        weights = list(portfolio.values())

        # Get predictions for all assets
        df = self.batch_predict(symbols, days, is_forex)

        # Calculate weighted portfolio return
        portfolio_return = 0
        portfolio_confidence = 0

        for symbol, weight in portfolio.items():
            asset_data = df[df['symbol'] == symbol]

            if len(asset_data) > 0 and asset_data.iloc[0]['status'] == 'success':
                portfolio_return += asset_data.iloc[0]['expected_return'] * weight
                portfolio_confidence += asset_data.iloc[0]['confidence'] * weight

        return {
            'portfolio_return': portfolio_return,
            'portfolio_confidence': portfolio_confidence,
            'num_assets': len(symbols),
            'prediction_horizon': days,
            'asset_predictions': df
        }

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics from batch predictions

        Returns:
            Dict with statistics
        """
        successful = df[df['status'] == 'success']

        if len(successful) == 0:
            return {
                'total_assets': len(df),
                'successful': 0,
                'failed': len(df),
                'average_return': 0,
                'average_confidence': 0
            }

        return {
            'total_assets': len(df),
            'successful': len(successful),
            'failed': len(df) - len(successful),
            'average_return': successful['expected_return'].mean(),
            'best_opportunity': successful.iloc[0]['symbol'] if len(successful) > 0 else None,
            'worst_opportunity': successful.iloc[-1]['symbol'] if len(successful) > 0 else None,
            'average_confidence': successful['confidence'].mean(),
            'high_confidence_count': len(successful[successful['confidence'] >= 0.7])
        }
