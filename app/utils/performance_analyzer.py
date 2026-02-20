"""
Performance Analytics Module
For Investment Professionals and Financial Institutions

Provides:
- Model performance tracking
- Prediction accuracy analysis
- Historical error analysis
- Confidence calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class PerformanceAnalyzer:
    """Track and analyze model performance"""
    
    def __init__(self):
        self.history = []
        
    def add_prediction(self, symbol: str, predicted_price: float, actual_price: float,
                      prediction_date: datetime, model: str, confidence: float):
        """
        Record a prediction for later analysis
        
        Args:
            symbol: Asset symbol
            predicted_price: Model's predicted price
            actual_price: Actual realized price
            prediction_date: When prediction was made
            model: Model name
            confidence: Model confidence (0-1)
        """
        error = abs(predicted_price - actual_price)
        error_pct = (error / actual_price) * 100
        
        direction_correct = (predicted_price > actual_price) == (actual_price > predicted_price)
        
        self.history.append({
            'symbol': symbol,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'error': error,
            'error_pct': error_pct,
            'date': prediction_date,
            'model': model,
            'confidence': confidence,
            'direction_correct': direction_correct
        })
    
    def get_accuracy_metrics(self, model: str = None, symbol: str = None,
                            days: int = None) -> Dict:
        """
        Calculate accuracy metrics
        
        Args:
            model: Filter by model name
            symbol: Filter by symbol
            days: Only include last N days
            
        Returns:
            Dict with accuracy metrics
        """
        df = pd.DataFrame(self.history)
        
        if len(df) == 0:
            return {
                'num_predictions': 0,
                'mae': 0,
                'mape': 0,
                'rmse': 0,
                'direction_accuracy': 0,
                'confidence_correlation': 0
            }
        
        # Apply filters
        if model:
            df = df[df['model'] == model]
        if symbol:
            df = df[df['symbol'] == symbol]
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= cutoff_date]
        
        if len(df) == 0:
            return {'num_predictions': 0}
        
        # Calculate metrics
        mae = df['error'].mean()
        mape = df['error_pct'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        direction_accuracy = df['direction_correct'].mean() * 100
        
        # Confidence calibration (correlation between confidence and accuracy)
        df['accuracy'] = 1 - (df['error_pct'] / 100)
        confidence_correlation = df['confidence'].corr(df['accuracy'])
        
        return {
            'num_predictions': len(df),
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'confidence_correlation': confidence_correlation,
            'best_prediction': df.loc[df['error_pct'].idxmin()].to_dict() if len(df) > 0 else None,
            'worst_prediction': df.loc[df['error_pct'].idxmax()].to_dict() if len(df) > 0 else None
        }
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance across all models
        
        Returns:
            DataFrame comparing models
        """
        df = pd.DataFrame(self.history)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        results = []
        
        for model in df['model'].unique():
            metrics = self.get_accuracy_metrics(model=model)
            metrics['model'] = model
            results.append(metrics)
        
        comparison = pd.DataFrame(results)
        comparison = comparison.sort_values('mape')
        
        return comparison
    
    def get_performance_over_time(self, window: int = 7) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            window: Rolling window size in predictions
            
        Returns:
            DataFrame with rolling metrics
        """
        df = pd.DataFrame(self.history)
        
        if len(df) < window:
            return pd.DataFrame()
        
        df = df.sort_values('date')
        
        df['rolling_mape'] = df['error_pct'].rolling(window=window).mean()
        df['rolling_direction_accuracy'] = df['direction_correct'].rolling(window=window).mean() * 100
        
        return df[['date', 'model', 'rolling_mape', 'rolling_direction_accuracy']]
    
    def get_asset_performance(self) -> pd.DataFrame:
        """
        Performance breakdown by asset
        
        Returns:
            DataFrame with per-asset metrics
        """
        df = pd.DataFrame(self.history)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        results = []
        
        for symbol in df['symbol'].unique():
            metrics = self.get_accuracy_metrics(symbol=symbol)
            metrics['symbol'] = symbol
            results.append(metrics)
        
        performance = pd.DataFrame(results)
        performance = performance.sort_values('mape')
        
        return performance
    
    def confidence_calibration_analysis(self) -> pd.DataFrame:
        """
        Analyze if confidence scores are well-calibrated
        
        Returns:
            DataFrame showing calibration by confidence bucket
        """
        df = pd.DataFrame(self.history)
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Create confidence buckets
        df['confidence_bucket'] = pd.cut(df['confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
                                         labels=['Low (0-50%)', 'Medium (50-60%)', 
                                                'High (60-70%)', 'Very High (70-80%)', 
                                                'Excellent (80-100%)'])
        
        # Calculate actual accuracy in each bucket
        calibration = df.groupby('confidence_bucket').agg({
            'error_pct': ['mean', 'std', 'count'],
            'direction_correct': 'mean'
        }).reset_index()
        
        calibration.columns = ['Confidence Bucket', 'Avg Error (%)', 'Error Std', 
                              'Sample Count', 'Direction Accuracy']
        
        calibration['Direction Accuracy'] = calibration['Direction Accuracy'] * 100
        
        return calibration
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """
        Get performance for recent predictions
        
        Args:
            days: Number of recent days
            
        Returns:
            Dict with recent metrics
        """
        return self.get_accuracy_metrics(days=days)
    
    def export_report(self, filename: str = None) -> str:
        """
        Export comprehensive performance report
        
        Returns:
            Report as string (markdown format)
        """
        report = "# Model Performance Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall metrics
        overall = self.get_accuracy_metrics()
        report += "## Overall Performance\n\n"
        report += f"- Total Predictions: {overall['num_predictions']}\n"
        report += f"- Mean Absolute Error: ${overall['mae']:.2f}\n"
        report += f"- Mean Absolute Percentage Error: {overall['mape']:.2f}%\n"
        report += f"- Direction Accuracy: {overall['direction_accuracy']:.1f}%\n"
        report += f"- Confidence Correlation: {overall['confidence_correlation']:.3f}\n\n"
        
        # Model comparison
        comparison = self.compare_models()
        if len(comparison) > 0:
            report += "## Model Comparison\n\n"
            report += comparison.to_markdown(index=False)
            report += "\n\n"
        
        # Asset performance
        asset_perf = self.get_asset_performance()
        if len(asset_perf) > 0:
            report += "## Performance by Asset\n\n"
            report += asset_perf.to_markdown(index=False)
            report += "\n\n"
        
        # Confidence calibration
        calibration = self.confidence_calibration_analysis()
        if len(calibration) > 0:
            report += "## Confidence Calibration\n\n"
            report += calibration.to_markdown(index=False)
            report += "\n\n"
        
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
        
        return report
    
    def get_dashboard_metrics(self) -> Dict:
        """
        Get key metrics for dashboard display
        
        Returns:
            Dict with dashboard-ready metrics
        """
        overall = self.get_accuracy_metrics()
        recent = self.get_recent_performance(days=7)
        
        comparison = self.compare_models()
        best_model = comparison.iloc[0]['model'] if len(comparison) > 0 else 'N/A'
        
        asset_perf = self.get_asset_performance()
        best_asset = asset_perf.iloc[0]['symbol'] if len(asset_perf) > 0 else 'N/A'
        worst_asset = asset_perf.iloc[-1]['symbol'] if len(asset_perf) > 0 else 'N/A'
        
        return {
            'total_predictions': overall['num_predictions'],
            'overall_mape': overall['mape'],
            'overall_direction_accuracy': overall['direction_accuracy'],
            'recent_mape': recent.get('mape', 0),
            'recent_direction_accuracy': recent.get('direction_accuracy', 0),
            'best_model': best_model,
            'best_asset': best_asset,
            'worst_asset': worst_asset,
            'confidence_correlation': overall['confidence_correlation']
        }


# Simulated performance data for demonstration
def generate_demo_performance_data() -> PerformanceAnalyzer:
    """
    Generate sample performance data for demo
    
    Returns:
        PerformanceAnalyzer with sample data
    """
    analyzer = PerformanceAnalyzer()
    
    np.random.seed(42)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    models = ['technical_transformer', 'early_fusion', 'late_fusion', 'attention_fusion']
    
    # Generate 100 sample predictions
    for i in range(100):
        symbol = np.random.choice(symbols)
        model = np.random.choice(models)
        
        # Actual price
        actual_price = 100 + np.random.randn() * 20
        
        # Predicted price (with some error)
        error_magnitude = np.random.uniform(1, 5)  # 1-5% error
        predicted_price = actual_price * (1 + np.random.randn() * error_magnitude / 100)
        
        # Confidence (inversely related to error)
        confidence = np.clip(1 - error_magnitude / 10, 0.5, 0.95)
        
        # Date
        prediction_date = datetime.now() - timedelta(days=100-i)
        
        analyzer.add_prediction(symbol, predicted_price, actual_price, 
                               prediction_date, model, confidence)
    
    return analyzer
