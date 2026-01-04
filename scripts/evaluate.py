"""
Model Evaluation and Visualization
Compare model performance and generate publication-quality visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Evaluate and visualize model performance
    """
    
    def __init__(self, results_dir: str = 'results', graphs_dir: str = 'graphs'):
        """
        Args:
            results_dir: Directory with model results
            graphs_dir: Directory to save visualizations
        """
        self.results_dir = results_dir
        self.graphs_dir = graphs_dir
        os.makedirs(graphs_dir, exist_ok=True)
        
    def load_results(self, filename: str) -> dict:
        """Load training results from JSON"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Price prediction metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        # Profit metrics (simulated trading)
        profit = self.calculate_trading_profit(y_true, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy,
            'Profit_Percentage': profit
        }
    
    def calculate_trading_profit(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 initial_capital: float = 10000) -> float:
        """
        Simulate simple trading strategy based on predictions
        
        Strategy: Buy when predicted to go up, sell when predicted to go down
        
        Returns:
            Profit percentage
        """
        capital = initial_capital
        position = 0  # 0: no position, 1: holding
        shares = 0
        
        for i in range(1, len(y_pred)):
            predicted_direction = y_pred[i] > y_pred[i-1]
            
            if predicted_direction and position == 0:
                # Buy signal
                shares = capital / y_true[i]
                position = 1
            elif not predicted_direction and position == 1:
                # Sell signal
                capital = shares * y_true[i]
                position = 0
                shares = 0
        
        # Close any remaining position
        if position == 1:
            capital = shares * y_true[-1]
        
        profit_pct = ((capital - initial_capital) / initial_capital) * 100
        return profit_pct
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str, ticker: str, save: bool = True):
        """
        Plot true vs predicted prices
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            model_name: Name of model
            ticker: Stock ticker
            save: Save figure
        """
        plt.figure(figsize=(14, 6))
        
        # Plot
        plt.plot(y_true, label='Actual Price', linewidth=2, alpha=0.8)
        plt.plot(y_pred, label='Predicted Price', linewidth=2, alpha=0.8)
        
        plt.title(f'{ticker} - {model_name} Price Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f"{self.graphs_dir}/{ticker}_{model_name}_predictions.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        plt.close()
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str, ticker: str, save: bool = True):
        """Plot prediction error distribution"""
        errors = y_true.flatten() - y_pred.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Error', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} - {model_name} Error Analysis', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filename = f"{self.graphs_dir}/{ticker}_{model_name}_errors.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        plt.close()
    
    def compare_models(self, results_df: pd.DataFrame, save: bool = True):
        """
        Compare performance across all models
        
        Args:
            results_df: DataFrame with model results
            save: Save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']
        titles = ['Root Mean Square Error (Lower is Better)',
                 'Mean Absolute Error (Lower is Better)',
                 'Mean Absolute Percentage Error (Lower is Better)',
                 'Directional Accuracy (Higher is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if metric in results_df.columns:
                data = results_df.sort_values(metric, ascending=(metric != 'Directional_Accuracy'))
                
                bars = ax.barh(range(len(data)), data[metric])
                
                # Color bars
                colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(bars)))
                if metric == 'Directional_Accuracy':
                    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels(data['Model'], fontsize=9)
                ax.set_xlabel(metric, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, (val, model) in enumerate(zip(data[metric], data['Model'])):
                    ax.text(val, i, f' {val:.2f}', va='center', fontsize=8)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            filename = f"{self.graphs_dir}/model_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        plt.close()
    
    def plot_training_curves(self, train_losses: list, val_losses: list,
                            model_name: str, ticker: str, save: bool = True):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
        
        plt.title(f'{ticker} - {model_name} Training Curves', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f"{self.graphs_dir}/{ticker}_{model_name}_training_curves.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        plt.close()
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str = None):
        """Generate comprehensive evaluation report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.results_dir}/evaluation_report_{timestamp}.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" STOCK PRICE PREDICTION - MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*80 + "\n\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("BEST MODELS\n")
            f.write("-"*80 + "\n\n")
            
            for metric in ['RMSE', 'MAE', 'MAPE']:
                best = results_df.nsmallest(1, metric).iloc[0]
                f.write(f"Best {metric}: {best['Model']} ({best[metric]:.4f})\n")
            
            best_da = results_df.nlargest(1, 'Directional_Accuracy').iloc[0]
            f.write(f"Best Directional Accuracy: {best_da['Model']} ({best_da['Directional_Accuracy']:.2f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Report saved to {output_path}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    print("Model Evaluation and Visualization Tool")
    print("Run after training models to generate comprehensive analysis")
