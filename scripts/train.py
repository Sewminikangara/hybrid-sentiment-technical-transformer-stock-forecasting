"""
Unified Training Script for All Models
Trains baseline and Transformer models for stock price prediction
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from scripts.data_preprocessing import StockDataPreprocessor

# Import baseline models
from models.baseline_arima import train_arima_baseline
from models.baseline_lstm import LSTMPredictor

# Import Transformer models
from models.transformer_technical import TechnicalTransformerPredictor
from models.transformer_early_fusion import EarlyFusionPredictor
from models.transformer_late_fusion import LateFusionPredictor
from models.transformer_attention_fusion import AttentionFusionPredictor


class ModelTrainer:
    """
    Unified trainer for all models
    """
    
    def __init__(self, data_path: str, results_dir: str = 'results'):
        """
        Args:
            data_path: Path to technical indicators CSV
            results_dir: Directory to save results
        """
        self.data_path = data_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.results = {}
        
    def prepare_data(self, ticker: str, sequence_length: int = 60):
        """Prepare data for a specific stock"""
        print(f"\n{'='*60}")
        print(f"Preparing data for {ticker}")
        print(f"{'='*60}")
        
        preprocessor = StockDataPreprocessor(sequence_length=sequence_length)
        data = preprocessor.prepare_stock_data(self.data_path, ticker=ticker)
        
        print(f"✓ Training samples: {data['X_train'].shape[0]}")
        print(f"✓ Test samples: {data['X_test'].shape[0]}")
        print(f"✓ Features: {data['n_features']}")
        print(f"✓ Sequence length: {data['sequence_length']}")
        
        return data, preprocessor
    
    def train_arima(self, ticker: str, data: dict, preprocessor):
        """Train ARIMA baseline"""
        print(f"\n--- Training ARIMA for {ticker} ---")
        
        # Get original scale prices
        y_train_orig = preprocessor.inverse_transform_predictions(data['y_train'])
        y_test_orig = preprocessor.inverse_transform_predictions(data['y_test'])
        
        # Train ARIMA
        results = train_arima_baseline(
            y_train_orig.flatten(),
            y_test_orig.flatten(),
            order=(5, 1, 0)
        )
        
        self.results[f"{ticker}_ARIMA"] = results['metrics']
        return results
    
    def train_lstm(self, ticker: str, data: dict, epochs: int = 50):
        """Train LSTM baseline"""
        print(f"\n--- Training LSTM for {ticker} ---")
        
        predictor = LSTMPredictor(
            input_size=data['n_features'],
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Split validation from test
        val_size = len(data['X_test']) // 2
        X_val = data['X_test'][:val_size]
        y_val = data['y_test'][:val_size]
        
        # Train
        train_losses, val_losses = predictor.train(
            data['X_train'],
            data['y_train'],
            X_val,
            y_val,
            epochs=epochs,
            verbose=True
        )
        
        # Predict on test set
        predictions = predictor.predict(data['X_test'])
        
        # Evaluate
        metrics = predictor.evaluate(data['y_test'], predictions)
        
        print(f"\nLSTM Results:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        
        self.results[f"{ticker}_LSTM"] = metrics
        
        # Save model
        model_path = f"{self.results_dir}/{ticker}_lstm.pt"
        predictor.save_model(model_path)
        print(f"✓ Model saved to {model_path}")
        
        return {
            'predictor': predictor,
            'metrics': metrics,
            'predictions': predictions,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def train_technical_transformer(self, ticker: str, data: dict, epochs: int = 100):
        """Train technical-only Transformer"""
        print(f"\n--- Training Technical Transformer for {ticker} ---")
        
        predictor = TechnicalTransformerPredictor(
            input_size=data['n_features'],
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # Split validation
        val_size = len(data['X_test']) // 2
        X_val = data['X_test'][:val_size]
        y_val = data['y_test'][:val_size]
        
        # Train
        train_losses, val_losses = predictor.train(
            data['X_train'],
            data['y_train'],
            X_val,
            y_val,
            epochs=epochs,
            verbose=True
        )
        
        # Predict
        predictions = predictor.predict(data['X_test'])
        
        # Evaluate
        metrics = predictor.evaluate(data['y_test'], predictions)
        
        print(f"\nTechnical Transformer Results:")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        
        self.results[f"{ticker}_Technical_Transformer"] = metrics
        
        # Save model
        model_path = f"{self.results_dir}/{ticker}_technical_transformer.pt"
        predictor.save_model(model_path)
        print(f"✓ Model saved to {model_path}")
        
        return {
            'predictor': predictor,
            'metrics': metrics,
            'predictions': predictions,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def save_results(self, filename: str = None):
        """Save all results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.results_dir}/training_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
        
    def create_results_summary(self):
        """Create summary DataFrame of all results"""
        rows = []
        
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by RMSE
        df = df.sort_values('RMSE')
        
        return df


def train_all_models(tickers: list, data_path: str, epochs: int = 50):
    """
    Train all models for multiple stocks
    
    Args:
        tickers: List of stock tickers
        data_path: Path to data file
        epochs: Number of training epochs
    """
    trainer = ModelTrainer(data_path)
    
    for ticker in tickers:
        print(f"\n\n{'#'*60}")
        print(f"# Training models for {ticker}")
        print(f"{'#'*60}")
        
        # Prepare data
        data, preprocessor = trainer.prepare_data(ticker, sequence_length=60)
        
        # Train baseline models
        try:
            trainer.train_arima(ticker, data, preprocessor)
        except Exception as e:
            print(f"  ARIMA training failed: {e}")
        
        trainer.train_lstm(ticker, data, epochs=epochs)
        
        # Train Transformer models
        trainer.train_technical_transformer(ticker, data, epochs=epochs)
    
    # Save results
    trainer.save_results()
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    summary_df = trainer.create_results_summary()
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = f"{trainer.results_dir}/results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to {summary_path}")
    
    return trainer


if __name__ == "__main__":
    # Example: Train on US stocks
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    data_path = 'data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv'
    
    print("="*60)
    print("STOCK PRICE PREDICTION - MODEL TRAINING")
    print("="*60)
    print(f"\nStocks: {', '.join(tickers)}")
    print(f"Data: {data_path}")
    print(f"\nModels:")
    print("  1. ARIMA (baseline)")
    print("  2. LSTM (baseline)")
    print("  3. Technical Transformer")
    print("  4. Early Fusion Transformer (requires sentiment)")
    print("  5. Late Fusion Transformer (requires sentiment)")
    print("  6. Attention Fusion Transformer (requires sentiment)")
    print("\n" + "="*60 + "\n")
    
    # Train all models
    trainer = train_all_models(tickers, data_path, epochs=50)
