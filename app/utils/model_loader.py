"""
Model Loader for Stock Prediction App
Loads trained PyTorch models
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add models to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.transformer_early_fusion import EarlyFusionTransformer
from models.transformer_late_fusion import LateFusionTransformer
from models.transformer_attention_fusion import AttentionFusionTransformer
from models.baseline_lstm import LSTMModel

class ModelLoader:
    """Load trained models"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.models_path = self.base_path / 'results'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_map = {
            'Early Fusion': 'early_fusion',
            'Late Fusion': 'late_fusion',
            'Attention Fusion': 'attention_fusion',
            'LSTM Baseline': 'lstm'
        }
    
    def load_model(self, stock, model_name, technical_dim=35, sentiment_dim=7):
        """Load a trained model"""
        try:
            model_key = self.model_map.get(model_name, 'early_fusion')
            model_path = self.models_path / f'{stock}_{model_key}.pt'
            
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return None
            
            # Initialize model architecture (using correct parameter names)
            if model_key == 'early_fusion':
                model = EarlyFusionTransformer(
                    technical_size=technical_dim,
                    sentiment_size=sentiment_dim,
                    d_model=128,
                    nhead=8,
                    num_encoder_layers=3,
                    dropout=0.1
                )
            elif model_key == 'late_fusion':
                model = LateFusionTransformer(
                    technical_size=technical_dim,
                    sentiment_size=sentiment_dim,
                    d_model=128,
                    nhead=8,
                    num_encoder_layers=3,
                    dropout=0.1
                )
            elif model_key == 'attention_fusion':
                model = AttentionFusionTransformer(
                    technical_size=technical_dim,
                    sentiment_size=sentiment_dim,
                    d_model=128,
                    nhead=8,
                    num_encoder_layers=3,
                    dropout=0.1
                )
            elif model_key == 'lstm':
                # LSTM uses combined input_size
                model = LSTMModel(
                    input_size=technical_dim + sentiment_dim,
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2
                )
            else:
                return None
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model = model.to(self.device)
            model.eval()
            
            print(f"Successfully loaded model: {model_path.name}")
            return model
            
        except Exception as e:
            print(f"Error loading model {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self, model_name):
        """Get model information"""
        info = {
            'Early Fusion': {
                'description': 'Combines sentiment and technical features at input layer',
                'best_for': 'Short-term predictions',
                'complexity': 'Medium'
            },
            'Late Fusion': {
                'description': 'Processes features separately, combines at decision layer',
                'best_for': 'Balanced predictions',
                'complexity': 'Medium'
            },
            'Attention Fusion': {
                'description': 'Dynamic weighting of sentiment vs technical signals',
                'best_for': 'Volatile markets',
                'complexity': 'High'
            },
            'LSTM Baseline': {
                'description': 'Traditional recurrent network',
                'best_for': 'Long-term trends',
                'complexity': 'Low'
            }
        }
        
        return info.get(model_name, {})
    
    def model_exists(self, stock, model_name):
        """Check if model file exists"""
        model_key = self.model_map.get(model_name, 'early_fusion')
        model_path = self.models_path / f'{stock}_{model_key}.pt'
        return model_path.exists()
