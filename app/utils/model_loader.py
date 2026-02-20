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
            
            # Load checkpoint - use weights_only=False for models saved with numpy objects
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Determine if this is forex/crypto model (has config) or stock model
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                # Forex or crypto model - use configuration from checkpoint
                d_model = 64
                nhead = 4
                num_layers = 2
                hidden_size_lstm = 64
                # These models use 29 technical + 7 sentiment = 36 total
                technical_dim = 29
                sentiment_dim = 7
            else:
                # Stock model - use original architecture
                d_model = 128
                nhead = 8
                num_layers = 3
                hidden_size_lstm = 128
                # Stock models use 35 technical + 7 sentiment = 42 total
                technical_dim = 35
                sentiment_dim = 7
            
            # Initialize model architecture (using correct parameter names)
            if model_key == 'early_fusion':
                model = EarlyFusionTransformer(
                    technical_size=technical_dim,
                    sentiment_size=sentiment_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_layers,
                    dropout=0.1
                )
            elif model_key == 'late_fusion':
                model = LateFusionTransformer(
                    technical_size=technical_dim,
                    sentiment_size=sentiment_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_layers,
                    dropout=0.1
                )
            elif model_key == 'attention_fusion':
                model = AttentionFusionTransformer(
                    technical_size=technical_dim,
                    sentiment_size=sentiment_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_layers,
                    dropout=0.1
                )
            elif model_key == 'lstm':
                # LSTM uses combined input_size - auto-detect from checkpoint weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    weight = checkpoint['model_state_dict']['lstm.weight_ih_l0']
                elif isinstance(checkpoint, dict) and 'lstm.weight_ih_l0' in checkpoint:
                    weight = checkpoint['lstm.weight_ih_l0']
                else:
                    weight = None
                
                if weight is not None:
                    actual_input_size = weight.shape[1]
                    actual_hidden = weight.shape[0] // 4
                else:
                    actual_input_size = technical_dim + sentiment_dim
                    actual_hidden = hidden_size_lstm
                
                model = LSTMModel(
                    input_size=actual_input_size,
                    hidden_size=actual_hidden,
                    num_layers=2,
                    dropout=0.2
                )
            else:
                return None
            
            # Load weights
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Forex model format (with nested dict)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Stock model format (direct state dict)
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
