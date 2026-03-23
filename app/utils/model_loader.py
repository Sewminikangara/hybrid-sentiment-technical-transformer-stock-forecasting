
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
        self.results_path = self.base_path / 'results'
        self.best_models_path = self.base_path / 'models'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_map = {
            'Early Fusion': 'early_fusion',
            'Late Fusion': 'late_fusion',
            'Attention Fusion': 'attention_fusion',
            'LSTM Baseline': 'lstm',
            'Technical Only': 'technical'
        }
    
    def load_model(self, stock, model_name, technical_dim=43, sentiment_dim=7):
        """Load a trained model with fallback logic"""
        model_path = None
        try:
            model_key = self.model_map.get(model_name, 'early_fusion')
            
            # Priority 1: Specific model for this stock in results/
            stock_model_path = self.results_path / f'{stock}_{model_key}.pt'
            
            # Priority 2: Generic best model in models/
            generic_model_path = self.best_models_path / f'best_{model_key}_transformer.pt'
            if model_key == 'lstm':
                generic_model_path = self.best_models_path / 'baseline_lstm.py' # Just for reference, usually .pt
                # Re-adjust for actual LSTM weights if named differently
                lstm_weights = self.best_models_path / 'best_lstm.pt'
                if lstm_weights.exists(): generic_model_path = lstm_weights
            
            if stock_model_path.exists():
                model_path = stock_model_path
                print(f"Loading stock-specific model for {stock}: {model_path.name}")
            elif generic_model_path.exists():
                model_path = generic_model_path
                print(f"Stock-specific model not found for {stock}. Falling back to generic: {model_path.name}")
            else:
                # Last resort: try just {model_key}.pt in results if it exists (some old naming)
                last_resort = self.results_path / f'{model_key}.pt'
                if last_resort.exists():
                    model_path = last_resort
                    print(f"Using legacy model file: {model_path.name}")
            
            if not model_path or not model_path.exists():
                print(f"No suitable model file found for {stock} {model_name}")
                return None
            
            # Load checkpoint - use weights_only=False for models saved with numpy objects
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract state dict if nested
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            
            # Determine if this is forex/crypto model (has config) or stock model
            is_forex_crypto = isinstance(checkpoint, dict) and 'config' in checkpoint
            
            if is_forex_crypto:
                d_model = 64
                nhead = 4
                num_layers = 2
                hidden_size_lstm = 64
            else:
                d_model = 128
                nhead = 8
                num_layers = 3
                hidden_size_lstm = 128

            # Robust dimensionality detection from checkpoint weights
            if 'input_projection.weight' in state_dict: # Early Fusion
                combined_size = state_dict['input_projection.weight'].shape[1]
                # Try to preserve sentiment_dim if it seems plausible, otherwise use heuristics
                if combined_size == 50: # Stock hybrid
                    technical_dim, sentiment_dim = 43, 7
                elif combined_size == 42: # Old stock hybrid
                    technical_dim, sentiment_dim = 35, 7
                elif combined_size == 40: # Forex/Crypto hybrid
                    technical_dim, sentiment_dim = 36, 4
                else:
                    # Fallback: assume sentiment_dim is 7 or 4
                    sentiment_dim = 7 if combined_size > 40 else 4
                    technical_dim = combined_size - sentiment_dim
            elif 'technical_projection.weight' in state_dict: # Late/Attention Fusion
                technical_dim = state_dict['technical_projection.weight'].shape[1]
                sentiment_dim = state_dict['sentiment_projection.weight'].shape[1]
            elif 'lstm.weight_ih_l0' in state_dict: # LSTM
                combined_size = state_dict['lstm.weight_ih_l0'].shape[1]
                hidden_size_lstm = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                # Use heuristics for dims if needed for consistent UI, though LSTM only cares about combined_size
                if combined_size == 50: technical_dim, sentiment_dim = 43, 7
                elif combined_size == 40: technical_dim, sentiment_dim = 36, 4
                else:
                    sentiment_dim = 7 if combined_size > 40 else 4
                    technical_dim = combined_size - sentiment_dim
            
            # Initialize model architecture
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
                model = LSTMModel(
                    input_size=technical_dim + sentiment_dim,
                    hidden_size=hidden_size_lstm,
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
        """Check if any version of the model exists"""
        model_key = self.model_map.get(model_name, 'early_fusion')
        stock_path = self.results_path / f'{stock}_{model_key}.pt'
        generic_path = self.best_models_path / f'best_{model_key}_transformer.pt'
        return stock_path.exists() or generic_path.exists()
