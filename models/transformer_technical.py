"""
Technical-Only Transformer Model
Transformer architecture using only technical indicators (baseline for hybrid comparison)
"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TechnicalTransformer(nn.Module):
    """
    Transformer model for stock price prediction using technical indicators
    """
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_length: int = 100):
        """
        Args:
            input_size: Number of input features (technical indicators)
            d_model: Dimension of model (must be divisible by nhead)
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TechnicalTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding (project features to d_model dimension)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, sequence_length, features]
            
        Returns:
            Predictions [batch, 1]
        """
        # Project input to d_model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Take last time step
        last_output = transformer_out[:, -1, :]
        
        # Final prediction layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TechnicalTransformerPredictor:
    """
    Trainer and predictor for technical-only Transformer
    """
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, learning_rate: float = 0.0001,
                 batch_size: int = 32):
        """
        Initialize Transformer predictor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = TechnicalTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.batch_size = batch_size
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = 100, verbose: bool = True):
        """
        Train the Transformer model
        
        Args:
            X_train: Training features [samples, sequence_length, features]
            y_train: Training targets [samples, 1]
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            verbose: Print training progress
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.criterion(val_predictions, y_val_tensor).item()
                    val_losses.append(val_loss)
                    
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('models/best_technical_transformer.pt')
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if X_val is not None:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")
        
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features [samples, sequence_length, features]
            
        Returns:
            Predictions [samples, 1]
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluate model performance
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    from scripts.data_preprocessing import StockDataPreprocessor
    
    # Prepare data
    preprocessor = StockDataPreprocessor(sequence_length=60)
    data = preprocessor.prepare_stock_data(
        'data_processed/technical/technical_indicators_all_stocks_20251218_061714.csv',
        ticker='AAPL'
    )
    
    # Initialize Transformer
    predictor = TechnicalTransformerPredictor(
        input_size=data['n_features'],
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Train
    print(f"\nTraining Technical Transformer on {data['X_train'].shape[0]} samples...")
    print(f"Features: {data['n_features']}, Sequence Length: {data['sequence_length']}")
    
    train_losses, val_losses = predictor.train(
        data['X_train'],
        data['y_train'],
        data['X_test'][:len(data['X_test'])//2],
        data['y_test'][:len(data['y_test'])//2],
        epochs=50,
        verbose=True
    )
    
    # Predict
    predictions = predictor.predict(data['X_test'])
    
    # Evaluate
    metrics = predictor.evaluate(data['y_test'], predictions)
    
    print("\nTechnical Transformer Results:")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
