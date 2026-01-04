"""
Hybrid Transformer with Early Fusion
Combines technical indicators and sentiment data at the input level
"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EarlyFusionTransformer(nn.Module):
    """
    Early Fusion Hybrid Transformer
    Concatenates technical and sentiment features at input level
    """
    
    def __init__(self, technical_size: int, sentiment_size: int,
                 d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_length: int = 100):
        """
        Args:
            technical_size: Number of technical indicator features
            sentiment_size: Number of sentiment features
            d_model: Dimension of model
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(EarlyFusionTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Combine technical and sentiment at input
        combined_size = technical_size + sentiment_size
        
        # Input projection
        self.input_projection = nn.Linear(combined_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
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
        
    def forward(self, technical_features, sentiment_features):
        """
        Forward pass
        
        Args:
            technical_features: Technical indicators [batch, seq_len, technical_size]
            sentiment_features: Sentiment features [batch, seq_len, sentiment_size]
            
        Returns:
            Predictions [batch, 1]
        """
        # Early fusion: Concatenate at input level
        x = torch.cat([technical_features, sentiment_features], dim=-1)
        
        # Project to d_model
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Take last time step
        last_output = transformer_out[:, -1, :]
        
        # Output layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class EarlyFusionPredictor:
    """
    Trainer and predictor for Early Fusion Hybrid Transformer
    """
    
    def __init__(self, technical_size: int, sentiment_size: int,
                 d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, learning_rate: float = 0.0001,
                 batch_size: int = 32):
        """
        Initialize Early Fusion Transformer predictor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = EarlyFusionTransformer(
            technical_size=technical_size,
            sentiment_size=sentiment_size,
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
        
    def train(self, X_technical_train: np.ndarray, X_sentiment_train: np.ndarray,
             y_train: np.ndarray, X_technical_val: np.ndarray = None,
             X_sentiment_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = 100, verbose: bool = True):
        """
        Train the model
        
        Args:
            X_technical_train: Technical features [samples, seq_len, technical_features]
            X_sentiment_train: Sentiment features [samples, seq_len, sentiment_features]
            y_train: Target values [samples, 1]
            X_technical_val: Validation technical features
            X_sentiment_val: Validation sentiment features
            y_val: Validation targets
            epochs: Number of epochs
            verbose: Print progress
        """
        # Convert to tensors
        X_tech_tensor = torch.FloatTensor(X_technical_train).to(self.device)
        X_sent_tensor = torch.FloatTensor(X_sentiment_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_tech_tensor, X_sent_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data
        if X_technical_val is not None:
            X_tech_val_tensor = torch.FloatTensor(X_technical_val).to(self.device)
            X_sent_val_tensor = torch.FloatTensor(X_sentiment_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_tech, batch_sent, batch_y in train_loader:
                # Forward pass
                predictions = self.model(batch_tech, batch_sent)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if X_technical_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_tech_val_tensor, X_sent_val_tensor)
                    val_loss = self.criterion(val_predictions, y_val_tensor).item()
                    val_losses.append(val_loss)
                    
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('models/best_early_fusion_transformer.pt')
            
            if verbose and (epoch + 1) % 10 == 0:
                if X_technical_val is not None:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")
        
        return train_losses, val_losses
    
    def predict(self, X_technical: np.ndarray, X_sentiment: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        
        X_tech_tensor = torch.FloatTensor(X_technical).to(self.device)
        X_sent_tensor = torch.FloatTensor(X_sentiment).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tech_tensor, X_sent_tensor)
            
        return predictions.cpu().numpy()
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate performance"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
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
        """Save model"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        """Load model"""
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    print("Early Fusion Hybrid Transformer")
    print("Combines technical and sentiment features at input level")
    print("\nModel Architecture:")
    print("  Input: Technical + Sentiment (concatenated)")
    print("  → Shared Transformer Encoder")
    print("  → Prediction Head")
