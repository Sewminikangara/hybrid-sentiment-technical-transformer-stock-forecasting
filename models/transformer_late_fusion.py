"""
Hybrid Transformer with Late Fusion
Processes technical and sentiment separately, then combines at decision level
"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    """Positional encoding"""
    
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


class LateFusionTransformer(nn.Module):
    """
    Late Fusion Hybrid Transformer
    Separate processing of technical and sentiment, fusion at output
    """
    
    def __init__(self, technical_size: int, sentiment_size: int,
                 d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_length: int = 100):
        """
        Args:
            technical_size: Number of technical features
            sentiment_size: Number of sentiment features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super(LateFusionTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Separate branches for technical and sentiment
        # Technical branch
        self.technical_projection = nn.Linear(technical_size, d_model)
        self.technical_pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        tech_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.technical_transformer = nn.TransformerEncoder(
            tech_encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Sentiment branch
        self.sentiment_projection = nn.Linear(sentiment_size, d_model)
        self.sentiment_pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        sent_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.sentiment_transformer = nn.TransformerEncoder(
            sent_encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Late fusion layers
        # Combine outputs from both branches
        self.fusion_layer = nn.Linear(d_model * 2, 128)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        
        # Final prediction layers
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, technical_features, sentiment_features):
        """
        Forward pass
        
        Args:
            technical_features: [batch, seq_len, technical_size]
            sentiment_features: [batch, seq_len, sentiment_size]
            
        Returns:
            Predictions [batch, 1]
        """
        # Technical branch
        tech_x = self.technical_projection(technical_features) * math.sqrt(self.d_model)
        tech_x = self.technical_pos_encoder(tech_x)
        tech_out = self.technical_transformer(tech_x)
        tech_last = tech_out[:, -1, :]  # [batch, d_model]
        
        # Sentiment branch
        sent_x = self.sentiment_projection(sentiment_features) * math.sqrt(self.d_model)
        sent_x = self.sentiment_pos_encoder(sent_x)
        sent_out = self.sentiment_transformer(sent_x)
        sent_last = sent_out[:, -1, :]  # [batch, d_model]
        
        # Late fusion: Concatenate outputs
        fused = torch.cat([tech_last, sent_last], dim=-1)  # [batch, d_model*2]
        
        # Fusion layers
        fused = self.fusion_layer(fused)
        fused = self.fusion_relu(fused)
        fused = self.fusion_dropout(fused)
        
        # Final prediction
        out = self.fc1(fused)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LateFusionPredictor:
    """
    Trainer and predictor for Late Fusion Hybrid Transformer
    """
    
    def __init__(self, technical_size: int, sentiment_size: int,
                 d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, learning_rate: float = 0.0001,
                 batch_size: int = 32):
        """Initialize Late Fusion Transformer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = LateFusionTransformer(
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
        """Train the model"""
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
                predictions = self.model(batch_tech, batch_sent)
                loss = self.criterion(predictions, batch_y)
                
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
                    self.save_model('models/best_late_fusion_transformer.pt')
            
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
    print("Late Fusion Hybrid Transformer")
    print("Separate processing branches for technical and sentiment")
    print("\nModel Architecture:")
    print("  Technical → Transformer → Representation A")
    print("  Sentiment → Transformer → Representation B")
    print("  [A + B] → Fusion → Prediction")
