"""
Hybrid Transformer with Attention Fusion
Uses cross-attention mechanism to dynamically fuse technical and sentiment information
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


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism for fusing technical and sentiment features
    """
    
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super(CrossAttentionFusion, self).__init__()
        
        # Multi-head attention for cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key_value):
        """
        Args:
            query: Query features (e.g., technical) [batch, seq_len, d_model]
            key_value: Key/Value features (e.g., sentiment) [batch, seq_len, d_model]
            
        Returns:
            Fused representation [batch, seq_len, d_model]
        """
        # Cross-attention: query attends to key_value
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value
        )
        
        # Residual connection and normalization
        x = self.norm1(query + attn_output)
        
        # Feedforward network
        ffn_output = self.ffn(x)
        
        # Residual connection and normalization
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights


class AttentionFusionTransformer(nn.Module):
    """
    Attention Fusion Hybrid Transformer
    Uses cross-attention to dynamically fuse technical and sentiment
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
        super(AttentionFusionTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Technical encoder branch
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
        
        # Sentiment encoder branch
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
        
        # Cross-attention fusion layers
        # Technical attends to sentiment
        self.tech_to_sent_fusion = CrossAttentionFusion(d_model, nhead, dropout)
        
        # Sentiment attends to technical
        self.sent_to_tech_fusion = CrossAttentionFusion(d_model, nhead, dropout)
        
        # Combine both fusion directions
        self.fusion_combine = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # Final prediction layers
        self.fc1 = nn.Linear(d_model, 64)
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
        # Process technical features
        tech_x = self.technical_projection(technical_features) * math.sqrt(self.d_model)
        tech_x = self.technical_pos_encoder(tech_x)
        tech_encoded = self.technical_transformer(tech_x)  # [batch, seq_len, d_model]
        
        # Process sentiment features
        sent_x = self.sentiment_projection(sentiment_features) * math.sqrt(self.d_model)
        sent_x = self.sentiment_pos_encoder(sent_x)
        sent_encoded = self.sentiment_transformer(sent_x)  # [batch, seq_len, d_model]
        
        # Cross-attention fusion
        # Technical queries sentiment
        tech_fused, tech_attn_weights = self.tech_to_sent_fusion(
            query=tech_encoded,
            key_value=sent_encoded
        )
        
        # Sentiment queries technical
        sent_fused, sent_attn_weights = self.sent_to_tech_fusion(
            query=sent_encoded,
            key_value=tech_encoded
        )
        
        # Take last time step from both fusion directions
        tech_last = tech_fused[:, -1, :]  # [batch, d_model]
        sent_last = sent_fused[:, -1, :]  # [batch, d_model]
        
        # Combine both directions
        combined = torch.cat([tech_last, sent_last], dim=-1)  # [batch, d_model*2]
        fused = self.fusion_combine(combined)  # [batch, d_model]
        fused = self.fusion_norm(fused)
        
        # Final prediction
        out = self.fc1(fused)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class AttentionFusionPredictor:
    """
    Trainer and predictor for Attention Fusion Hybrid Transformer
    """
    
    def __init__(self, technical_size: int, sentiment_size: int,
                 d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, learning_rate: float = 0.0001,
                 batch_size: int = 32):
        """Initialize Attention Fusion Transformer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = AttentionFusionTransformer(
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
                    self.save_model('models/best_attention_fusion_transformer.pt')
            
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
    print("Attention Fusion Hybrid Transformer")
    print("Cross-attention mechanism for dynamic fusion")
    print("\nModel Architecture:")
    print("  Technical → Transformer → Encoded Technical")
    print("  Sentiment → Transformer → Encoded Sentiment")
    print("  Cross-Attention:")
    print("    - Technical queries Sentiment (what sentiment is relevant?)")
    print("    - Sentiment queries Technical (what technical patterns matter?)")
    print("  → Fusion → Prediction")
