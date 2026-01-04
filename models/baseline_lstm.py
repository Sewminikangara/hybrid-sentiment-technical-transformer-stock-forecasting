"""
LSTM Baseline Model for Stock Price Prediction
Deep learning baseline using Long Short-Term Memory networks
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """
    LSTM architecture for time series prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
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
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMPredictor:
    """
    LSTM model trainer and predictor
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 0.001, batch_size: int = 32):
        """
        Initialize LSTM predictor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = 100, verbose: bool = True):
        """
        Train the LSTM model
        
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
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
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
    
    # Initialize LSTM
    predictor = LSTMPredictor(
        input_size=data['n_features'],
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    # Train
    print(f"Training LSTM on {data['X_train'].shape[0]} samples...")
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
    
    print("\nLSTM Baseline Results:")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
