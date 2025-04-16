"""Transformer model for stock price forecasting."""

import sys
import warnings

# Check NumPy version before importing
try:
    import numpy as np
    if np.__version__.startswith('2.'):
        warnings.warn(
            "NumPy 2.x detected. TensorFlow requires NumPy < 2.0. "
            "Please downgrade NumPy using: pip install numpy<2.0",
            RuntimeWarning
        )
        # Optionally exit to prevent further errors
        # sys.exit(1)
except ImportError:
    print("NumPy not found. Please install numpy<2.0")
    sys.exit(1)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ..utils.preprocessing import print_space


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of the model
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass."""
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_dim, model_dim=64, num_heads=2, num_layers=3, dropout=0.1):
        """
        Initialize transformer model.
        
        Args:
            input_dim (int): Input dimension
            model_dim (int, optional): Model dimension. Defaults to 64.
            num_heads (int, optional): Number of attention heads. Defaults to 2.
            num_layers (int, optional): Number of transformer layers. Defaults to 3.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, x):
        """Forward pass."""
        x = self.encoder(x)  # (batch_size, model_dim)
        x = x.unsqueeze(1)  # Add a sequence length dimension: (batch_size, 1, model_dim)
        x = x.permute(1, 0, 2)  # Permute to (seq_len, batch_size, model_dim) -> (1, batch_size, model_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Mean over the sequence length
        x = self.decoder(x)
        return x


def create_transformer_model(X_train, X_test, y_train, y_test, epochs=70):
    """
    Create and train a transformer model.
    
    Args:
        X_train (numpy.ndarray): Training features
        X_test (numpy.ndarray): Testing features
        y_train (numpy.ndarray): Training targets
        y_test (numpy.ndarray): Testing targets
        epochs (int, optional): Number of training epochs. Defaults to 70.
        
    Returns:
        tuple: predictions, actuals, r2, mse
    """
    print_space("Transformer")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_train, X_test = torch.Tensor(X_train_scaled), torch.Tensor(X_test_scaled)
    y_train, y_test = torch.Tensor(y_train_scaled), torch.Tensor(y_test_scaled)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Validation set
    val_size = int(0.1 * len(train_data))
    train_data, val_data = torch.utils.data.random_split(train_data, [len(train_data) - val_size, val_size])
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    print(f"Transformer model ...{X_train.shape[1]}")

    model = TransformerModel(input_dim=X_train.shape[1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output.squeeze(), target).item()
        val_loss /= len(val_loader)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, "
                  f"Validation Loss: {val_loss:.4f}")

    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            predictions.extend(output.cpu().numpy().squeeze())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual', color='blue', linewidth=2)
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)

    return predictions, actuals, r2, mse
