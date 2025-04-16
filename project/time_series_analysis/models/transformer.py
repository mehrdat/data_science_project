import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=2, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout), num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout), num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)

        tgt = torch.zeros_like(x)
        tgt = self.pos_encoder(tgt)

        output = self.transformer_decoder(tgt, x)
        output = output[-1, :, :]
        output = self.decoder(output)

        return output

def create_transformer_model(X_train, X_test, y_train, y_test, epochs=50):
    """Create, train, and evaluate a Transformer model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = TransformerModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            predictions.extend(output.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)

    return predictions, actuals, r2, mse

def plot_transformer_predictions(actuals, predictions):
    """Plot actual vs predicted values for Transformer model."""
    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual', color='blue', linewidth=2)
    plt.plot(predictions, label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
