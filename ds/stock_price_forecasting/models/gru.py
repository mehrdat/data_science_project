"""GRU model for stock price forecasting."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Nadam

from ..utils.preprocessing import print_space


def create_gru_model(input_shape):
    """
    Create a GRU neural network model.
    
    Args:
        input_shape (tuple): Shape of the input data
        
    Returns:
        tensorflow.keras.models.Sequential: Compiled GRU model
    """
    print_space("GRU")
    model = Sequential([
        GRU(80, return_sequences=False, activation="relu", input_shape=(input_shape[0], 1)),
        Dropout(0.1),
        Dense(1)
    ])
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
