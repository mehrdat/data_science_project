from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.optimizers import Nadam

def create_gru_model(input_shape):
    """Create a GRU model."""
    model = Sequential([
        GRU(80, return_sequences=False, activation="relu", input_shape=(input_shape[0], 1)),
        Dropout(0.1),
        Dense(1)
    ])
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
