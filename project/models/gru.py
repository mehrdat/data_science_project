from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_gru_model(input_shape):
    model = Sequential([
        GRU(80, return_sequences=False, activation="relu", input_shape=(input_shape[0], 1)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model