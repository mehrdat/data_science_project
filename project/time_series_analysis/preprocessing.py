import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_data(data):
    """Scale the time series data to a range of [0, 1]."""
    min_val = data.min()
    max_val = data.max()
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def create_sequences(data, sequence_length):
    """Create sequences of data for time series analysis."""
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return sequences

def prepare_time_series_data(data, lookback=7, train_size=0.8):
    """Prepare time series data for model training and testing."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = data[:int(len(data) * train_size)]
    test_data = data[int(len(data) * train_size):]

    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(lookback, len(scaled_train_data)):
        X_train.append(scaled_train_data[i-lookback:i, 0])
        y_train.append(scaled_train_data[i, 0])

    X_test, y_test = [], []
    for i in range(lookback, len(scaled_test_data)):
        X_test.append(scaled_test_data[i-lookback:i, 0])
        y_test.append(scaled_test_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, X_test, y_train, y_test, scaler
