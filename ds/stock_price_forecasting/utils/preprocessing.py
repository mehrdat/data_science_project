"""Data preprocessing utilities for time series data."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_time_series_data(data, lookback=7, train_size=0.8):
    """
    Prepare time series data for model training and testing.
    
    Args:
        data (pandas.DataFrame): Stock price data
        lookback (int): Number of previous time steps to use as input features
        train_size (float): Proportion of data to use for training
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = data[:int(len(data) * train_size)]
    test_data = data[int(len(data) * train_size):]

    # Scale the data
    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))

    # Create sequences for training and testing
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


def print_space(model):
    """Print a separator with the model name."""
    print(" "*80)
    print("-"*80)
    print(f"... Training data on {model} ...")
    print(" "*80)
