import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_time_series_data(data, lookback=30, train_size=0.8):
    """Prepare time series data with lookback window"""
    ticker = data.columns[0]
    scaler = MinMaxScaler(feature_range=(0, 1))

    split_idx = int(len(data) * train_size)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

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

    test_dates = test_data.index[lookback:lookback+len(y_test)]
    train_data_raw = train_data[ticker]
    test_data_raw = test_data[ticker].iloc[lookback:lookback+len(y_test)]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'test_dates': test_dates,
        'train_data_raw': train_data_raw,
        'test_data_raw': test_data_raw,
        'scaler': scaler
    }