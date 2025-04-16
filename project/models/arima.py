from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def check_stationarity(time_series):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(time_series.dropna())
    print('ADF Statistic: {:.4f}'.format(result[0]))
    print('p-value: {:.4f}'.format(result[1]))
    return result[1] <= 0.05

def find_optimal_arima_params_auto(time_series):
    model = auto_arima(time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    return model.order

def fit_and_evaluate_arima(train_data, test_data, ticker, order):
    model = ARIMA(train_data, order=order)
    results = model.fit()
    print(results.summary())

    train_predictions = results.predict(start=train_data.index[0], end=train_data.index[-1])
    test_predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    test_predictions.index = test_data.index

    test_mse = mean_squared_error(test_data, test_predictions)
    test_r2 = r2_score(test_data, test_predictions)

    print(f"Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

    return results, train_predictions, test_predictions, {'test_mse': test_mse, 'test_r2': test_r2}