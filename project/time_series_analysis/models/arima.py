from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
#from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def check_stationarity(time_series):
    """Check if a time series is stationary using the Dickey-Fuller test."""
    result = adfuller(time_series.dropna())
    print("Checking the stationarity of data... ")

    print('ADF Statistic: {:.4f}'.format(result[0]))
    print('p-value: {:.4f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.4f}'.format(key, value))

    if result[1] <= 0.05:
        print("Series is stationary (reject H0)")
        return True
    else:
        print("Series is not stationary (fail to reject H0) so we need to find a d value fofr differencing.")
        return False

# def find_optimal_arima_params_auto(time_series):
#     """Find optimal ARIMA parameters automatically using auto_arima."""
#     print("Finding the best parameters for the ARIMA model...")
#     model = auto_arima(time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
#     return model.order

def find_optimal_arima_params(time_series, p_range, d_range, q_range):
    """Find optimal ARIMA parameters manually."""
    best_aic = float('inf')
    best_params = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(time_series, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                except ValueError as e:
                    print(f"Mehrdad there is an error : {e}")
                    continue

    print(f"Best ARIMA parameters (p,d,q): {best_params} with AIC: {best_aic}")
    return best_params

def fit_and_evaluate_arima(train_data, test_data, ticker, order):
    """Fit ARIMA model, make predictions, and evaluate performance."""
    # fit model
    model = ARIMA(train_data, order=order)
    results = model.fit()

    # model summary
    print(f"\nARIMA Model Summary for {ticker}:")
    print(results.summary())

    # predictions(training)
    train_predictions = results.predict(start=train_data.index[0], end=train_data.index[-1])

    # prediction (test)
    test_predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    test_predictions.index = test_data.index

    # Calculate metrics for test data
    test_mse = mean_squared_error(test_data, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_data, test_predictions)

    print(f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    return results, train_predictions, test_predictions, {
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

def plot_arima_results(train_data, test_data, train_preds, test_preds, ticker):
    """Plot original data and ARIMA predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Train Data')
    plt.plot(test_data, label='Test Data')
    plt.plot(train_preds, label='Train Predictions', linestyle='--')
    plt.plot(test_preds, label='Test Predictions', linestyle='--', color='red')
    plt.axvline(x=test_data.index[0], color='black', linestyle='-.')
    plt.title(f'ARIMA Model Fit for {ticker}')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_arima_residuals(results, ticker):
    """Plot ARIMA model residuals."""
    plt.figure(figsize=(12, 6))
    residuals = results.resid
    plt.plot(residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'ARIMA Model Residuals for {ticker}')
    plt.ylabel('Residual Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
