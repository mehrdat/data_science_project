"""ARIMA model for stock price forecasting."""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

from ..utils.preprocessing import print_space


def check_stationarity(time_series):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        time_series (pandas.Series): Time series data
        
    Returns:
        bool: True if stationary, False otherwise
    """
    # Dickey-Fuller test
    result = adfuller(time_series.dropna())
    print("Checking the stationarity of data... ")
    print(" "*80)

    print('ADF Statistic: {:.4f}'.format(result[0]))
    print('p-value: {:.4f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.4f}'.format(key, value))

    # If p-value is less than 0.05, we reject the null hypothesis and conclude the series is stationary
    if result[1] <= 0.05:
        print("Series is stationary (reject H0)")
        return True
    else:
        print("Series is not stationary (fail to reject H0) so we need to find a d value for differencing.")
        return False


def find_optimal_arima_params(time_series, p_range, d_range, q_range):
    """
    Find optimal ARIMA parameters using grid search.
    
    Args:
        time_series (pandas.Series): Time series data
        p_range (range): Range of p values to try
        d_range (range): Range of d values to try
        q_range (range): Range of q values to try
        
    Returns:
        tuple: Best (p,d,q) parameters
    """
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
                    print(f"Error: {e}")
                    continue

    print(f"Best ARIMA parameters (p,d,q): {best_params} with AIC: {best_aic}")
    return best_params


def fit_and_evaluate_arima(train_data, test_data, ticker, order):
    """
    Fit an ARIMA model and evaluate its performance.
    
    Args:
        train_data (pandas.Series): Training data
        test_data (pandas.Series): Testing data
        ticker (str): Stock ticker symbol
        order (tuple): ARIMA model order (p,d,q)
        
    Returns:
        tuple: model, train_predictions, test_predictions, metrics
    """
    print_space("ARIMA")
    # Fit model
    model = ARIMA(train_data, order=order)
    results = model.fit()

    # Model summary
    print(f"\nARIMA Model Summary for {ticker}:")
    print(results.summary())

    # Predictions (training)
    train_predictions = results.predict(start=train_data.index[0], end=train_data.index[-1])

    # Prediction (test)
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


def analyze_stocks_with_arima(train_data, test_data, ticker):
    """
    Analyze stock data with ARIMA model.
    
    Args:
        train_data (pandas.Series): Training data
        test_data (pandas.Series): Testing data
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: metrics, test_predictions
    """
    all_data = {}
    model_metrics = []

    print(f"\n{'='*50}")
    print(f"Processing {ticker} stock data...")
    print(f"{'='*50}")

    print(f"Training data: {len(train_data)} points, Test data: {len(test_data)} points")

    # Store data
    all_data[ticker] = {
        'train': train_data,
        'test': test_data
    }

    # Check stationarity on training data
    print("\nChecking stationarity of the training time series:")
    is_stationary = check_stationarity(train_data)

    # Find optimal ARIMA parameters using training data
    print("\nFinding optimal ARIMA parameters...")
    # Limiting the search space for efficiency - adjust ranges as needed
    p_range = range(0, 3)
    d_range = range(0, 3) if not is_stationary else [0]
    q_range = range(0, 3)

    best_params = find_optimal_arima_params(train_data, p_range, d_range, q_range)
    all_data[ticker]['params'] = best_params

    # Fit ARIMA model and get predictions for both train and test
    results, train_preds, test_preds, metrics = fit_and_evaluate_arima(
        train_data, test_data, ticker, best_params
    )

    # Add metrics to the comparison dataframe
    model_metrics.append({
        'Ticker': ticker,
        'ARIMA_Model': f"ARIMA{best_params}",
        'Test_R2': metrics['test_r2']
    })

    # Plot original and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Train Data')
    plt.plot(test_data, label='Test Data')
    plt.plot(train_preds, label='Train Predictions', linestyle='--')
    plt.plot(test_preds, label='Test Predictions', linestyle='--', color='red')
    plt.axvline(x=test_data.index[0], color='black', linestyle='-.')
    plt.title(f'ARIMA Model Fit for {ticker} (ARIMA{best_params})')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {'Model': "ARIMA", 'Ticker': ticker, 'MSE': metrics["test_mse"], 'R2': metrics['test_r2']}, test_preds
