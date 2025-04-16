import pandas as pd
from models.arima import (check_stationarity, find_optimal_arima_params,
                           fit_and_evaluate_arima, plot_arima_results,
                           plot_arima_residuals)
import yfinance as yf
import matplotlib.pyplot as plt

def main():
    # Define the ticker symbol
    ticker = "AAPL"  # Example: Apple Inc.

    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

    # Use only the 'Close' prices
    time_series = data['Close']

    # Split data into training and testing sets
    train_size = int(len(time_series) * 0.8)
    train_data, test_data = time_series[:train_size], time_series[train_size:]

    # Check stationarity
    is_stationary = check_stationarity(train_data)

    # If not stationary, differencing might be needed (example: d=1)
    # For simplicity, let's assume d=0 or the series is already stationary
    d = 0

    # Define the range of p and q values to explore
    p_range = range(0, 3)
    q_range = range(0, 3)

    # Find optimal ARIMA parameters
    optimal_params = find_optimal_arima_params(train_data, p_range, [d], q_range)

    print(f"Optimal parameters found: {optimal_params}")  # Debugging print

    # Fit and evaluate ARIMA model
    if optimal_params:
        results, train_predictions, test_predictions, metrics = fit_and_evaluate_arima(
            train_data, test_data, ticker, optimal_params
        )

        # Plot results
        plot_arima_results(train_data, test_data, train_predictions, test_predictions, ticker)

        # Plot residuals
        plot_arima_residuals(results, ticker)
    else:
        print("No optimal parameters found.")

if __name__ == "__main__":
    main()
