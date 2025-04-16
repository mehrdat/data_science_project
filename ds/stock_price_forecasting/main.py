"""Main script for stock price forecasting."""

import warnings
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the project root directory to Python path BEFORE any project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

# Now use absolute imports
from ds.stock_price_forecasting.data.loader import download_stock_data
from ds.stock_price_forecasting.utils.preprocessing import prepare_time_series_data
from ds.stock_price_forecasting.utils.evaluation import evaluate_model, evaluate_lookback
from ds.stock_price_forecasting.utils.visualization import plot_model_comparison
from ds.stock_price_forecasting.models.knn import create_knn_model
from ds.stock_price_forecasting.models.gru import create_gru_model
from ds.stock_price_forecasting.models.arima import analyze_stocks_with_arima
from ds.stock_price_forecasting.models.xgboost_model import create_xgboost_model
from ds.stock_price_forecasting.models.transformer import create_transformer_model

# Suppress warnings
warnings.filterwarnings('ignore')


def run_model(model_name: str, data, lookback_values, train_size, ticker):
    """
    Run a specified model and return predictions and metrics.
    
    Args:
        model_name (str): Name of the model to run
        data (pandas.DataFrame): Stock price data
        lookback_values (list): List of lookback values to evaluate
        train_size (float): Proportion of data to use for training
        ticker (str): Stock ticker symbol
        
    Returns:
        tuple: metrics_list, preds, y_test, scaler, test_dates, train_data, test_data
    """
    metrics_list = []
    X_train, X_test, y_train, y_test, scaler = None, None, None, None, None
    test_dates = None
    train_data = None
    test_data = None
    preds = None

    if model_name == "KNN":
        print(f"Evaluating lookback for KNN model...")
        knn_results = evaluate_lookback(data, lookback_values, model="KNN", train_size=train_size)
        best_lookback_knn = max(knn_results, key=knn_results.get)
        print(f"Best lookback for KNN: {best_lookback_knn} (R²: {knn_results[best_lookback_knn]:.4f})")
        X_train, X_test, y_train, y_test, scaler = prepare_time_series_data(data, lookback=best_lookback_knn, train_size=train_size)
        test_start_idx = int(len(data) * train_size)
        test_dates = data.index[test_start_idx:test_start_idx + len(y_test)]
        train_data = data[ticker].iloc[:test_start_idx]
        test_data = data[ticker].iloc[test_start_idx:test_start_idx + len(y_test)]
        knn_model = create_knn_model(X_train, y_train, X_test, y_test)
        preds = knn_model.predict(X_test)
        knn_metrics = evaluate_model(y_test, preds, scaler, 'KNN', ticker)
        metrics_list.append(knn_metrics)

    elif model_name == "Transformer":
        Transformer_results = evaluate_lookback(data, lookback_values, model="Transformer", train_size=train_size)
        best_lookback_Transformer = max(Transformer_results, key=Transformer_results.get)
        print(f"Best lookback for Transformer: {best_lookback_Transformer} "
              f"(R²: {Transformer_results[best_lookback_Transformer]:.4f})")
        X_train, X_test, y_train, y_test, scaler = prepare_time_series_data(
            data, lookback=best_lookback_Transformer, train_size=train_size
        )
        print(f"The shapes... X_test {X_test.shape}, X_train: {X_train.shape}, "
              f"y_train: {y_train.shape}, y_test: {y_test.shape}")
        test_start_idx = int(len(data) * train_size)
        test_dates = data.index[test_start_idx:test_start_idx + len(y_test)]
        train_data = data[ticker].iloc[:test_start_idx]
        test_data = data[ticker].iloc[test_start_idx:test_start_idx + len(y_test)]
        tr_predictions, tr_actuals, tr_r2, tr_mse = create_transformer_model(
            X_train, X_test, y_train, y_test, epochs=100
        )
        preds = tr_predictions
        tr_metrics = evaluate_model(y_test, tr_predictions, scaler, 'Transformer', ticker)
        metrics_list.append(tr_metrics)

    elif model_name == 'xgb':
        xgb_results = evaluate_lookback(data, lookback_values, model="KNN", train_size=train_size)
        best_lookback_xgb = max(xgb_results, key=xgb_results.get)
        X_train, X_test, y_train, y_test, scaler = prepare_time_series_data(
            data, lookback=best_lookback_xgb, train_size=train_size
        )
        test_start_idx = int(len(data) * train_size)
        test_dates = data.index[test_start_idx:test_start_idx + len(y_test)]
        train_data = data[ticker].iloc[:test_start_idx]
        test_data = data[ticker].iloc[test_start_idx:test_start_idx + len(y_test)]
        xgb = create_xgboost_model(X_train, y_train, X_test, y_test)
        preds = xgb.predict(X_test)
        xgb_metrics = evaluate_model(y_test, preds, scaler, 'XGB', ticker)
        metrics_list.append(xgb_metrics)

    elif model_name == 'gru':
        print(f"Evaluating lookback for GRU model...")
        gru_results = evaluate_lookback(data, lookback_values, model="GRU", train_size=train_size)
        best_lookback_gru = max(gru_results, key=gru_results.get)
        print(f"Best lookback for GRU: {best_lookback_gru} (R²: {gru_results[best_lookback_gru]:.4f})")
        X_train, X_test, y_train, y_test, scaler = prepare_time_series_data(
            data, lookback=best_lookback_gru, train_size=train_size
        )
        test_start_idx = int(len(data) * train_size)
        test_dates = data.index[test_start_idx:test_start_idx + len(y_test)]
        train_data = data[ticker].iloc[:test_start_idx]
        test_data = data[ticker].iloc[test_start_idx:test_start_idx + len(y_test)]
        gru_model = create_gru_model((X_train.shape[1], 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        gru_model.fit(
            X_train.reshape(X_train.shape[0], X_train.shape[1], 1), 
            y_train,
            epochs=60, 
            batch_size=32, 
            validation_split=0.1,
            callbacks=[early_stop], 
            verbose=0
        )
        preds = gru_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()
        gru_metrics = evaluate_model(y_test, preds, scaler, 'GRU', ticker)
        metrics_list.append(gru_metrics)

    elif model_name == 'arima':
        print(f"Training ARIMA model for {ticker}...")
        test_start_idx = int(len(data) * train_size)
        test_dates = data.index[test_start_idx:]
        train_data = data[ticker].iloc[:test_start_idx]
        test_data = data[ticker].iloc[test_start_idx:]
        arima_metrics, preds = analyze_stocks_with_arima(train_data, test_data, ticker)
        metrics_list.append(arima_metrics)
        scaler = MinMaxScaler()
        scaler.fit(data.values.reshape(-1, 1))

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return metrics_list, preds, y_test, scaler, test_dates, train_data, test_data


def main():
    """Main function to run the stock price forecasting analysis."""
    # Define variables
    model_names = ["Transformer", "KNN"]  # Add more models as needed
    tickers = ['AAPL', 'AMZN']  # Add more tickers as needed
    company_names = ['Apple Inc.', 'Amazon.com Inc.']  # Add more companies as needed
    start_date = '2021-01-01'
    end_date = '2024-01-31'
    lookback_values = [7]  # Add more lookback values as needed
    train_size = 0.8

    all_metrics = []
    for ticker, company_name in zip(tickers, company_names):
        print(f"\nProcessing {company_name} ({ticker})...")
        data = download_stock_data(ticker, start_date, end_date)
        if len(data) < 100:
            print(f"Skipping {ticker}: insufficient data")
            continue

        all_model_preds = {}
        y_test_all = None
        scaler_all = None
        test_dates_all = None

        for model_name in model_names:
            metrics, preds, y_test, scaler, test_dates, train_data, test_data = run_model(
                model_name, data, lookback_values, train_size, ticker
            )
            all_metrics.extend(metrics)
            if model_name != 'arima':
                all_model_preds[model_name] = preds
                y_test_all = y_test
                scaler_all = scaler
                test_dates_all = test_dates
            else:
                all_model_preds[model_name] = preds.values
                y_test_all = test_data.values
                scaler_all = MinMaxScaler()
                scaler_all.fit(data.values.reshape(-1, 1))
                test_dates_all = test_dates

        plot_model_comparison(ticker, company_name, y_test_all, all_model_preds, scaler_all, test_dates_all)

    metrics_df = pd.DataFrame(all_metrics)
    print("\nModel Performance Metrics:")
    print(metrics_df)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
