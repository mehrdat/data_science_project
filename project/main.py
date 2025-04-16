from data.downloader import download_stock_data
from data.preprocessing import prepare_time_series_data
from models.knn import create_knn_model
from models.gru import create_gru_model
from models.arima import check_stationarity, find_optimal_arima_params_auto, fit_and_evaluate_arima
from evaluation.metrics import evaluate_model
from evaluation.visualization import plot_predictions

# Example usage
def main():
    ticker = 'AAPL'
    start_date = '2019-01-01'
    end_date = '2025-01-31'

    # Download data
    data = download_stock_data(ticker, start_date, end_date)

    # Prepare data
    prepared_data = prepare_time_series_data(data, lookback=30, train_size=0.8)
    X_train, X_test = prepared_data['X_train'], prepared_data['X_test']
    y_train, y_test = prepared_data['y_train'], prepared_data['y_test']
    scaler = prepared_data['scaler']
    test_dates = prepared_data['test_dates']

    # Train and evaluate KNN
    knn_model = create_knn_model(X_train, y_train, X_test, y_test)
    knn_preds = knn_model.predict(X_test)
    knn_metrics = evaluate_model(y_test, knn_preds, scaler)
    print("KNN Metrics:", knn_metrics)
    plot_predictions(y_test, knn_preds, scaler, test_dates, model_name="KNN")

    # Train and evaluate GRU
    gru_model = create_gru_model((X_train.shape[1], 1))
    gru_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
    gru_preds = gru_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()
    gru_metrics = evaluate_model(y_test, gru_preds, scaler)
    print("GRU Metrics:", gru_metrics)
    plot_predictions(y_test, gru_preds, scaler, test_dates, model_name="GRU")

    # Train and evaluate ARIMA
    arima_order = find_optimal_arima_params_auto(data[ticker])
    arima_results, _, arima_preds, arima_metrics = fit_and_evaluate_arima(
        data[ticker].iloc[:int(len(data) * 0.8)],
        data[ticker].iloc[int(len(data) * 0.8):],
        ticker,
        arima_order
    )
    print("ARIMA Metrics:", arima_metrics)
    plot_predictions(y_test, arima_preds, scaler, test_dates, model_name="ARIMA")

if __name__ == "__main__":
    main()