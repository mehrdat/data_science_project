from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

def print_space(model):
    """Print formatted spacing for model output."""
    print(" " * 80)
    print("-" * 80)
    print(f"... Training data on {model} ...")
    print(" " * 80)

def evaluate_model(y_true, y_pred, scaler, model_name, ticker):
    """Evaluate model performance using MSE and R²."""
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)

    return {'Model': model_name, 'Ticker': ticker, 'MSE': mse, 'R2': r2}

def evaluate_lookback(data, lookback_values, model_type, train_size,
                      prepare_time_series_data, create_knn_model, create_gru_model, create_transformer_model):
    """Evaluate different lookback values for a given model."""
    results = {}
    for lookback in lookback_values:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_time_series_data(data, lookback=lookback, train_size=train_size)

        # Train model
        if model_type == "KNN":
            knn_model = create_knn_model(X_train, y_train, X_test, y_test)
            y_pred = knn_model.predict(X_test)
        elif model_type == "GRU":
            gru_model = create_gru_model((X_train.shape[1], 1))
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            gru_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                          epochs=20, batch_size=32, validation_split=0.1,
                          callbacks=[early_stop], verbose=0)
            y_pred = gru_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()
        elif model_type == "Transformer":
            y_pred, _, _, _ = create_transformer_model(X_train, X_test, y_train, y_test, epochs=20)
            y_pred = y_pred.flatten()
        else:
            raise ValueError("Unsupported model type")

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        results[lookback] = r2

    return results

def plot_model_comparison(ticker, company_name, y_test, knn_preds, gru_preds, arima_preds, xgb_preds, transformer_preds, scaler, test_dates):
    """Plot comparison of different models."""
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    knn_preds_inv = scaler.inverse_transform(knn_preds.reshape(-1, 1)).flatten()
    gru_preds_inv = scaler.inverse_transform(gru_preds.reshape(-1, 1)).flatten()
    xgb_preds_inv = scaler.inverse_transform(xgb_preds.reshape(-1, 1)).flatten()
    transformer_preds_inv = scaler.inverse_transform(transformer_preds.reshape(-1, 1)).flatten()

    min_length = min(len(y_test_inv), len(knn_preds_inv), len(gru_preds_inv), len(arima_preds))

    y_test_inv = y_test_inv[:min_length]
    knn_preds_inv = knn_preds_inv[:min_length]
    gru_preds_inv = gru_preds_inv[:min_length]
    arima_preds_inv = arima_preds[:min_length]
    xgb_preds_inv = xgb_preds_inv[:min_length]
    transformer_preds_inv = transformer_preds_inv[:min_length]

    test_dates = test_dates[:min_length]

    plt.figure(figsize=(14, 8))
    plt.plot(test_dates, y_test_inv, label='Actual', color='blue', linewidth=2)
    plt.plot(test_dates, knn_preds_inv, label='KNN', color='red', linestyle='--')
    plt.plot(test_dates, gru_preds_inv, label='GRU', color='green', linestyle='--')
    plt.plot(test_dates, arima_preds_inv, label='ARIMA', color='purple', linestyle='--')
    plt.plot(test_dates, xgb_preds_inv, label='XGB', color='orange', linestyle='--')
    plt.plot(test_dates, transformer_preds_inv, label='Transformer', color='black', linestyle='--')
    plt.title(f'Actual vs Predicted', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
