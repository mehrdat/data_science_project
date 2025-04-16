"""Utilities for model evaluation."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, scaler, model_name, ticker):
    """
    Evaluate model performance.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for data normalization
        model_name (str): Name of the model
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)

    return {'Model': model_name, 'Ticker': ticker, 'MSE': mse, 'R2': r2}


def evaluate_lookback(data, lookback_values, model, train_size=0.8):
    """
    Evaluate different lookback values for a given model.
    
    Args:
        data (pandas.DataFrame): Stock price data
        lookback_values (list): List of lookback values to evaluate
        model (str): Model name
        train_size (float, optional): Proportion of data to use for training. Defaults to 0.8.
        
    Returns:
        dict: Dictionary mapping lookback values to R² scores
    """
    from ..utils.preprocessing import prepare_time_series_data
    from ..models.knn import create_knn_model
    from ..models.gru import create_gru_model
    from ..models.transformer import create_transformer_model
    from tensorflow.keras.callbacks import EarlyStopping
    
    results = {}
    for lookback in lookback_values:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_time_series_data(
            data, lookback=lookback, train_size=train_size
        )

        # Train model
        if model == "KNN":
            knn_model = create_knn_model(X_train, y_train, X_test, y_test)
            y_pred = knn_model.predict(X_test)
        elif model == "GRU":
            gru_model = create_gru_model((X_train.shape[1], 1))
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            gru_model.fit(
                X_train.reshape(X_train.shape[0], X_train.shape[1], 1), 
                y_train,
                epochs=20, 
                batch_size=32, 
                validation_split=0.1,
                callbacks=[early_stop], 
                verbose=0
            )
            y_pred = gru_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).flatten()
        elif model == "Transformer":
            # Assuming create_transformer_model returns predictions
            y_pred, _, _, _ = create_transformer_model(X_train, X_test, y_train, y_test, epochs=20)
            y_pred = y_pred.flatten()  # Ensure y_pred is 1D
        else:
            raise ValueError("Unsupported model type")

        # Evaluate performance
        r2 = r2_score(y_test, y_pred)
        results[lookback] = r2

    return results
