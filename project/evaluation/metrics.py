from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, scaler=None):
    """Evaluate model performance"""
    if scaler:
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_inv, y_pred_inv = y_true, y_pred

    mse = mean_squared_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)

    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2
    }