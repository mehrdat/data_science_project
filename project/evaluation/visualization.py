import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, scaler=None, dates=None, model_name="Model", title=None):
    """Plot actual vs predicted values"""
    if scaler:
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_inv, y_pred_inv = y_true, y_pred

    plt.figure(figsize=(14, 7))
    if dates is not None:
        plt.plot(dates, y_true_inv, label='Actual', color='blue', linewidth=2)
        plt.plot(dates, y_pred_inv, label=f'Predicted ({model_name})', color='red', linestyle='--')
        plt.xlabel('Date')
    else:
        plt.plot(y_true_inv, label='Actual', color='blue', linewidth=2)
        plt.plot(y_pred_inv, label=f'Predicted ({model_name})', color='red', linestyle='--')
        plt.xlabel('Time')

    plt.title(title or f'{model_name}: Actual vs Predicted')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()