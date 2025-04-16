"""XGBoost model for stock price forecasting."""

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from ..utils.preprocessing import print_space


def create_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Create and train an XGBoost regression model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing targets
        
    Returns:
        xgboost.XGBRegressor: Trained XGBoost model
    """
    print_space("XGBoost")
    print("Finding the best parameters for the XGB model")
    print(" "*80)

    params = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.2, 0.5, 0.7, 0.9, 1.0]
    }

    xgb = XGBRegressor()
    xgb_grid = GridSearchCV(xgb, cv=5, param_grid=params, n_jobs=-1, verbose=2, scoring="neg_mean_squared_error")
    model = xgb_grid.fit(X_train, y_train)
    print("Best params for XGB model: ", model.best_params_)

    # Compute R² Score
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score for XGB model: {r2:.4f}")
    print(" "*80)
    print("XGB model training completed!")
    print(" "*80)

    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label="actual", color="blue")
    plt.plot(y_pred, label="predicted", color="red")
    plt.legend()
    plt.show()

    return model
