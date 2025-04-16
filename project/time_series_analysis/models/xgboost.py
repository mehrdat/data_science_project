from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def create_xgboost_model(X_train, y_train, X_test, y_test):
    """Create and train an XGBoost model using GridSearchCV."""

    params = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.2, 0.5, 0.7, 0.9, 1.0]
    }

    xgb = XGBRegressor()
    xgb_grid = GridSearchCV(xgb, cv=5, param_grid=params, n_jobs=-1, verbose=0, scoring="neg_mean_squared_error")
    model = xgb_grid.fit(X_train, y_train)
    print("Best params for XGB model : ", model.best_params_)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score for XGB model: {r2:.4f}")

    return model, y_pred

def plot_xgboost_predictions(y_test, y_pred):
    """Plot actual vs predicted values for XGBoost model."""
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label="actual", color="blue")
    plt.plot(y_pred, label="predicted", color="red")
    plt.legend()
    plt.show()
