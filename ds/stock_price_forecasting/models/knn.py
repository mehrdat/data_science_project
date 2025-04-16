"""K-Nearest Neighbors model for stock price forecasting."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from ..utils.preprocessing import print_space


def create_knn_model(X_train, y_train, X_test, y_test, n_neighbors=5):
    """
    Create and train a KNN regression model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing targets
        n_neighbors (int, optional): Number of neighbors. Defaults to 5.
        
    Returns:
        sklearn.neighbors.KNeighborsRegressor: Trained KNN model
    """
    print_space("KNN")
    print("Finding the best parameters for the KNN model")
    print(" "*80)

    param_grid = {
        'n_neighbors': range(1, 40),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Create the KNN model
    knn = KNeighborsRegressor()

    # Use GridSearchCV with R² as the scoring metric
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Best model
    best_knn = grid_search.best_estimator_
    print("Best Parameters: ", grid_search.best_params_)
    print("Best R² Score: ", grid_search.best_score_)

    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual', color='blue', linewidth=2)
    plt.plot(best_knn.predict(X_test), label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return best_knn
