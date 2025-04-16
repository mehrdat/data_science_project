from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def create_knn_model(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_neighbors': range(1, 40),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

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