import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import seaborn as sns
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings('ignore')

# import data
def download_stock_data(ticker, start_date, end_date):

    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    data.rename(columns={'Close': ticker}, inplace=True)
    return data

# prepare data
def prepare_time_series_data(data, lookback=30, train_size=0.8):

    ticker = data.columns[0]
    scaler = MinMaxScaler(feature_range=(0, 1))

    split_idx = int(len(data) * train_size)


    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    # Scaleed
    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(lookback, len(scaled_train_data)):
        X_train.append(scaled_train_data[i-lookback:i, 0])
        y_train.append(scaled_train_data[i, 0])

    X_test, y_test = [], []
    for i in range(lookback, len(scaled_test_data)):
        X_test.append(scaled_test_data[i-lookback:i, 0])
        y_test.append(scaled_test_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    test_dates = test_data.index[lookback:lookback+len(y_test)]
    #test_dates = test_data.index
    train_data_raw = train_data[ticker]
    test_data_raw = test_data[ticker].iloc[lookback:lookback+len(y_test)]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'test_dates': test_dates,
        'train_data_raw': train_data_raw,
        'test_data_raw': test_data_raw,
        'scaler': scaler
    }



# evaluation
def evaluate_model(y_true, y_pred, scaler=None):

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

def plot_predictions(y_true, y_pred, scaler=None, dates=None, model_name="Model", title=None):

    if scaler:
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_inv, y_pred_inv = y_true, y_pred

    # R2
    r2 = r2_score(y_true_inv, y_pred_inv)

    plt.figure(figsize=(14, 7))
    if dates is not None:
        plt.plot(dates, y_true_inv, label='Actual', color='blue', linewidth=2)
        plt.plot(dates, y_pred_inv, label=f'Predicted ({model_name}, R²: {r2:.4f})',
                 color='red', linestyle='--')
        plt.xlabel('Date')
    else:
        plt.plot(y_true_inv, label='Actual', color='blue', linewidth=2)
        plt.plot(y_pred_inv, label=f'Predicted ({model_name}, R²: {r2:.4f})',
                 color='red', linestyle='--')
        plt.xlabel('Time')

    plt.title(title or f'{model_name}: Actual vs Predicted (R² = {r2:.4f})')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    return r2

def compare_models(ticker, company_name, models_data):

    # plt sizes
    plt.figure(figsize=(14, 8))

    r2_scores = {}
    first_plot = True
    min_length = float('inf')

    # finding minimum length
    for model_name, data in models_data.items():
        y_test = data['y_test']
        y_pred = data['predictions']
        curr_len = min(len(y_test), len(y_pred))
        min_length = min(min_length, curr_len)

    # plot models
    for model_name, data in models_data.items():
        y_test = data['y_test'][:min_length]
        y_pred = data['predictions'][:min_length]
        scaler = data['scaler']
        test_dates = data['test_dates'][:min_length]

        # back to origin
        if scaler:
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_test_inv, y_pred_inv = y_test, y_pred

        # R2
        r2 = r2_score(y_test_inv, y_pred_inv)
        r2_scores[model_name] = r2

        # plot actual
        if first_plot:
            plt.plot(test_dates, y_test_inv, label='Actual', color='blue', linewidth=2)
            first_plot = False

        # plot preds
        plt.plot(test_dates, y_pred_inv, label=f'{model_name} (R²: {r2:.4f})',
                 linestyle='--', alpha=0.8)

    plt.title(f'{company_name} ({ticker}) - Model Performance Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True)
    plt.tight_layout()

    # model comparison
    textstr = 'Model R² Comparison:\n' + '\n'.join([f'{m}: {r2:.4f}' for m, r2 in r2_scores.items()])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.02, 0.02, textstr, fontsize=10, bbox=props)

    plt.show()

    # plot r2 results
    plt.figure(figsize=(10, 6))
    models = list(r2_scores.keys())
    r2_values = list(r2_scores.values())

    bars = plt.bar(models, r2_values, alpha=0.7)
    plt.title(f'R² Score Comparison for {company_name} ({ticker})', fontsize=16)
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim([0, 1])

    # values on top of each plot
    for bar, score in zip(bars, r2_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return r2_scores

###############################################
##################### KNN #####################
###############################################
def add_knn_features(X):

    features = []
    for i in range(len(X)):
        window = X[i]
        lookback_period = len(window)
        window_diff = np.diff(window)

        # basic statistics
        mean_price = np.mean(window)
        std_price = np.std(window)

        trend = window[-1] - window[0]

        momentum = window[-1] - window[-6] if len(window) >= 6 else 0

        rolling_mean = np.mean(window[-5:]) if len(window) >= 5 else mean_price

        sma_period = min(10, lookback_period)
        sma = np.mean(window[-sma_period:]) if lookback_period >= sma_period else mean_price

        # relative strength index
        rsi_period = min(14, lookback_period)
        rsi = 0.5
        if lookback_period > rsi_period:
            delta = np.diff(window[-(rsi_period + 1):])
            gain = delta[delta > 0].sum()
            loss = -delta[delta < 0].sum()
            if loss == 0:
                rs = np.inf
            else:
                rs = gain / loss
            rsi = 1.0 - (1.0 / (1.0 + rs))


        feature_vector = np.array([
            window[-1],
            mean_price,
            std_price,
            trend,
            momentum,
            sma,
            rsi,
            rolling_mean,
            np.mean(np.abs(window_diff))
        ])

        # replace NaN values with zeros
        feature_vector = np.nan_to_num(feature_vector,nan=0.0,posinf=1.0, neginf=0.0) # Handle potential inf from RSI
        features.append(feature_vector)

    return np.array(features)

def train_knn_model(X_train, y_train, X_test=None, y_test=None):

    # adding features
    X_train_features = add_knn_features(X_train)

    # parameter grid for grid search
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        #'p': [1,2],
        #'leaf_size': [10, 30, 50, 100]  # Optimization parameter for KD-tree or ball-tree
    }

    # create grid search
    knn = KNeighborsRegressor()
    cv_folds = min(5, X_train_features.shape[0] // 2) if X_train_features.shape[0] > 1 else 3

    grid_search = GridSearchCV(knn, param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_features, y_train)

    # best model
    best_model = grid_search.best_estimator_
    print(f"Best KNN Parameters: {grid_search.best_params_}")
    print(f"Best R² Score (CV): {grid_search.best_score_:.4f}")

    # evaluate on test data
    test_r2 = None
    if X_test is not None and y_test is not None:
        X_test_features = add_knn_features(X_test)
        test_r2 = best_model.score(X_test_features, y_test)
        print(f"Test R² Score: {test_r2:.4f}")

    return best_model, test_r2

def predict_knn(model, X):

    if model is None:
        raise ValueError("Model not trained")

    X_features = add_knn_features(X)
    return model.predict(X_features)

def score_knn(model, X, y):

    if model is None:
        raise ValueError("Model not trained")

    X_features = add_knn_features(X)
    return model.score(X_features, y)


###############################################
##################### GRU #####################
###############################################

def train_gru_model(X_train, y_train, X_test=None, y_test=None):

    # reshape input for GRU [samples, time steps, features]
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # model
    model = Sequential([
        GRU(80, return_sequences=False, activation="relu",
            input_shape=(X_train.shape[1], 1)),
        Dropout(0.1),
        Dense(1)
    ])

    # compile
    model.compile(optimizer='adam', loss='mean_squared_error')

    # early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)

    # train model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # vvaluate test data
    test_r2 = None
    if X_test is not None and y_test is not None:
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_loss = model.evaluate(X_test_reshaped, y_test, verbose=0)
        y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
        test_r2 = r2_score(y_test, y_pred)
        print(f"GRU Test Loss: {test_loss:.4f}")
        print(f"GRU Test R² Score: {test_r2:.4f}")

    return model, test_r2

def predict_gru(model, X):

    if model is None:
        raise ValueError("Model not trained")

    # Reshape input for GRU [samples, time steps, features]
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    return model.predict(X_reshaped, verbose=0).flatten()


###############################################
################### ARIMA #####################
###############################################

def check_stationarity(time_series):

    # Dickey-Fuller test
    result = adfuller(time_series.dropna())

    print('ADF Statistic: {:.4f}'.format(result[0]))
    print('p-value: {:.4f}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {:.4f}'.format(key, value))

    # If p-value is less than 0.05, we reject the null hypothesis and conclude the series is stationary
    if result[1] <= 0.05:
        print("Series is stationary (reject H0)")
        return True
    else:
        print("Series is not stationary (fail to reject H0) so we need to find a d value for differencing.")
        return False



def find_optimal_arima_params_auto(time_series):

    model = auto_arima(time_series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

    return model.order

def find_optimal_arima_params(time_series, p_range, d_range, q_range):

    best_aic = float('inf')
    best_params = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(time_series, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                except ValueError as e:
                    print(f"Error in ARIMA({p},{d},{q}): {e}")
                    continue

    print(f"Best ARIMA parameters (p,d,q): {best_params} with AIC: {best_aic}")
    return best_params

def fit_and_evaluate_arima(train_data, test_data, ticker, order):

    # fit model
    model = ARIMA(train_data, order=order)
    #model=SARIMAX(train_data, order=order, seasonal_order=(0, 0, 0, 0))
    results = model.fit()

    # summary
    print(f"\nARIMA Model Summary for {ticker}:")
    print(results.summary())

    # pred train
    train_predictions = results.predict(start=train_data.index[0], end=train_data.index[-1])

    # pred test
    test_predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    test_predictions.index = test_data.index

    # metrics for test data
    test_mse = mean_squared_error(test_data, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_data, test_predictions)

    print(f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    return results, train_predictions, test_predictions, {
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

def analyze_stocks_with_arima(train_data, test_data, ticker):

    all_data = {}
    model_metrics = []

    print(f"\n{'='*50}")
    print(f"Processing {ticker} stock data...")
    print(f"{'='*50}")

    print(f"Training data: {len(train_data)} points, Test data: {len(test_data)} points")

    # store data
    all_data[ticker] = {
        'train': train_data,
        'test': test_data
    }

    # check stationarity
    print("\nChecking stationarity of the training time series:")
    is_stationary = check_stationarity(train_data)

    # optimizing ARIMA parameters
    print("\nFinding optimal ARIMA parameters...")


    p_range = range(0, 5)
    d_range = range(0, 3) if not is_stationary else [0]
    q_range = range(0, 5)

    best_params = find_optimal_arima_params(train_data, p_range, d_range, q_range)
    #best_params = find_optimal_arima_params_auto(train_data)
    all_data[ticker]['params'] = best_params

    # fit ARIMA
    results, train_preds, test_preds, metrics = fit_and_evaluate_arima(
        train_data, test_data, ticker, best_params
    )

    # metrics
    model_metrics.append({
        'Ticker': ticker,
        'ARIMA_Model': f"ARIMA{best_params}",
        'Test_R2': metrics['test_r2']
    })

    # plot original and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Train Data')
    plt.plot(test_data, label='Test Data')
    plt.plot(train_preds, label='Train Predictions', linestyle='--')
    plt.plot(test_preds, label='Test Predictions', linestyle='--', color='red')
    plt.axvline(x=test_data.index[0], color='black', linestyle='-.')
    plt.title(f'ARIMA Model Fit for {ticker} (ARIMA{best_params})')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot residuals

    # plt.figure(figsize=(12, 6))
    # residuals = results.resid
    # plt.plot(residuals)
    # plt.axhline(y=0, color='r', linestyle='-')
    # plt.title(f'ARIMA Model Residuals for {ticker}')
    # plt.ylabel('Residual Value')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # metrics dataframe for comparison
    metrics_df = pd.DataFrame(model_metrics)
    print("\nModel Comparison:")
    print(metrics_df)

    return {'Model': "ARIMA", 'Ticker': ticker, 'MSE': metrics["test_mse"], 'R2': metrics['test_r2']}, test_preds


def perform_exploratory_data_analysis(data_df):
    """
    Performs and prints Exploratory Data Analysis (EDA) on the input DataFrame.

    Args:
        data_df (pd.DataFrame): DataFrame with DatetimeIndex and stock closing prices in columns.
    """

    if not isinstance(data_df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame index must be a DatetimeIndex.")

    tickers = data_df.columns.tolist()
    print("\n" + "="*30 + " STARTING EXPLORATORY DATA ANALYSIS " + "="*30)

    # Data Types
    print("\n--- 3.1 Identifying Data Types ---")
    print("Index Type:", type(data_df.index))
    print("Column Data Types:")
    print(data_df.dtypes)
    print("Target Variable(s) (Close Price): Numerical, Continuous")
    print("Index Variable (Date/Time): Ordinal (Time)")

    # Univariate

    for ticker in tickers:
        print(f"\nAnalyzing Ticker: {ticker}")
        series = data_df[ticker].dropna()

        # Summary Statistics
        print("\nSummary Statistics:")
        desc_stats = series.describe()
        print(desc_stats)
        q1 = desc_stats['25%']
        q3 = desc_stats['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        print(f"IQR: {iqr:.4f}")
        print(f"Potential Outlier Thresholds: < {lower_bound:.4f} or > {upper_bound:.4f}")

        # Visualizations
        print(f"Generating plots for {ticker}...")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # Increased figure size
        fig.suptitle(f'Univariate Analysis for {ticker}', fontsize=16)

        # Line Plot
        axes[0].plot(series.index, series, label=f"{ticker} Close")
        axes[0].set_title("Close Price Over Time")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Close Price (USD)")
        axes[0].legend()
        axes[0].grid(True)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Histogram
        axes[1].hist(series, bins=50, color='skyblue', alpha=0.8, edgecolor='black')
        axes[1].set_title("Histogram of Close Prices")
        axes[1].set_xlabel("Close Price")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis='y')

        # Box Plot
        axes[2].boxplot(series, vert=False, patch_artist=True,
                        medianprops={'color': 'red', 'linewidth': 2},
                        boxprops={'facecolor': 'lightblue'})
        axes[2].set_title("Boxplot of Close Prices")
        axes[2].set_xlabel("Close Price")
        axes[2].set_yticks([])
        axes[2].grid(axis='x')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Bivariate
    print("\n--- 3.3 Bivariate Analysis (Autocorrelation) ---")
    if tickers:
        example_ticker = tickers[0]
        print(f"Generating ACF and PACF plots for {example_ticker} (Lags=40)...")
        series_example = data_df[example_ticker].dropna()
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(series_example, lags=40, ax=axes[0], title=f'Autocorrelation Function (ACF) for {example_ticker}')
        axes[0].grid(True)
        plot_pacf(series_example, lags=40, ax=axes[1], title=f'Partial Autocorrelation Function (PACF) for {example_ticker}', method='ywm')
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No tickers found for autocorrelation analysis.")

    # MultiVariate
    print("\n--- 3.4 Multivariate Analysis ---")
    if len(tickers) > 1:
        # Correlation
        print("\nCorrelation Matrix:")
        correlation_matrix = data_df.corr()
        print(correlation_matrix)
        print("Generating Correlation Heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Closing Prices')
        plt.show()


        print("Generating Scatter Plot Matrix...")
        try:
            scatter_matrix(data_df, figsize=(12, 12), diagonal='kde', alpha=0.6)
            plt.suptitle('Scatter Plot Matrix of Closing Prices', y=1.02)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not generate scatter matrix: {e}")
    else:
        print("Skipping multivariate analysis (requires more than one ticker).")

    print("\n" + "="*30 + " FINISHED EXPLORATORY DATA ANALYSIS " + "="*30 + "\n")


def run_analysis(tickers, company_names, start_date, end_date, lookback_values, train_size):

    all_results = {}

    for ticker, company_name in zip(tickers, company_names):
        print(f"\n{'-'*70}")
        print(f"Processing {company_name} ({ticker})...")
        print(f"{'-'*70}")

        data = download_stock_data(ticker, start_date, end_date)
        print(f"Downloaded {len(data)} data points for {ticker}")

        if len(data) < 100:
            print(f"Skipping {ticker}: insufficient data")
            continue

        perform_exploratory_data_analysis(data)

        # KNN
        best_lookback_knn = lookback_values
        knn_data = prepare_time_series_data(data, lookback=best_lookback_knn, train_size=train_size)

        print(f"Training KNN model...")
        knn_model, _ = train_knn_model(knn_data['X_train'], knn_data['y_train'],
                                       knn_data['X_test'], knn_data['y_test'])
        knn_preds = predict_knn(knn_model, knn_data['X_test'])

        knn_r2 = score_knn(knn_model, knn_data['X_test'], knn_data['y_test'])
        print(f"KNN Test R² Score: {knn_r2:.4f}")

        plot_predictions(
            knn_data['y_test'], knn_preds, knn_data['scaler'],
            knn_data['test_dates'], "KNN", f"KNN Model for {ticker}")

        # GRU
        best_lookback_gru = lookback_values
        gru_data = prepare_time_series_data(data, lookback=best_lookback_gru, train_size=train_size)

        print(f"Training GRU model...")
        gru_model, _ = train_gru_model(gru_data['X_train'], gru_data['y_train'],
                                       gru_data['X_test'], gru_data['y_test'])
        gru_preds = predict_gru(gru_model, gru_data['X_test'])

        gru_r2 = plot_predictions(
            gru_data['y_test'], gru_preds, gru_data['scaler'],
            gru_data['test_dates'], "GRU", f"GRU Model for {ticker}")

        # ARIMA
        models_data = {
            'KNN': {
                'predictions': knn_preds,
                'y_test': knn_data['y_test'],
                'scaler': knn_data['scaler'],
                'test_dates': knn_data['test_dates']
            },
            'GRU': {
                'predictions': gru_preds,
                'y_test': gru_data['y_test'],
                'scaler': gru_data['scaler'],
                'test_dates': gru_data['test_dates']
            }
        }

        try:
            train_data_raw = data[ticker].iloc[:int(len(data) * train_size)]
            test_data_raw = data[ticker].iloc[int(len(data) * train_size):]

            arima_metrics, arima_preds = analyze_stocks_with_arima(
                train_data_raw, test_data_raw, ticker
            )

            # relate the models to common dates
            common_dates = knn_data['test_dates'][:len(arima_preds)]

            # KNN
            models_data['KNN']['test_dates'] = common_dates
            models_data['KNN']['y_test'] = knn_data['y_test'][:len(common_dates)]
            models_data['KNN']['predictions'] = knn_preds[:len(common_dates)]

            # GRU
            models_data['GRU']['test_dates'] = common_dates
            models_data['GRU']['y_test'] = gru_data['y_test'][:len(common_dates)]
            models_data['GRU']['predictions'] = gru_preds[:len(common_dates)]

            # ARIMA
            models_data['ARIMA'] = {
                'predictions': arima_preds.loc[common_dates].values,
                'y_test': test_data_raw.loc[common_dates].values,
                'scaler': None,
                'test_dates': common_dates
            }

            print(f"ARIMA model added to comparison (R²: {arima_metrics['R2']:.4f})")

        except Exception as e:
            print(f"Error with ARIMA model: {e}")
            print("Continuing without ARIMA in the comparison.")

        # compare all models
        results = compare_models(ticker, company_name, models_data)
        print("\nModel Comparison Results (R²):")
        for model_name, r2 in results.items():
            print(f"{model_name}: {r2:.4f}")

        all_results[ticker] = results

    if len(all_results) > 1:
        print("\nOverall R² Comparison Across All Stocks:")
        for ticker in all_results:
            print(f"\n{ticker}:")
            for model_name, r2 in all_results[ticker].items():
                print(f"  {model_name}: {r2:.4f}")

    return all_results



if __name__ == "__main__":

    tickers = ['AAPL', 'AMZN','IBM', 'META', 'GOOGL','MSFT']
    company_names = ['Apple Inc.','Amazon.com Inc.', 'IBM','Meta Platforms Inc.','Alphabet Inc.','Microsoft Corp.']

    start_date = '2019-01-01'
    end_date = '2025-01-31'
    #look_back= range(2,90,1) :=> optimum= 6

    end_result = run_analysis(tickers, company_names, start_date, end_date,lookback_values=6, train_size=0.90)

    print(end_result)

    print("\n--- Generating Summary Plots ---")

    #plot_average_r2_scores(end_result)
    #plot_r2_across_tickers(end_result)
    #plot_r2_heatmap(end_result)

