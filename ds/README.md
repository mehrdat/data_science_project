# Stock Price Forecasting

A data science project for financial stock price forecasting using various machine learning models.

## Models Implemented

- K-Nearest Neighbors (KNN)
- Gated Recurrent Unit (GRU)
- Auto-Regressive Integrated Moving Average (ARIMA)
- Transformer
- XGBoost

## Requirements

- Python >= 3.11
- Dependencies listed in `pyproject.toml`

## Installation

```bash
cd /path/to/project
poetry install
```

## Usage

```bash
poetry run python -m stock_price_forecasting
```

## Project Structure

```
stock-price-forecasting/
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── stock_price_forecasting/
│   ├── __init__.py        # Package initialization
│   ├── main.py            # Main application entry point
│   ├── data/
│   │   └── loader.py      # Data loading utilities
│   ├── models/
│   │   ├── arima.py       # ARIMA model implementation
│   │   ├── gru.py         # GRU model implementation
│   │   ├── knn.py         # KNN model implementation
│   │   ├── transformer.py # Transformer model implementation
│   │   └── xgboost_model.py # XGBoost model implementation
│   └── utils/
│       ├── preprocessing.py # Data preprocessing utilities
│       ├── evaluation.py    # Model evaluation utilities
│       └── visualization.py # Visualization utilities
└── tests/                 # Test directory
    └── __init__.py
```
