"""Utilities for data visualization."""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_model_comparison(ticker, company_name, y_test, model_preds, scaler, test_dates):
    """
    Plot comparison of actual vs predicted values for multiple models.
    
    Args:
        ticker (str): Stock ticker symbol
        company_name (str): Company name
        y_test (numpy.ndarray): True values
        model_preds (dict): Dictionary mapping model names to predictions
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler used for data normalization
        test_dates (pandas.DatetimeIndex): Dates for the test data
    """
    plt.figure(figsize=(14, 8))
    plt.plot(test_dates, scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), 
             label='Actual', color='blue', linewidth=2)

    for model_name, preds in model_preds.items():
        preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        preds = preds.to_numpy() if isinstance(preds, pd.Series) else preds
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)  
        
        preds_inv = scaler.inverse_transform(preds).flatten()
        plt.plot(test_dates[:len(preds_inv)], preds_inv, label=model_name, linestyle='--')

    plt.title(f'{company_name} ({ticker}) - Actual vs Predicted', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
