"""Data loading utilities for stock price data."""

import yfinance as yf
import pandas as pd
import torch
from torch.utils.data import Dataset


def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a specific ticker within a date range.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pandas.DataFrame: DataFrame containing the closing prices
    """
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    data.rename(columns={'Close': ticker}, inplace=True)
    return data


class StockData(Dataset):
    """PyTorch Dataset for stock price data."""
    
    def __init__(self, data, seq_len):
        """
        Initialize the dataset.
        
        Args:
            data (pandas.DataFrame): Stock price data
            seq_len (int): Sequence length for time series
        """
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        seq = self.data.iloc[idx:idx+self.seq_len]
        label = self.data.iloc[idx+self.seq_len, 0]  # Assuming the first column is the target
        return {
            'seq': torch.tensor(seq.values, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
