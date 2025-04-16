import yfinance as yf

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]
    data.rename(columns={'Close': ticker}, inplace=True)
    return data
