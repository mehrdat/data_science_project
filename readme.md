# 📈 Stock Price Forecasting Using Advanced Time Series Models

![Project Overview](photo.png)

## 🚀 Project Overview

This project demonstrates the forecasting of stock prices for ten leading tech companies using cutting-edge statistical and deep learning models. The goal is to compare the performance of various models and provide insights into the best approaches for stock price prediction.

## 🏢 Companies Analyzed
- **Microsoft Corp. (MSFT)**
- **Nvidia Corp. (NVDA)**
- **Alphabet Inc. (GOOGL)**
- **Amazon.com Inc. (AMZN)**
- **Meta Platforms Inc. (META)**
- **Taiwan Semiconductor Manufacturing Co. Ltd. (TSM)**
- **ASML Holding NV (ASML)**
- **Adobe Inc. (ADBE)**
- **International Business Machines Corp. (IBM)**
- **Arista Networks Inc. (ANET)**

## 🔍 Models Implemented
1. **ARIMA (AutoRegressive Integrated Moving Average)**
2. **LSTM (Long Short-Term Memory)**
3. **BiLSTM (Bidirectional LSTM)**
4. **N-BEATS (Neural Basis Expansion Analysis)** 
5. **DeepAR**
6. **Prophet**
7. **Temporal Convolutional Network (TCN)**

## 📊 Key Features
- **Data Collection:** Automated using Yahoo Finance API.
- **Data Processing:** Includes time-series decomposition, normalization, and handling of missing values.
- **Model Evaluation:** RMSE, MAE, and MAPE metrics used to compare models.
- **Visualizations:** Interactive plots for comparing predicted vs. actual prices.

## 🫠 Tools & Libraries
- Python: `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- Deep Learning: `TensorFlow`, `PyTorch`
- APIs: `yfinance`
- Visualization: `plotly`

## 📁 Repository Structure
```
🗁 src/
    ├── data_preprocessing.py
    ├── model_training.py
    ├── evaluation.py
🗁 datasets/
🗁 models/
🗁 results/
README.md
photo.png
```

## 🌟 Results
Our analysis highlights:
- **ARIMA** excels in short-term predictions but struggles with non-linearity.
- **LSTM and BiLSTM** effectively capture sequential dependencies.
- **DeepAR and TCN** perform exceptionally well in handling volatility.
- **Prophet** offers flexibility with seasonality but may lack precision in trend shifts.
- **N-BEATS** provides interpretable and robust results across datasets.

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.