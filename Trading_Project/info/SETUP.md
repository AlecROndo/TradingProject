# Trading Project — Setup Guide

A Bitcoin price prediction and trading system that uses machine learning to predict BTC derivative prices on the [Kalshi](https://kalshi.com) prediction market.

---

## Prerequisites

- Python 3.9+
- A Kalshi Elections API key and RSA private key

---

## 1. Install Dependencies

pip install numpy==2.4.4 pandas==2.3.1 xarray==2026.2.0 requests==2.32.4 scikit-learn==1.7.1 xgboost==3.2.0 lightgbm==4.6.0 matplotlib==3.10.3 cryptography==45.0.6 torch==2.11.0


---

## 2. Configure API Credentials

The project authenticates to the Kalshi API using RSA-PSS signing.

> The Coinbase API (used for BTC price data) is public and requires no credentials.
> The Kalshi REST API is also availible to the public.
---

## 3. Collect Data

**Market data** (Kalshi candlesticks for each BTC event):

python data_dashboard/dataMarketCollect.py

Saves `.nc` files to `new_data/`.

**Bitcoin price data** (Coinbase 1-minute OHLCV):

Saves `new_btc_data.nc` to the project root.

---

## 4. Evaluate Models

Runs 7-fold cross-validation across all collected events and prints RMSE per model:

python data_dashboard/sklearn_test.py


Models compared: **LinearReg**, **PolyReg** (degree 3), **XGBReg**, **LGBMReg**.

---

## 5. Run the LSTM Notebook (Google Colab)

The deep learning model is in `data_dashboard/FinalProject (8).ipynb` and is designed to run on Google Colab with GPU support.

1. Upload `new_data/`, `back/`, `new_btc_data.nc`, and `btc_back.nc` to your Google Drive
2. Open the notebook in Colab and run all cells
3. The notebook trains a 4-layer LSTM, evaluates on val/test splits, and runs the backtester

> Requires `torch` (included in requirements). GPU is recommended but not required.

---

## 6. Run Backtest (sklearn models)

Simulates a long/short trading strategy starting with $1,000:

python data_dashboard/backtester.py

Output: final portfolio value, total trades executed, and a portfolio history plot.

---

## 7. Analyze Results

Plots and compares model performance across all events:

python data_dashboard/DataAnalyze.py

---

## Project Structure

```
Trading_Project/
├── Backend API + Keys/
│   ├── API.py                  # Kalshi API auth (set your key here)
│   └── PrivKey.pem             # RSA private key (not committed)
├── data_dashboard/
│   ├── coinbase_data.py        # Fetch BTC price data from Coinbase
│   ├── dataMarketCollect.py    # Fetch Kalshi market candlestick data
│   ├── models.py               # ML model definitions & dataset builder
│   ├── sklearn_test.py         # Cross-validation evaluation
│   ├── backtester.py           # Backtesting engine (sklearn models)
│   ├── DataAnalyze.py          # Results analysis & plotting
│   └── FinalProject (8).ipynb  # LSTM model — run on Google Colab
├── new_data/                   # Collected market data (.nc files)
├── new_btc_data.nc             # Collected BTC price data
└── SETUP.md
```

---

## Data Format

All datasets are stored as **NetCDF4** (`.nc`) files using `xarray`.

- **Market data:** dimensions `(market, time_left)` — `time_left` is seconds until expiry (0–3540s)
- **BTC data:** dimensions `(event, time_left)` — includes OHLCV + rolling volatilities (5/15/60-min)

---

## Feature Set (12 features used by all models)

| Feature | Description |
|---|---|
| `time_left` | Seconds until market expiration |
| `cutoff` | Strike price of the market |
| `btc_price` | BTC spot price at that moment |
| `vol5` / `vol15` / `vol60` | Rolling BTC volatility (5/15/60-min) |
| `pct_dist` | % distance of BTC price from strike |
| `log_price` | Log of market price |
| `low` / `high` | BTC candle low/high |
| `candle_spread` | High − low |
| `vol` | Market volume |
