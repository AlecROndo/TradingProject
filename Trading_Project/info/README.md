# Bitcoin Derivatives Trading System

## What it Does

This project builds a machine learning pipeline that predicts the price of Bitcoin binary options on the [Kalshi](https://kalshi.com) prediction market. It collects live candlestick data from the Kalshi API and 1-minute BTC price data from Coinbase, engineers features like rolling volatility windows, strike distance, and time-to-expiry, then trains and evaluates five models — Lasso (Linear), Lasso (Poly-3), XGBoost, LightGBM, and a 4-layer NN + LSTM — to forecast market prices. A backtesting engine uses those predictions to simulate long/short trading strategies across historical events, starting with $1,000 of capital and sizing positions dynamically based on model confidence.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect market and BTC data
python data_dashboard/dataMarketCollect.py

# 3. Evaluate sklearn models (cross-validated RMSE)
python data_dashboard/sklearn_test.py

# 4. Run sklearn backtest
python data_dashboard/backtester.py

# 5. Analyze and plot results
python data_dashboard/DataAnalyze.py

# 6. LSTM model — open in Google Colab
#    data_dashboard/FinalProject (8).ipynb
```

See [SETUP.md](SETUP.md) for full setup instructions including API credential configuration.

---

## Video Links

**Demo:**

**Technical Walkthrough:**

---

## Evaluation

### sklearn Models — 7-fold cross-validated RMSE (19 events)

| Model | Mean Test RMSE |
|---|---|
| Lasso (Linear) | 0.1929 |
| Lasso (Poly-3) | 0.1446 |
| XGBoost | 0.0810 |
| LightGBM | 0.0797 |

RMSE is on a 0–1 scale (market prices are probabilities). LightGBM and XGBoost are roughly 2.4× better than the linear baseline. Chart saved to [rmse_by_model.png](rmse_by_model.png).

### LSTM Model — Train/Val/Test split (FinalProject notebook)

| Split | MAE | RMSE |
|---|---|---|
| Train | 0.1367 | 0.1709 |
| Val | 0.1373 | 0.1702 |
| Test | 0.1425 | 0.1721 |
| Naive baseline (test) | 0.0744 | 0.1303 |

4-layer LSTM (hidden=256) trained for 15 epochs with early stopping. Training used class-balanced upsampling across 10 price bins.

### Backtesting Results (3% deviation threshold, $1,000 starting capital)

| Run | Avg Final Value | Win Rate |
|---|---|---|
| LSTM — val markets | $1,092.83 | 3/12 profitable |
| LSTM — test markets | $1,016.15 | 4/27 profitable |
| Naive baseline — test | $930.11 | 8/27 profitable |

While we did see a profit in the test markets of ~$400, after 35 markets, the % profitablility is clearly a red flag with my current LSTM model that I will continue to look into.
---

## Design Choices

The project evolved in two distinct phases.

**Phase 1 — Options-style price prediction:** The original goal was to predict the Kalshi market price using only current BTC price data and time remaining until expiry, analogous to how options pricing models (e.g. Black-Scholes) derive contract value from the underlying asset price, strike distance, and time-to-expiry. Features like rolling volatility windows, strike distance, and a time-decay term were engineered to capture this relationship. The sklearn models (Lasso, XGBoost, LightGBM) were the primary output of this phase and served as a baseline for how well contract price could be explained by current market state alone.

**Phase 2 — Sequential prediction with NN/LSTM:** After establishing that the options-style features could meaningfully explain price, the question became whether it was possible to actually *profit* from predictions — i.e., whether future price movements were predictable from historical sequences of those same features. This motivated adding an NN/LSTM that takes a sliding window of past observations as input and tries to forecast where the market price is headed, enabling a long/short backtesting strategy on top of the predictions.

---

## Individual Contributions

This project was completed individually.

For a detailed breakdown of which functions were written, edited, or debugged with AI assistance, see [ATTRIBUTION.md](ATTRIBUTION.md).