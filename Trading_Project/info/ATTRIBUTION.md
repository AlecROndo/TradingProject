# Attribution

This file documents the origin and authorship of code, external references, and AI assistance used throughout this project.

---

## AI Assistance (Claude Code)

The following functions received AI assistance from **Claude Code** (Anthropic):

### [data_dashboard/dataMarketCollect.py](data_dashboard/dataMarketCollect.py)

| Function | Attribution |
|---|---|
| `get_market_candles` (line 33) | Originally written by author; edited and improved by Claude Code |
| `get_database_markets` (line 122) | Originally written by author; edited and improved by Claude Code |
| `get_btc_by_event` (line 169) | Primarily written by Claude Code |

### [data_dashboard/backtester.py](data_dashboard/backtester.py)

| Function | Attribution |
|---|---|
| `backtester` (line 62) | Primarily written by Claude Code |
| `plot_portfolio` (line 111) | Primarily written by Claude Code |

### [data_dashboard/FinalProject (8).ipynb](data_dashboard/FinalProject%20(8).ipynb)

| Function | Attribution |
|---|---|
| `DatasetPrepare` | Partially written by Claude Code |
| `build_windows` | Debugged by Claude Code |
| `make_loader` | Debugged by Claude Code |
| `balance_by_price_bin` | Debugged by Claude Code |
| `backtester_lstm` | Primarily written by Claude Code |
| `NaiveModel` | Bug-tested by Claude Code |

---

## External References & Sources

### Kalshi API

- **YouTube — Kalshi API walkthrough:**
  https://www.youtube.com/watch?v=E2mgWN4ReqQ

- **Kalshi API Docs — Historical cutoff timestamps:**
  https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps

---

## Documentation Formatting

The following files were formatted by **Claude Code** (Anthropic) for readability. Content and technical details were provided by the author.

- [ATTRIBUTION.md](ATTRIBUTION.md)
- [SETUP.md](SETUP.md)
- [README.md](README.md)

---

## Data Sources

| Source | Usage |
|---|---|
| [Kalshi Elections API](https://api.elections.kalshi.com/trade-api/v2) | Prediction market candlestick data |
| [Coinbase Exchange API](https://api.exchange.coinbase.com) | Bitcoin OHLCV price data (public, no auth) |
