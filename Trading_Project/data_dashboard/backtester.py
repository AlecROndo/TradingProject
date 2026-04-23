import xarray as xr
import os
from models import build_dataset, LinearReg, PolyReg, XGBReg, LGBMReg, data, btcdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
back_data = {
    f.replace(".nc", ""): xr.open_dataset(f"back/{f}")
    for f in os.listdir("back") if f.endswith(".nc")
}


btc_back = xr.open_dataset("btc_back.nc")


def backtestdata(back_data, btc_back):
    all_markets = []
    for event, ds in back_data.items():
        tl_arr = ds.time_left.values
        btc_prices = btc_back["btc_price"].sel(event=event).values

        vol_5 = btc_back["vol_5"].sel(event=event).values
        vol_15 = btc_back["vol_15"].sel(event=event).values
        vol_60 = btc_back["vol_60"].sel(event=event).values
        low = btc_back["low"].sel(event=event).values
        high = btc_back["high"].sel(event=event).values
        vol = btc_back["vol"].sel(event=event).values

        for market in ds.market.values:
            rows = []
            price = ds["price"].sel(market=market).values
            cut   = ds["cutoff"].sel(market=market).values

            for i, tl in enumerate(tl_arr):
                if np.isnan(price[i]):
                    continue
                rows.append({
                    "time_left": float(tl),
                    "cutoff":    cut[i],
                    "btc_price": btc_prices[i],
                    "price":     price[i],
                    "vol5": vol_5[i],
                    "vol15": vol_15[i],
                    "vol60": vol_60[i],
                    "pct_dist": (btc_prices[i] - cut[i]) / cut[i],
                    "log_price": np.log(btc_prices[i]/cut[i]),
                    "low": low[i],
                    "high": high[i],
                    "candle_spread": high[i] - low[i],
                    "vol": vol[i]
                })

            df = pd.DataFrame(rows).dropna()
            X = df[["time_left", "cutoff", "btc_price", "vol5", "vol15", "vol60", "pct_dist", "log_price", "low", "high", "candle_spread", "vol"]].values
            y = df["price"].values
            all_markets.append([X, y])
    return all_markets
        
            

def backtester(model, markets, deviation):
    final_values = []
    final_trades = []
    final_port_hist = []
    for X, y in markets:
        cash = 1000
        position = 0
        portfolio_value = 0
        total_trades = 0
        portfolio_hist = []
        for timestep in range(X.shape[0]):
            predprice = model.predict(X[timestep].reshape(1, -1))[0]
            current_price = y[timestep]
            diff = (predprice - current_price)*100
            multiple = round(2 * abs(diff))
            if diff > (deviation - 0.005)*100:
                # close any short first, then go long
                if position < 0:
                    cash += current_price * abs(position)
                    total_trades += abs(position)
                    position = 0
                if position < 1000:
                    position += multiple
                    cash -= current_price * multiple
                    total_trades += multiple
            elif -diff > (deviation - 0.005)*100:
                # close any long first, then go short
                if position > 0:
                    cash -= current_price * position
                    total_trades += position
                    position = 0
                if position > -1000:
                    position -= multiple
                    cash += current_price * multiple
                    total_trades += multiple
            else:
                # no signal — close position
                if position != 0:
                    cash += position * current_price
                    total_trades += abs(position)
                    position = 0
            portfolio_value = cash + position * current_price
            portfolio_hist.append(portfolio_value)
        final_values.append(int(portfolio_value))
        final_trades.append(total_trades)
        final_port_hist.append(portfolio_hist)
    return final_values, final_trades, final_port_hist


def plot_portfolio(final_values):
    plt.figure(figsize=(12, 5))
    plt.plot(final_values, marker="o", markersize=3, linewidth=1)
    plt.axhline(y=1000, color="gray", linestyle="--", linewidth=1, label="Starting capital ($1000)")
    plt.axhline(y=0, color="red", linestyle="--", linewidth=0.8, label="Zero")
    plt.title("Final Portfolio Value per Market")
    plt.xlabel("Market Index")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


all_markets = backtestdata(back_data, btc_back)
X,y = build_dataset(data, btcdata)
model = PolyReg(X, y, 3)
final_values, final_trades, final_port = backtester(model, all_markets, .03)
print(final_values, final_trades)
print(sum(final_values)/len(final_values))
plot_portfolio(final_values)






