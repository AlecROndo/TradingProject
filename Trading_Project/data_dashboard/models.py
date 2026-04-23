import numpy as np
import pandas as pd
import xarray as xr
import os

from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt


btcdata = xr.open_dataset("new_btc_data.nc")
data = {
    f.replace(".nc", ""): xr.open_dataset(f"new_data/{f}")
    for f in os.listdir("new_data") if f.endswith(".nc")
}


def build_dataset(data, btcdata):
    rows = []

    for event, ds in data.items():
        if event not in btcdata.event.values:
            continue
        tl_arr = ds.time_left.values
        # Select btcdata for this event by coordinate, not by raw index.
        # btcdata has a union time_left from all events (outer join), so
        # positional indexing [i] would pair market row i with the wrong BTC timestamp.
        btc_event = btcdata.sel(event=event)

        for market in ds.market.values:
            price = ds["price"].sel(market=market).values
            cut   = ds["cutoff"].sel(market=market).values

            for i, tl in enumerate(tl_arr):
                if np.isnan(price[i]):
                    continue
                btc = btc_event.sel(time_left=tl, method="nearest")
                btc_price = float(btc["btc_price"])
                rows.append({
                    "time_left": float(tl),
                    "cutoff":    cut[i],
                    "btc_price": btc_price,
                    "price":     price[i],
                    "vol5":      float(btc["vol_5"]),
                    "vol15":     float(btc["vol_15"]),
                    "vol60":     float(btc["vol_60"]),
                    "pct_dist":  (btc_price - cut[i]) / cut[i],
                    "log_price": np.log(btc_price / cut[i]),
                    "low":       float(btc["low"]),
                    "high":      float(btc["high"]),
                    "candle_spread": float(btc["high"]) - float(btc["low"]),
                    "vol":       float(btc["vol"]),
                })

    df = pd.DataFrame(rows).dropna()
    X = df[["time_left", "cutoff", "btc_price", "vol5", "vol15", "vol60", "pct_dist", "log_price", "low", "high", "candle_spread", "vol"]].values
    y = df["price"].values
    return X, y


X, y = build_dataset(data, btcdata)

def LinearReg(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Lasso(alpha=0.01)
    model.fit(X_scaled, y)
    return model


def PolyReg(X, y, deg):
    model = Pipeline([
        ("poly",   PolynomialFeatures(degree=deg)),
        ("scaler", StandardScaler()),
        ("lasso",  Lasso(alpha=0.01))
    ])
    model.fit(X, y)
    return model


def XGBReg(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    return model
 

def LGBMReg(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, verbose=-1)
    model.fit(X_scaled, y)
    return model