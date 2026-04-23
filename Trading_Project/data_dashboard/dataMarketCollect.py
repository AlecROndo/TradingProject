import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import xarray as xr
from coinbase_data import bitcoin_data


BASE = "https://api.elections.kalshi.com/trade-api/v2"


def market_time(market):
    """
    Get open_time and close_time for a specific market ticker.
    """
    url = f"{BASE}/markets/{market}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    if "market" in data:
        m = data["market"]
    else:
        m = data

    if "open_time" not in m or "close_time" not in m:
        raise ValueError(f"Missing open/close time for {market}: {data}")

    return m["open_time"], m["close_time"]


def get_market_candles(market, open_time, close_time):
    """
    Fetch candlesticks for ONE specific market ticker only.
    Returns an xarray Dataset with dims: market, time_left
    """
    close_time_ts = int(pd.to_datetime(close_time, utc=True).timestamp())
    start_ts = close_time_ts - 3600
    end_ts = close_time_ts

    # cutoff / strike from ticker
    cutoff_str = market.split("-")[-1]   # e.g. T68799.99
    cutoff = float(cutoff_str[1:])       # -> 68799.99

    params = {
        "period_interval": 1,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "limit": 100
    }

    series_ticker = market.split("-")[0]
    url = f"{BASE}/series/{series_ticker}/markets/{market}/candlesticks"

    all_candles = []
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor
        elif "cursor" in params:
            del params["cursor"]

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        candles = data.get("candlesticks", [])
        all_candles.extend(candles)

        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break

    if len(all_candles) == 0:
        raise ValueError(f"No candlesticks returned for {market}")

    all_ts = sorted({c["end_period_ts"] for c in all_candles})
    time_index = np.array(all_ts, dtype=np.int64)

    time_left = close_time_ts - time_index
    order = np.argsort(-time_left)

    time_left = time_left[order]
    time_index = time_index[order]

    ask_map = {c["end_period_ts"]: float(c["yes_ask"]["close_dollars"]) for c in all_candles}
    bid_map = {c["end_period_ts"]: float(c["yes_bid"]["close_dollars"]) for c in all_candles}
    volume_map = {c["end_period_ts"]: float(c["volume_fp"]) for c in all_candles}

    price_map = {
        c["end_period_ts"]: (
            float(c["yes_ask"]["close_dollars"]) + float(c["yes_bid"]["close_dollars"])
        ) / 2
        for c in all_candles
    }

    ask_ar = np.array([ask_map.get(ts, np.nan) for ts in time_index], dtype=float)
    bid_ar = np.array([bid_map.get(ts, np.nan) for ts in time_index], dtype=float)
    vol_ar = np.array([volume_map.get(ts, np.nan) for ts in time_index], dtype=float)
    price_ar = np.array([price_map.get(ts, np.nan) for ts in time_index], dtype=float)

    cutoff_ar = np.full(len(time_left), cutoff, dtype=float)

    ds = xr.Dataset(
        {
            "bid": (("market", "time_left"), np.array([bid_ar])),
            "ask": (("market", "time_left"), np.array([ask_ar])),
            "volume": (("market", "time_left"), np.array([vol_ar])),
            "price": (("market", "time_left"), np.array([price_ar])),
            "cutoff": (("market", "time_left"), np.array([cutoff_ar])),
        },
        coords={
            "time_left": time_left,
            "market": np.array([market]),
        }
    )

    return ds

def get_database_markets(market_list):
    """
    Market-based version of your get_database.
    Returns a dict of {event_str: xr.Dataset} where each Dataset has
    dims = (market, time_left) — no shared event dimension.

    market_list example:
    [
        "KXBTCD-26APR0621-T68799.99",
        "KXBTCD-26APR0621-T68899.99",
        "KXHIGHLAX-26APR03-B78.5"
    ]
    """
    # group selected markets by event ticker
    event_to_markets = {}
    for market in market_list:
        event = "-".join(market.split("-")[:2])   # e.g. KXBTCD-26APR0621
        event_to_markets.setdefault(event, []).append(market)

    all_event_datasets = {}

    for event, markets in event_to_markets.items():
        market_ds_list = []

        for market in markets:
            open_time, close_time = market_time(market)
            ds = get_market_candles(market, open_time, close_time)
            time.sleep(0.4)
            market_ds_list.append(ds)

        # union timeline only across the SELECTED markets in this event
        all_time = np.unique(np.concatenate([ds.time_left.values for ds in market_ds_list]))[::-1]

        for i in range(len(market_ds_list)):
            market_ds_list[i] = market_ds_list[i].reindex(time_left=all_time)
            market_ds_list[i] = market_ds_list[i].assign_coords(
                time_left=market_ds_list[i].time_left.values
            )
            market_ds_list[i] = market_ds_list[i].ffill(dim="time_left")

        event_ds = xr.concat(market_ds_list, dim="market", join="outer")
        all_event_datasets[event] = event_ds

    return all_event_datasets



def get_btc_by_event(market_list):
    """
    Build an xarray Dataset of BTC data for each event.

    Input:
        market_list: list[str]
            Example:
            [
                "KXBTCD-26APR0621-T68799.99",
                "KXBTCD-26APR0621-T68899.99",
                "KXBTCD-26APR0721-T69099.99"
            ]

    Returns:
        xr.Dataset with dims:
            event, time_left

        Variables expected:
            btc_price(event, time_left)

    Assumes:
        - market_time(market) -> (open_time, close_time)
        - bitcoin_data(start_ts, end_ts) returns a pandas DataFrame
          with at least:
              timestamp (or time)
              price (or close)
    """
    event_to_markets = {}
    for market in market_list:
        event = "-".join(market.split("-")[:2])
        event_to_markets.setdefault(event, []).append(market)

    all_event_btc = []

    for event, markets in event_to_markets.items():
        # Collect times for all markets in this event
        event_open_times = []
        event_close_times = []

        for market in markets:
            open_time, close_time = market_time(market)
            event_open_times.append(pd.to_datetime(open_time, utc=True))
            event_close_times.append(pd.to_datetime(close_time, utc=True))

        # Use one event-level window
        event_open_time = min(event_open_times)
        event_close_time = max(event_close_times)

        event_open_ts = int(event_open_time.timestamp())
        event_close_ts = int(event_close_time.timestamp())

        # Fetch BTC data with 60-min lookback for volatility calculation
        lookback_str  = pd.to_datetime(event_open_ts - 3600, unit="s", utc=True).isoformat()
        event_close_str = pd.to_datetime(event_close_ts, unit="s", utc=True).isoformat()

        extended_df = bitcoin_data(lookback_str, event_close_str).copy()

        # --- adjust these column names if needed ---
        if "timestamp" in extended_df.columns:
            ts_col = "timestamp"
        elif "time" in extended_df.columns:
            ts_col = "time"
        else:
            raise ValueError("bitcoin_data output must contain 'timestamp' or 'time' column")

        if "price" in extended_df.columns:
            price_col = "price"
        elif "close" in extended_df.columns:
            price_col = "close"
        else:
            raise ValueError("bitcoin_data output must contain 'price' or 'close' column")

        # Convert timestamps, sort ascending for rolling calc
        extended_df[ts_col] = pd.to_datetime(extended_df[ts_col], utc=True)
        extended_df["timestamp_sec"] = extended_df[ts_col].astype("int64") // 10**9
        extended_df = extended_df.sort_values("timestamp_sec")
        
        # Compute volatility on full extended series (lookback + event window)
        ret = extended_df[price_col].diff()
        vol_5  = ret.rolling(window=5,  min_periods=5).std()
        vol_15 = ret.rolling(window=15, min_periods=15).std()
        vol_60 = ret.rolling(window=60, min_periods=60).std()

        # Slice to event window only (time_left 0–3540)
        extended_df["time_left"] = event_close_ts - extended_df["timestamp_sec"]
        btc_df = extended_df[(extended_df["time_left"] >= 0) & (extended_df["time_left"] <= 3540)].copy()
        btc_df["vol_5"]  = vol_5.loc[btc_df.index].values
        btc_df["vol_15"] = vol_15.loc[btc_df.index].values
        btc_df["vol_60"] = vol_60.loc[btc_df.index].values

        # sort descending, drop duplicates
        btc_df = btc_df.sort_values("time_left", ascending=False)
        btc_df = btc_df.drop_duplicates(subset="time_left", keep="last")

        time_left = btc_df["time_left"].to_numpy(dtype=np.int64)
        btc_price = btc_df[price_col].to_numpy(dtype=float)
        # build event-level xarray
        event_ds = xr.Dataset(
            {
                "btc_price":  ("time_left", btc_price),
                "high":   ("time_left", btc_df["high"].to_numpy(dtype=float)),
                "low":    ("time_left", btc_df["low"].to_numpy(dtype=float)),
                "vol": ("time_left", btc_df["volume"].to_numpy(dtype=float)),
                "vol_5":      ("time_left", btc_df["vol_5"].to_numpy(dtype=float)),
                "vol_15":     ("time_left", btc_df["vol_15"].to_numpy(dtype=float)),
                "vol_60":     ("time_left", btc_df["vol_60"].to_numpy(dtype=float)),
            },
            coords={
                "time_left": time_left,
            }
        )

        event_ds = event_ds.expand_dims(event=[event])
        all_event_btc.append(event_ds)

    combined_btc = xr.concat(all_event_btc, dim="event", join="outer")
    return combined_btc



def volatility_calc(start, end):
    if not isinstance(start, datetime):
        start = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    if not isinstance(end, datetime):
        end = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    price = bitcoin_data((start - timedelta(minutes=60)).isoformat(), end.isoformat())

    ret = price["close"].diff()
    vol_5  = ret.rolling(window=5,  min_periods=5).std().iloc[-60:]
    vol_15 = ret.rolling(window=15, min_periods=15).std().iloc[-60:]
    vol_60 = ret.rolling(window=60, min_periods=60).std().iloc[-60:]


    return vol_5, vol_15, vol_60
