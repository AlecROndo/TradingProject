import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import xarray as xr
import numpy as np
import time


BASE = "https://api.elections.kalshi.com/trade-api/v2"
#Extracting events with specific Ticker:
def get_tickers(): 
    rows = []
    cursor = None

    while True:
        params = {
            "status": "settled"
            }
        if cursor:
            params["cursor"] = cursor

        r = requests.get(f"{BASE}/events", params={**params, "series_ticker": "KXBTCD"})
        data = r.json()

        for event in data.get("events"):
            rows.append([event["title"], event["event_ticker"]])

        cursor = data.get("cursor") 
        if not cursor:
            break
    frame = pd.DataFrame(rows, columns=["title", "ticker"])
    return frame

def market_time(market): 
    while True:
        params = {
            "limit": 200,
        }

        url = f"{BASE}/markets/{market}"
        r = requests.get(url, params=params)
        data = r.json()
        market = data['market']
        return market['open_time'], market['close_time']
        
def get_candles(event, open_time, close_time): 
    #Determining Timeframe for candlesticks
    now = datetime.now(timezone.utc)
    params = {
        "start_ts": int(datetime.fromisoformat(open_time.replace("Z", "+00:00")).timestamp()),
        "end_ts": int(datetime.fromisoformat(close_time.replace("Z", "+00:00")).timestamp()),
        "period_interval": 1
    }


    #3D xarray Dataset

    time_index = None
    #This is where I will start with the 3D DataFrame
    np.set_printoptions(threshold=np.inf)


    url = f"{BASE}/series/KXBTCD/events/{event}/candlesticks"
    r = requests.get(url, params=params)
    data = r.json()



    bid_m = []
    ask_m = []
    prices = []
        

    # 1) Build union of timestamps across markets for THIS event
    all_ts = sorted({
        c['end_period_ts']
        for candles in data['market_candlesticks']
        for c in candles
    })

    time_index = np.array(all_ts, dtype=np.int64)

    bid_m = []
    ask_m = []

    # 2) Align each market to the union timeline
    for candles in data['market_candlesticks']:
        ask_map = {c['end_period_ts']: c['yes_ask']['close_dollars'] for c in candles}
        bid_map = {c['end_period_ts']: c['yes_bid']['close_dollars'] for c in candles}

        ask_ar = np.array([ask_map.get(ts, np.nan) for ts in time_index], dtype=float)
        bid_ar = np.array([bid_map.get(ts, np.nan) for ts in time_index], dtype=float)

        ask_m.append(ask_ar)
        bid_m.append(bid_ar)

    # bid_m and ask_m are now same-length arrays per market
    bid_array = np.array(bid_m)
    ask_array = np.array(ask_m)

    market_ids = np.array(data['market_tickers'])

    ds = xr.Dataset(
    {
        "bid": (("market", "time"), bid_array),
        "ask": (("market", "time"), ask_array),
    },
    coords={
        "time": time_index,
        "market": market_ids
    }
    )
    return ds


def get_market(series,ticker, time):
    url = f'{BASE}/series/{series}/markets/{ticker}/candlesticks'
    now = datetime.now(timezone.utc)
    params = {
    "start_ts": int((now - timedelta(hours=time)).timestamp()),
    "end_ts": int(now.timestamp()),
    "period_interval": 1
    }
    r = requests.get(url, params=params)
    data = r.json()

    list = []

    for candles in data.get('candlesticks'):
        list.append([candles['price']['close'], candles['yes_ask']['close'], candles['yes_bid']['close']])
    df = pd.DataFrame(list, columns=['price','yes_ask', 'yes_bid'])
    print(df)

frame = get_tickers()
print(frame)



