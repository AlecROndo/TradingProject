import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# Coinbase endpoint

def bitcoin_data(start, end):
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

    # Time range

    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    coinbase_start = str(int(dt.timestamp()))
    dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    coinbase_end = str(int(dt.timestamp()))
    params = {
        "start": coinbase_start,
        "end": coinbase_end,
        "granularity": 60  # 1 minute candles
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Convert to dataframe
    df = pd.DataFrame(data, columns=[
        "time", "low", "high", "open", "close", "volume"
    ])

    # Convert timestamp
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Coinbase returns newest first → reverse
    df = df.sort_values("time").reset_index(drop=True)
    return df
