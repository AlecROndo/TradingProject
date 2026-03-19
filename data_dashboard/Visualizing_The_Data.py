import streamlit as st 
from dataCollect import get_tickers, get_candles
from dataCollect import market_time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from datetime import datetime
from zoneinfo import ZoneInfo
from coinbase_data import bitcoin_data

xr.set_options(display_max_rows=1000)
xr.set_options(display_width=200)
event_ticker = "KXBTCD-26MAR1723"
frame = get_tickers
frame = pd.DataFrame()
st.write(frame)
start_time, end_time = market_time('KXBTCD-26MAR1723-T74249.99')
bitcoin = bitcoin_data(start_time, end_time)
print(start_time, end_time)
data = get_candles(event_ticker, start_time, end_time)
x = data


bidlist = np.array(x.sel(market='KXBTCD-26MAR1723-T74249.99')['bid'].values)
asklist = np.array(x.sel(market='KXBTCD-26MAR1723-T74249.99')['ask'].values)
time = x.indexes['time']

utc_time = pd.to_datetime(time, unit="s", utc=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    y= bidlist,
    x = utc_time,
    mode = 'lines',
    name = 'Price Graph Bid'
))
fig.add_trace(go.Scatter(
    y = asklist,
    x = utc_time,
    mode = 'lines',
    name = 'Price Graph Ask'
))
fig.update_layout(
    title = f"Price of {event_ticker}",
    yaxis_title = "Price",
    xaxis_title = "ETC Time",
)
st.plotly_chart(fig, width='stretch')

bitcoin_end = np.array(bitcoin['close'])
bitcoin_time = np.array(bitcoin['time'])
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    y= bitcoin_end,
    x = bitcoin_time,
    mode = 'lines',
    name = 'Price Graph Bid'
))
fig.update_layout(
    title = f"Price of Bitcoin",
    yaxis_title = "Price",
    xaxis_title = "ETC Time",
)
st.plotly_chart(fig2, width='stretch')