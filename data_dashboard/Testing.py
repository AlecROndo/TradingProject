import streamlit as st
from dataCollect import get_candles, market_time
from coinbase_data import bitcoin_data
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

st.set_page_config(page_title="Kalshi Multi-Market Analysis", layout="wide")

# Fill these in yourself
MARKETS = [
    {
        "label": "Example 1",
        "event_ticker": "KXBTCD-26MAR1723",
        "market_ticker": "KXBTCD-26MAR1723-T74249.99",
    },
    {
        "label": "Example 2",
        "event_ticker": "KXBTCD-26MAR0617",
        "market_ticker": "KXBTCD-26MAR0617-T72749.99",
    },
]


@st.cache_data(show_spinner=False)
def load_market_bundle(event_ticker, market_ticker):
    start_time, end_time = market_time(market_ticker)

    if end_time is None:
        end_dt = datetime.now(timezone.utc)
        end_time = end_dt.isoformat().replace("+00:00", "Z")
    else:
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    hours = max(1, int(np.ceil((end_dt - start_dt).total_seconds() / 3600)))

    ds = get_candles(event_ticker, start_time, end_time)
    btc = bitcoin_data(start_time, end_time)

    return ds, btc, start_dt, end_dt, start_time, end_time


def build_market_dataframe(ds, market_ticker):
    bid = np.array(ds.sel(market=market_ticker)["bid"].values, dtype=float)
    ask = np.array(ds.sel(market=market_ticker)["ask"].values, dtype=float)
    ts = ds.indexes["time"]

    utc_time = pd.to_datetime(ts, unit="s", utc=True)
    market_df = pd.DataFrame({
        "time_utc": utc_time,
        "bid": bid,
        "ask": ask,
    })
    market_df["mid"] = (market_df["bid"] + market_df["ask"]) / 2
    market_df["spread"] = market_df["ask"] - market_df["bid"]
    return market_df


def prepare_btc_dataframe(btc, start_dt, end_dt):
    btc_df = btc.copy()
    btc_df["time"] = pd.to_datetime(btc_df["time"], utc=True)
    btc_df = btc_df.sort_values("time")
    btc_df = btc_df[(btc_df["time"] >= start_dt) & (btc_df["time"] <= end_dt)].copy()
    btc_df = btc_df.rename(columns={"time": "time_utc", "close": "btc_close"})
    return btc_df


def merge_market_and_btc(market_df, btc_df):
    left = market_df.sort_values("time_utc").copy()
    right = btc_df[["time_utc", "btc_close"]].sort_values("time_utc").copy()

    merged = pd.merge_asof(
        left,
        right,
        on="time_utc",
        direction="nearest",
        tolerance=pd.Timedelta("2min")
    )
    return merged


def to_est(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York")


def make_dual_chart(merged_df, label):
    est_time = to_est(merged_df["time_utc"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"{label} Kalshi Bid / Ask", "BTC Price at Same Time")
    )

    fig.add_trace(
        go.Scatter(x=est_time, y=merged_df["bid"], mode="lines", name="Bid"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=est_time, y=merged_df["ask"], mode="lines", name="Ask"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=est_time, y=merged_df["btc_close"], mode="lines", name="BTC"),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Kalshi Price", row=1, col=1)
    fig.update_yaxes(title_text="BTC Price", row=2, col=1)
    fig.update_xaxes(title_text="Eastern Time", row=2, col=1)
    fig.update_layout(height=850, title=label)
    return fig


def make_scatter_chart(merged_df, label):
    clean = merged_df.dropna(subset=["mid", "btc_close"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean["btc_close"],
            y=clean["mid"],
            mode="markers",
            name="Kalshi vs BTC",
            text=to_est(clean["time_utc"]).astype(str),
            hovertemplate="BTC: %{x}<br>Kalshi mid: %{y}<br>Time: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{label} | Kalshi Mid vs BTC",
        xaxis_title="BTC Price",
        yaxis_title="Kalshi Mid",
        height=500,
    )
    return fig


def summary_stats(merged_df):
    clean = merged_df.dropna(subset=["bid", "ask", "mid", "btc_close"]).copy()
    if clean.empty:
        return pd.DataFrame({"metric": ["rows"], "value": [0]})

    stats = {
        "rows": len(clean),
        "first_time": clean["time_utc"].iloc[0],
        "last_time": clean["time_utc"].iloc[-1],
        "avg_bid": clean["bid"].mean(),
        "avg_ask": clean["ask"].mean(),
        "avg_spread": clean["spread"].mean(),
        "min_mid": clean["mid"].min(),
        "max_mid": clean["mid"].max(),
        "btc_min": clean["btc_close"].min(),
        "btc_max": clean["btc_close"].max(),
        "corr_mid_btc": clean[["mid", "btc_close"]].corr().iloc[0, 1],
    }
    return pd.DataFrame({"metric": list(stats.keys()), "value": list(stats.values())})


st.title("Kalshi BTC Market Analysis")
st.write("Each top-level tab is one market. Inside each market tab, you get charts, joined data, and quick diagnostics.")

if not MARKETS:
    st.warning("Add your event_ticker and market_ticker values to MARKETS first.")
    st.stop()

market_tab_labels = [m["label"] for m in MARKETS]
market_tabs = st.tabs(market_tab_labels)

for tab, market_info in zip(market_tabs, MARKETS):
    with tab:
        label = market_info["label"]
        event_ticker = market_info["event_ticker"]
        market_ticker = market_info["market_ticker"]

        st.subheader(label)
        st.code(f"event_ticker = '{event_ticker}'\nmarket_ticker = '{market_ticker}'")

        try:
            ds, btc, start_dt, end_dt, start_time, end_time = load_market_bundle(event_ticker, market_ticker)
            market_df = build_market_dataframe(ds, market_ticker)
            btc_df = prepare_btc_dataframe(btc, start_dt, end_dt)
            merged_df = merge_market_and_btc(market_df, btc_df)
        except Exception as e:
            st.error(f"Failed to load {label}: {e}")
            continue

        analysis_tabs = st.tabs(["Charts", "Scatter", "Summary", "Joined Data", "Raw Market Data"])

        with analysis_tabs[0]:
            st.plotly_chart(make_dual_chart(merged_df, label), width='stretch')

        with analysis_tabs[1]:
            st.plotly_chart(make_scatter_chart(merged_df, label), width='stretch')

        with analysis_tabs[2]:
            c1, c2, c3 = st.columns(3)
            c1.metric("Start", start_time)
            c2.metric("End", end_time)
            c3.metric("Joined rows", int(merged_df.dropna(subset=["btc_close"]).shape[0]))
            st.dataframe(summary_stats(merged_df), width='stretch', hide_index=True)

        with analysis_tabs[3]:
            show = merged_df.copy()
            show["time_est"] = to_est(show["time_utc"])
            st.dataframe(show[["time_est", "bid", "ask", "mid", "spread", "btc_close"]], width='stretch', hide_index=True)

        with analysis_tabs[4]:
            raw = market_df.copy()
            raw["time_est"] = to_est(raw["time_utc"])
            st.dataframe(raw[["time_est", "bid", "ask", "mid", "spread"]], width='stretch', hide_index=True)
