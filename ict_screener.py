import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("ICT Stock Screener Dashboard")

TIMEFRAMES = {
    "6M": "1mo",
    "3M": "1wk",
    "1M": "1d",
    "1W": "60m",
    "1D": "30m",
    "1H": "15m",
}

interval_options = list(TIMEFRAMES.keys())

def fetch_data(ticker, interval):
    # download last 5 years max, adjust interval accordingly
    try:
        df = yf.download(ticker, period="5y", interval=interval, progress=False)
        if df is None or df.empty or len(df) == 0:
            return None
        df.index.name = "Date"
        return df
    except Exception:
        return None

def find_equals(df, tolerance=0.005):
    """find pairs of equal highs or lows with no intervening price exceeding that level."""
    equals = []

    lows = df['Low'].values
    highs = df['High'].values
    dates = df.index

    n = len(df)
    for i in range(n - 1):
        for j in range(i + 1, n):
            # check for equal lows
            low1, low2 = lows[i], lows[j]
            if abs(low1 - low2) / low1 <= tolerance:
                # check if no high > max(low1,low2) between i and j
                segment_highs = highs[i+1:j]
                if len(segment_highs) == 0 or np.all(segment_highs <= max(low1, low2)):
                    equals.append(('low', dates[i], dates[j], (low1 + low2) / 2))
            # check for equal highs
            high1, high2 = highs[i], highs[j]
            if abs(high1 - high2) / high1 <= tolerance:
                # check if no low < min(high1,high2) between i and j
                segment_lows = lows[i+1:j]
                if len(segment_lows) == 0 or np.all(segment_lows >= min(high1, high2)):
                    equals.append(('high', dates[i], dates[j], (high1 + high2) / 2))
    return equals

def plot_candles_with_equals(df, equals, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name=ticker
    ))

    # add equals lines between two candles
    for eq_type, date1, date2, price_level in equals:
        fig.add_trace(go.Scatter(
            x=[date1, date2],
            y=[price_level, price_level],
            mode='lines',
            line=dict(color='yellow' if eq_type == 'low' else 'cyan', width=3),
            name=f"{eq_type} equals"
        ))

    fig.update_layout(
        title=f"{ticker} Candlestick Chart with Equals",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# --- UI elements ---

single_ticker = st.text_input("Enter ticker for detailed view", value="")
single_tf = st.selectbox("Select timeframe for detailed view", interval_options, index=2)

if st.button("Fetch Data and Plot") and single_ticker:
    interval = TIMEFRAMES.get(single_tf, "1d")
    df_single = fetch_data(single_ticker, interval)
    if df_single is None:
        st.error(f"No data found for ticker {single_ticker}")
    else:
        equals_single = find_equals(df_single, tolerance=0.01)  # 1% tolerance
        plot_candles_with_equals(df_single, equals_single, single_ticker)
