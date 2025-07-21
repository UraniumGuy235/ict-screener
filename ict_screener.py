import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("ICT Stock Screener Dashboard")

TIMEFRAMES = {
    "1D": "1d",
    "1W": "1wk",
    "1M": "1mo"
}

interval_option = st.selectbox("Select timeframe for analysis:", list(TIMEFRAMES.keys()), index=1)
interval = TIMEFRAMES[interval_option]

start_date = st.date_input("Start date", datetime.today() - timedelta(days=365 * 5))
end_date = st.date_input("End date", datetime.today())

st.sidebar.title("Manual Ticker Entry")
tickers_input = st.sidebar.text_area("Enter tickers separated by commas", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end, interval):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        data.dropna(inplace=True)
        return data
    except:
        return None

def detect_fvg(df):
    df = df.copy()
    df['bullish_fvg'] = (df['Low'].shift(1) > df['High'].shift(2))
    return df

def find_equal_lows(df, tolerance=0.001):
    lows = df['Low'].rolling(window=3).apply(lambda x: abs(x[0]-x[1]) < tolerance and abs(x[1]-x[2]) < tolerance)
    return lows.fillna(0).astype(bool)

def find_equal_highs(df, tolerance=0.001):
    highs = df['High'].rolling(window=3).apply(lambda x: abs(x[0]-x[1]) < tolerance and abs(x[1]-x[2]) < tolerance)
    return highs.fillna(0).astype(bool)

def open_confluence(df):
    return df['Open'].rolling(3).apply(lambda x: abs(x[0] - x.mean()) < 0.1).fillna(0).astype(bool)

st.subheader("Scan Results")

cols = st.columns(len(tickers))
found_setups = False

for idx, ticker in enumerate(tickers):
    df = fetch_data(ticker, start=start_date, end=end_date, interval=interval)
    if df is None or df.empty:
        continue
    df = detect_fvg(df)
    df['eq_lows'] = find_equal_lows(df)
    df['eq_highs'] = find_equal_highs(df)
    df['open_confluence'] = open_confluence(df)

    df['setup'] = df['bullish_fvg'] & (df['eq_lows'] | df['eq_highs']) & df['open_confluence']

    if df['setup'].any():
        found_setups = True
        with cols[idx % len(cols)]:
            st.markdown(f"### {ticker}")
            chart = go.Figure()
            chart.add_trace(go.Candlestick(x=df.index,
                                           open=df['Open'],
                                           high=df['High'],
                                           low=df['Low'],
                                           close=df['Close'],
                                           name='Price'))
            setup_indices = df[df['setup']].index
            chart.add_trace(go.Scatter(x=setup_indices,
                                       y=df.loc[setup_indices]['Close'],
                                       mode='markers',
                                       marker=dict(color='green', size=8),
                                       name='Setup'))
            st.plotly_chart(chart, use_container_width=True)

if not found_setups:
    st.info("No setups found today. Try adjusting the timeframe or start date.")
