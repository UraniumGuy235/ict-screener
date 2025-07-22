import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("ICT Stock Screener - Monthly Equals Only")

TIMEFRAMES = {
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1m": "1mo",
    "3m": "3mo",
    "6m": "6mo",
}
SCREENER_INTERVAL = "1mo"  # fixed for screener only monthly

# sidebar for user input
st.sidebar.header("Screener Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers for screening (comma separated)",
    "AAPL,MSFT,GOOGL,TSLA,AMZN,IBM,INTC,AMD,NFLX",
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

st.sidebar.header("Single Stock Lookup")
single_ticker = st.sidebar.text_input("Ticker for detailed chart", "AAPL").upper()
single_tf = st.sidebar.selectbox(
    "Select timeframe for single stock chart", list(TIMEFRAMES.keys()), index=3
)

@st.cache_data(show_spinner=False)
def batch_fetch_1m_data(tickers):
    try:
        df_all = yf.download(
            tickers, period="5y", interval=SCREENER_INTERVAL, group_by="ticker", progress=False
        )
        cleaned = {}
        for t in tickers:
            if t in df_all:
                df = df_all[t].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                df.index.name = "Date"
                cleaned[t] = df
        return cleaned
    except Exception as e:
        st.error(f"batch fetch failed: {e}")
        return {}

def find_equals(df, tolerance=0.01):
    """Find pairs of equal highs or lows with no price breaches between."""
    eq_lines = []

    # Helper to check no price breaches between candles i and j at price level
    def no_breach(df, i, j, level, is_high):
        if i > j:
            i, j = j, i
        for idx in range(i + 1, j):
            row = df.iloc[idx]
            if is_high and row["High"] > level:
                return False
            if not is_high and row["Low"] < level:
                return False
        return True

    # Scan lows
    lows = df["Low"].values
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) <= tolerance * lows[i]:
                if no_breach(df, i, j, lows[i], is_high=False):
                    eq_lines.append(("low", i, j, lows[i]))

    # Scan highs
    highs = df["High"].values
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) <= tolerance * highs[i]:
                if no_breach(df, i, j, highs[i], is_high=True):
                    eq_lines.append(("high", i, j, highs[i]))

    return eq_lines

def plot_candles_with_equals(df, eq_lines, ticker):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="green",
                decreasing_line_color="red",
                name=ticker,
            )
        ]
    )
    # add equals lines
    for etype, i, j, price in eq_lines:
        x0, x1 = df.index[i], df.index[j]
        fig.add_shape(
            type="line",
            x0=x0,
            y0=price,
            x1=x1,
            y1=price,
            line=dict(color="cyan", width=3, dash="solid"),
            xref="x",
            yref="y",
        )
    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} Candlestick with Equals (1M)",
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def bullish_equals_above_price(df, eq_lines):
    """Return True if any equals line is above latest close price."""
    if df.empty or not eq_lines:
        return False
    current_price = df["Close"].iloc[-1]
    for etype, i, j, price in eq_lines:
        if price > current_price:
            return True
    return False

# ========== SCREENER ==========
st.header("Monthly Screener - Top 3 Stocks with Bullish Equals Above Price")

if tickers:
    data_dict = batch_fetch_1m_data(tickers)
    bullish_candidates = []
    for tkr, df in data_dict.items():
        eqs = find_equals(df, tolerance=0.01)  # 1% tolerance
        if bullish_equals_above_price(df, eqs):
            bullish_candidates.append((tkr, df, eqs))
    # sort by how high the equals are above price (descending)
    bullish_candidates.sort(
        key=lambda x: max([price for _, _, _, price in x[2]]), reverse=True
    )
    top3 = bullish_candidates[:3]

    if top3:
        for tkr, df, eqs in top3:
            st.subheader(f"{tkr} - Bullish Equals on 1M")
            plot_candles_with_equals(df, eqs, tkr)
    else:
        st.info("No bullish equals above current price found in the given tickers.")
else:
    st.warning("Enter tickers in the sidebar to run the screener.")

# ========== SINGLE STOCK ==========
st.header("Single Stock Detailed Chart")

if single_ticker:
    try:
        interval = TIMEFRAMES[single_tf]
        df_single = yf.download(
            single_ticker, period="5y", interval=interval, progress=False
        )
        if df_single.empty:
            st.error("No data found for ticker.")
        else:
            df_single.index.name = "Date"
            eqs_single = find_equals(df_single, tolerance=0.01)
            plot_candles_with_equals(df_single, eqs_single, single_ticker)
    except Exception as e:
        st.error(f"Error fetching data for {single_ticker}: {e}")
