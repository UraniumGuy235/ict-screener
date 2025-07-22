import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="ICT Screener", initial_sidebar_state="expanded")

# set dark mode plotly template
plot_template = "plotly_dark"

@st.cache_data
def get_data(ticker, interval, lookback_period):
    try:
        df = yf.download(ticker, interval=interval, period=lookback_period, progress=False)
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        df = df.loc[~df.index.duplicated(keep='first')]  # drop duplicate index
        return df
    except Exception as e:
        st.error(f"error loading {ticker}: {e}")
        return pd.DataFrame()

def find_equals(df, tolerance=0.015):
    equals = []
    highs = df['High']
    lows = df['Low']
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):
            hi1, hi2 = highs.iloc[i], highs.iloc[j]
            lo1, lo2 = lows.iloc[i], lows.iloc[j]

            between_highs = highs.iloc[i+1:j]
            between_lows = lows.iloc[i+1:j]

            if abs(hi1 - hi2) <= tolerance and all(between_highs < min(hi1, hi2)):
                equals.append(('high', df.index[i], df.index[j], min(hi1, hi2)))
            if abs(lo1 - lo2) <= tolerance and all(between_lows > max(lo1, lo2)):
                equals.append(('low', df.index[i], df.index[j], max(lo1, lo2)))
    return equals

def plot_chart(df, equals, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    for eq_type, idx1, idx2, lvl in equals:
        fig.add_trace(go.Scatter(
            x=[idx1, idx2],
            y=[lvl, lvl],
            mode='lines',
            line=dict(color='cyan', width=1.5),
            name=f'{eq_type}-equals',
            hoverinfo='text',
            showlegend=False
        ))

    fig.update_layout(
        title=f"{ticker} ICT Chart",
        xaxis_title='Date',
        yaxis_title='Price',
        template=plot_template,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.sidebar.title("ICT Screener")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    interval = st.sidebar.selectbox("Interval", ["1M", "3mo", "6mo", "1wk", "1d", "1h"])
    lookback_map = {
        "1M": "1y",
        "3mo": "2y",
        "6mo": "5y",
        "1wk": "5y",
        "1d": "1y",
        "1h": "60d"
    }
    interval_map = {
        "1M": "1mo",
        "3mo": "1wk",
        "6mo": "1wk",
        "1wk": "1wk",
        "1d": "1d",
        "1h": "1h"
    }

    lookback_period = lookback_map[interval]
    yf_interval = interval_map[interval]

    if st.sidebar.button("Fetch Data"):
        df = get_data(ticker, yf_interval, lookback_period)
        if not df.empty:
            equals = find_equals(df, tolerance=0.015)
            plot_chart(df, equals, ticker)

if __name__ == "__main__":
    main()
