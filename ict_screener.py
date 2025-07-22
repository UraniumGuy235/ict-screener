import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ----- config -----
st.set_page_config(layout="wide")
st.title("FVG Screener - 1Y Candles")

# ----- functions -----
def fetch_data(ticker, period="max"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty or not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            print(f"invalid or missing data for {ticker}")
            return None
        df = df[~df.index.duplicated()]  # drop dupes just in case
        return df
    except Exception as e:
        print(f"error fetching {ticker}: {e}")
        return None

def resample_to_yearly(df):
    return df.resample('1Y').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

def find_fvgs(df):
    fvg_list = []
    for i in range(2, len(df)):
        c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        if c1['High'] < c3['Low']:  # bullish FVG
            fvg_list.append((df.index[i - 2], df.index[i], c1['High'], c3['Low'], 'bullish'))
        elif c1['Low'] > c3['High']:  # bearish FVG
            fvg_list.append((df.index[i - 2], df.index[i], c3['High'], c1['Low'], 'bearish'))
    return fvg_list

def price_inside_any_fvg(price, fvgs):
    for _, _, top, bottom, _ in fvgs:
        if bottom <= price <= top:
            return True
    return False

def plot_chart(df, fvgs, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price'))

    for start, end, top, bottom, kind in fvgs:
        color = 'rgba(0,255,0,0.2)' if kind == 'bullish' else 'rgba(255,0,0,0.2)'
        fig.add_shape(type="rect",
                      x0=end, x1=df.index[-1], y0=bottom, y1=top,
                      fillcolor=color, line=dict(width=0), layer='below')

    fig.update_layout(title=f"{ticker} - 1Y Candles with FVGs",
                      xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    return fig

# ----- ui -----
ticker_input = st.text_input("Enter Ticker (e.g., EOSE)", "EOSE")

if ticker_input:
    df = fetch_data(ticker_input)
    if df is not None:
        yearly_df = resample_to_yearly(df)
        fvgs = find_fvgs(yearly_df)

        latest_price = df['Close'].iloc[-1] if not df.empty else None

        if latest_price and price_inside_any_fvg(latest_price, fvgs):
            st.success(f"{ticker_input} is inside a yearly FVG")
            chart = plot_chart(yearly_df, fvgs, ticker_input)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning(f"{ticker_input} is NOT inside any yearly FVG")
    else:
        st.error("Data unavailable or invalid for ticker")
