import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

st.title("Minimal Candlestick Plot Test")

ticker = st.text_input("Ticker", "AAPL")
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.today())

if ticker:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        st.warning("No data fetched")
    else:
        df = df.reset_index()
        # ensure datetime is naive
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # force float in OHLC columns
        for col in ['Open','High','Low','Close']:
            df[col] = df[col].astype(float)

        st.write(df.head())  # debug output

        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'])])

        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600)

        st.plotly_chart(fig, use_container_width=True)
