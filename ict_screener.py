import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from io import BytesIO
from PIL import Image

st.title("📈 Candlestick Chart Viewer")

ticker = st.text_input("Enter ticker:", "AAPL")
period = st.selectbox("Period:", ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
interval = st.selectbox("Interval:", ['1d', '1h', '30m', '15m'], index=0)

if st.button("Fetch and Plot"):
    try:
        df = yf.download(ticker, period=period, interval=interval)

        if df.empty:
            st.error("No data returned. Check ticker or timeframe.")
        else:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            # coerce to numeric; invalid -> NaN
            df = df.apply(pd.to_numeric, errors='coerce')

            # drop rows with NaNs anywhere
            df.dropna(inplace=True)

            # force float
            df = df.astype(float)

            df.index = pd.to_datetime(df.index)

            # plot with mplfinance, save to buffer
            buf = BytesIO()
            mpf.plot(df, type='candle', style='charles', volume=True, mav=(3,6,9),
                     show_nontrading=True, savefig=buf)
            buf.seek(0)

            img = Image.open(buf)
            st.image(img, caption=f"{ticker} Candlestick Chart", use_column_width=True)

    except Exception as e:
        st.error(f"something went wrong: {e}")
