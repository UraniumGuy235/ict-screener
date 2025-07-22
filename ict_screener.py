import streamlit as st
import yfinance as yf
import pandas as pd

st.title("debug yfinance data")

ticker = st.text_input("ticker", "AAPL")

if st.button("fetch data"):
    df = yf.download(ticker, period='6mo', interval='1d')
    st.write("raw dtypes:")
    st.write(df.dtypes)
    st.write("raw data sample:")
    st.write(df.head())

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    st.write("after coercion + dropna, dtypes:")
    st.write(df.dtypes)
    st.write("after coercion + dropna, sample:")
    st.write(df.head())

    try:
        import mplfinance as mpf
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        mpf.plot(df, type='candle', style='charles', volume=True, mav=(3,6,9),
                 show_nontrading=True, savefig=buf)
        buf.seek(0)
        img = Image.open(buf)
        st.image(img, caption=f"{ticker} Candlestick Chart", use_column_width=True)
    except Exception as e:
        st.error(f"plotting error: {e}")
