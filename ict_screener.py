import streamlit as st
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

# streamlit app title
st.title("candlestick chart viewer")

# ticker input
ticker = st.text_input("enter stock ticker (e.g. AAPL)", "AAPL").upper()

# fetch historical data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period='6mo', interval='1d')
    df.dropna(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df.index.name = 'Date'
    return df

try:
    df = load_data(ticker)
    st.write(f"showing last {len(df)} days of data for {ticker}")

    # plot candlestick chart
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style='charles',
        volume=True,
        mav=(3, 6, 9),
        show_nontrading=True,
        returnfig=True
    )

    # display the plot
    st.pyplot(fig)

except Exception as e:
    st.error(f"something went wrong: {e}")
