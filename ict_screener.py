import streamlit as st
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

st.title("📉 Candlestick Chart (matplotlib)")
ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
start = datetime.today() - timedelta(days=180)
end = datetime.today()

df = yf.download(ticker, start=start, end=end)
if df.empty:
    st.error("No data found.")
else:
    mpf.plot(df, type='candle', style='charles', volume=True, mav=(3,6,9), show_nontrading=True, savefig='chart.png')
    st.image("chart.png")
