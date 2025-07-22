import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.title("📈 Candlestick Chart Viewer")

# input ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL").upper()
start = st.date_input("Start Date", datetime.today() - timedelta(days=180))
end = st.date_input("End Date", datetime.today())

# fetch + check data
if ticker:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        st.error("No data found.")
    else:
        df = df.reset_index()
        df['Date'] = df['Date'].dt.tz_localize(None)  # remove timezone
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(
            title=f'{ticker} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)
