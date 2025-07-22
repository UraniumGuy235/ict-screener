import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🌑 Interactive Dark Candlestick Chart")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, GOOGL):", "AAPL")
period = st.selectbox("Select historical period:", ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
interval = st.selectbox("Select interval:", ['1d', '1h', '30m', '15m'], index=0)

if st.button("Fetch and Plot"):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='cyan',
        decreasing_line_color='magenta',
        name='Price'
    )])

    fig.update_layout(
        template='plotly_dark',
        title=f'{ticker} Candlestick Chart',
        xaxis_rangeslider_visible=True,
        xaxis_title='Date',
        yaxis_title='Price',
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)
