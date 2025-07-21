
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("ICT Stock Screener Dashboard")

TIMEFRAMES = {
    "1D": "1d",
    "1W": "1wk",
    "1M": "1mo"
}

interval_option = st.selectbox("Select timeframe for analysis:", list(TIMEFRAMES.keys()), index=1)
interval = TIMEFRAMES[interval_option]

start_date = st.date_input("Start date", datetime.today() - timedelta(days=365 * 5))
end_date = st.date_input("End date", datetime.today())

st.sidebar.title("Manual Ticker Entry")
tickers_input = st.sidebar.text_area("Enter tickers separated by commas", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end, interval):
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        data.dropna(inplace=True)
        return data
    except:
        return None

def detect_fvg(df):
    df = df.copy()
    df['bullish_fvg'] = (df['Low'].shift(1) > df['High'].shift(2))
    return df

def find_equal_lows(df, tolerance=0.01):
    # tolerance loosened to 1%
    lows = df['Low'].rolling(window=3).apply(lambda x: abs(x[0]-x[1]) < tolerance * x[0] and abs(x[1]-x[2]) < tolerance * x[1])
    return lows.fillna(0).astype(bool)

def find_equal_highs(df, tolerance=0.01):
    highs = df['High'].rolling(window=3).apply(lambda x: abs(x[0]-x[1]) < tolerance * x[0] and abs(x[1]-x[2]) < tolerance * x[1])
    return highs.fillna(0).astype(bool)

def open_confluence(df):
    # use percent diff to be scale agnostic
    return df['Open'].rolling(3).apply(lambda x: abs(x[0] - x.mean()) / x.mean() < 0.01).fillna(0).astype(bool)

st.subheader("Scan Results")

if not tickers:
    st.warning("please enter at least one ticker.")
else:
    cols = st.columns(min(len(tickers), 4))
    found_setups = False

    for idx, ticker in enumerate(tickers):
        df = fetch_data(ticker, start=start_date, end=end_date, interval=interval)
        if df is None or df.empty:
            continue
        df = detect_fvg(df)
        df['eq_lows'] = find_equal_lows(df)
        df['eq_highs'] = find_equal_highs(df)
        df['open_confluence'] = open_confluence(df)

        # setup triggers if any indicator fires
        df['setup'] = df['bullish_fvg'] | df['eq_lows'] | df['eq_highs'] | df['open_confluence']

        if df['setup'].any():
            found_setups = True
            with cols[idx % len(cols)]:
                st.markdown(f"### {ticker}")
                fig = go.Figure()

                # candlesticks
                fig.add_trace(go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'],
                                             name='Price'))

                # horizontal lines for equal lows
                eq_lows_levels = df.loc[df['eq_lows'], 'Low'].unique()
                for level in eq_lows_levels:
                    fig.add_hline(y=level, line=dict(color='blue', dash='dash'), annotation_text='Equal Low', annotation_position='bottom left')

                # horizontal lines for equal highs
                eq_highs_levels = df.loc[df['eq_highs'], 'High'].unique()
                for level in eq_highs_levels:
                    fig.add_hline(y=level, line=dict(color='red', dash='dash'), annotation_text='Equal High', annotation_position='top left')

                # markers for bullish FVG
                bullish_fvg_points = df[df['bullish_fvg']]
                fig.add_trace(go.Scatter(x=bullish_fvg_points.index,
                                         y=bullish_fvg_points['Low'] * 0.995,
                                         mode='markers',
                                         marker=dict(color='green', size=10, symbol='triangle-up'),
                                         name='Bullish FVG'))

                # markers for open confluence
                open_conf_points = df[df['open_confluence']]
                fig.add_trace(go.Scatter(x=open_conf_points.index,
                                         y=open_conf_points['Open'],
                                         mode='markers',
                                         marker=dict(color='purple', size=10, symbol='circle'),
                                         name='Open Confluence'))

                fig.update_layout(height=600,
                                  title=f"{ticker} Price Chart with Indicators",
                                  xaxis_title="Date",
                                  yaxis_title="Price")

                st.plotly_chart(fig, use_container_width=True)

    if not found_setups:
        st.info("no setups found today. try adjusting timeframe or start date.")
```

