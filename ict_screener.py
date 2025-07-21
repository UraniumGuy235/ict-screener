import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("ICT Stock Screener Dashboard (Plotly Express)")

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
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def detect_fvg(df):
    df = df.copy()
    df['bullish_fvg'] = (df['Low'].shift(1) > df['High'].shift(2))
    return df

def find_equal_lows(df, tolerance=0.01):
    def is_equal(arr):
        if len(arr) < 3:
            return False
        return (abs(arr[0] - arr[1]) < tolerance * arr[0]) and (abs(arr[1] - arr[2]) < tolerance * arr[1])
    lows_bool = df['Low'].rolling(window=3).apply(is_equal, raw=True)
    return lows_bool.fillna(0).astype(bool)

def find_equal_highs(df, tolerance=0.01):
    def is_equal(arr):
        if len(arr) < 3:
            return False
        return (abs(arr[0] - arr[1]) < tolerance * arr[0]) and (abs(arr[1] - arr[2]) < tolerance * arr[1])
    highs_bool = df['High'].rolling(window=3).apply(is_equal, raw=True)
    return highs_bool.fillna(0).astype(bool)

def open_confluence(df):
    def is_confluent(x):
        if len(x) < 3:
            return False
        return abs(x[0] - x.mean()) / x.mean() < 0.01
    return df['Open'].rolling(3).apply(is_confluent).fillna(0).astype(bool)

st.subheader("Scan Results")

if not tickers:
    st.warning("please enter at least one ticker.")
else:
    cols = st.columns(min(len(tickers), 4))
    found_setups = False

    for idx, ticker in enumerate(tickers):
        df = fetch_data(ticker, start=start_date, end=end_date, interval=interval)
        if df is None or df.empty:
            st.warning(f"no data for {ticker}")
            continue

        df = detect_fvg(df)
        df['eq_lows'] = find_equal_lows(df)
        df['eq_highs'] = find_equal_highs(df)
        df['open_confluence'] = open_confluence(df)

        missing_cols = [c for c in ['eq_lows', 'eq_highs', 'open_confluence'] if c not in df.columns]
        if missing_cols:
            st.error(f"missing columns {missing_cols} for {ticker}, skipping")
            continue

        df['setup'] = df['bullish_fvg'] | df['eq_lows'] | df['eq_highs'] | df['open_confluence']

        if df['setup'].any():
            found_setups = True
            with cols[idx % len(cols)]:
                st.markdown(f"### {ticker}")

                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

                # force float dtype for OHLC columns
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col].astype(float)

                # plotly express candlestick chart
                fig = px.candlestick(df, x='Date', open='Open', high='High', low='Low', close='Close',
                                     title=f"{ticker} Price Chart with Indicators", height=600)

                try:
                    eq_lows_mask = df['eq_lows'].astype(bool)
                    eq_lows_levels = df.loc[eq_lows_mask, 'Low']
                    if isinstance(eq_lows_levels, pd.DataFrame):
                        eq_lows_levels = eq_lows_levels.iloc[:, 0]
                    eq_lows_levels = eq_lows_levels.unique()
                    for level in eq_lows_levels:
                        fig.add_hline(y=level, line_dash='dash', line_color='blue',
                                      annotation_text='Equal Low', annotation_position='bottom left')
                except Exception as e:
                    st.error(f"Error plotting equal lows for {ticker}: {e}")

                try:
                    eq_highs_mask = df['eq_highs'].astype(bool)
                    eq_highs_levels = df.loc[eq_highs_mask, 'High']
                    if isinstance(eq_highs_levels, pd.DataFrame):
                        eq_highs_levels = eq_highs_levels.iloc[:, 0]
                    eq_highs_levels = eq_highs_levels.unique()
                    for level in eq_highs_levels:
                        fig.add_hline(y=level, line_dash='dash', line_color='red',
                                      annotation_text='Equal High', annotation_position='top left')
                except Exception as e:
                    st.error(f"Error plotting equal highs for {ticker}: {e}")

                bullish_fvg_points = df[df['bullish_fvg']]
                fig.add_scatter(x=bullish_fvg_points['Date'], y=bullish_fvg_points['Low'] * 0.995,
                                mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                                name='Bullish FVG')

                open_conf_points = df[df['open_confluence']]
                fig.add_scatter(x=open_conf_points['Date'], y=open_conf_points['Open'],
                                mode='markers', marker=dict(color='purple', size=10, symbol='circle'),
                                name='Open Confluence')

                st.plotly_chart(fig, use_container_width=True)

    if not found_setups:
        st.info("no setups found today. try adjusting timeframe or start date.")
