import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

TIMEFRAMES = {
    "6M": ("1mo", 3),
    "1W": ("1wk", 2),
    "1D": ("1d", 1),
    "1H": ("60m", 0.5),
}

# example universe (replace with larger universe or user input as needed)
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "DIS", "NFLX", "ADBE", "PYPL", "CSCO",
    "INTC", "CMCSA", "PEP", "KO", "XOM", "BA", "CRM", "ABT", "NKE",
    "WMT", "T", "CVX", "MCD", "COST", "ACN", "IBM", "TXN", "QCOM",
    "ORCL", "MDT", "AMGN", "HON", "UPS", "UNH", "LOW", "CAT", "AXP"
]

def fetch_data(ticker, interval):
    try:
        df = yf.download(ticker, period="1y", interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        return df
    except Exception:
        return None

def find_fvg(df):
    fvg_list = []
    atr = df['High'] - df['Low']
    atr_ma = atr.rolling(28).mean()
    atr_mult = 1.5

    for i in range(3, len(df)):
        if i >= len(atr_ma) or pd.isna(atr_ma.iloc[i-2]):
            continue
        price_diff = df['High'].iloc[i-2] - df['Low'].iloc[i-2]
        middle_candle_vol = price_diff
        if middle_candle_vol <= atr_ma.iloc[i-2] * atr_mult:
            continue

        bear_cond = (df['Close'].iloc[i-3] <= df['High'].iloc[i-2] and
                     df['Close'].iloc[i-1] <= df['Close'].iloc[i-2] and
                     df['High'].iloc[i] < df['Low'].iloc[i-2])

        bull_cond = (df['Close'].iloc[i-3] >= df['Low'].iloc[i-2] and
                     df['Close'].iloc[i-1] >= df['Close'].iloc[i-2] and
                     df['Low'].iloc[i] > df['High'].iloc[i-2])

        if bear_cond or bull_cond:
            is_up_candle = df['Open'].iloc[i-1] <= df['Close'].iloc[i-1]
            top = df['Low'].iloc[i-2] if is_up_candle else df['Low'].iloc[i]
            bottom = df['High'].iloc[i] if is_up_candle else df['High'].iloc[i-2]
            fvg_list.append((i-2, i, bottom, top))

    return fvg_list

def price_in_fvg(price, fvg):
    _, _, low, high = fvg
    return low < price < high

def plot_candles_with_fvg(df, fvg_list=None, title=""):
    x_vals = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    fig = go.Figure(data=[go.Candlestick(
        x=x_vals,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='price'
    )])

    if fvg_list:
        for start, end, low, high in fvg_list:
            fig.add_shape(
                type='rect',
                x0=start, x1=end,
                y0=low, y1=high,
                xref='x', yref='y',
                fillcolor='rgba(255, 165, 0, 0.3)',
                line=dict(color='rgba(255,165,0,0.8)', width=2, dash='dash')
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

mode = st.radio("select mode", ("screener", "single ticker"))

if mode == "screener":
    batch_size = 10
    found_stocks = []

    tickers_input = st.text_input("enter tickers to screen (comma separated, leave blank to use universe)", "")
    if tickers_input.strip():
        universe = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    else:
        universe = UNIVERSE.copy()

    # session state for pagination
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'found_stocks' not in st.session_state:
        st.session_state.found_stocks = []

    def screen_batch():
        idx = st.session_state.index
        batch = universe[idx:idx+batch_size]
        new_found = []
        interval, _ = TIMEFRAMES["6M"]
        for ticker in batch:
            df = fetch_data(ticker, interval)
            if df is None or df.empty:
                continue
            last_close = df['Close'].iloc[-1]
            fvg_list = find_fvg(df)
            fvg_below = [fvg for fvg in fvg_list if price_in_fvg(last_close, fvg) and fvg[3] < last_close]
            if fvg_below:
                new_found.append({'ticker': ticker, 'df': df, 'fvg': fvg_below})
        st.session_state.found_stocks.extend(new_found)
        st.session_state.index += batch_size

    if st.button("search next batch"):
        screen_batch()

    # initial search if nothing found yet
    if not st.session_state.found_stocks and st.session_state.index == 0:
        screen_batch()

    found_stocks = st.session_state.found_stocks

    st.subheader("stocks with price inside 6M FVG below current price")
    if not found_stocks:
        st.info("no stocks found yet, press 'search next batch' to scan more")
    else:
        cols = st.columns(min(3, len(found_stocks)))
        for i, stock in enumerate(found_stocks[:3]):
            plot_candles_with_fvg(
                stock['df'],
                fvg_list=stock['fvg'],
                title=f"{stock['ticker']} price inside 6M FVG (6M timeframe)"
            )

elif mode == "single ticker":
    ticker = st.text_input("enter ticker symbol", "AAPL").upper()
    tf_selected = st.multiselect("select timeframe(s)", options=list(TIMEFRAMES.keys()), default=["1D", "1W"])

    if ticker and tf_selected:
        for tf_label in tf_selected:
            interval, _ = TIMEFRAMES[tf_label]
            df = fetch_data(ticker, interval)
            if df is None or df.empty:
                st.warning(f"no data for {ticker} on {tf_label}")
                continue
            fvg_list = find_fvg(df)
            plot_candles_with_fvg(df, fvg_list=fvg_list, title=f"{ticker} {tf_label} chart with FVG")
