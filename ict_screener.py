import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

TIMEFRAMES = {
    "6M": ("1mo", 3),  # only 6M for screener now
    "1W": ("1wk", 2),
    "1D": ("1d", 1),
    "1H": ("60m", 0.5),
}

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
    """
    Find Fair Value Gaps (FVG) on dataframe
    Returns list of tuples (start_idx, end_idx, low, high)
    Ported simplified logic from pine script: looks for gaps between candle1 and candle3
    where middle candle volatility > threshold
    """
    fvg_list = []
    atr = df['High'] - df['Low']  # rough ATR proxy (simplified)
    atr_ma = atr.rolling(28).mean()
    atr_mult = 1.5

    for i in range(3, len(df)):
        price_diff = df['High'].iloc[i-2] - df['Low'].iloc[i-2]
        middle_candle_vol = price_diff
        if atr_ma.iloc[i-2] is None or np.isnan(atr_ma.iloc[i-2]):
            continue
        if middle_candle_vol <= atr_ma.iloc[i-2] * atr_mult:
            continue

        # bear condition FVG (gap down)
        bear_cond = (df['Close'].iloc[i-3] <= df['High'].iloc[i-2] and
                     df['Close'].iloc[i-1] <= df['Close'].iloc[i-2] and
                     df['High'].iloc[i] < df['Low'].iloc[i-2])

        # bull condition FVG (gap up)
        bull_cond = (df['Close'].iloc[i-3] >= df['Low'].iloc[i-2] and
                     df['Close'].iloc[i-1] >= df['Close'].iloc[i-2] and
                     df['Low'].iloc[i] > df['High'].iloc[i-2])

        if bear_cond or bull_cond:
            is_up_candle = df['Open'].iloc[i-1] <= df['Close'].iloc[i-1]
            top = df['Low'].iloc[i-2] if is_up_candle else df['Low'].iloc[i]
            bottom = df['High'].iloc[i] if is_up_candle else df['High'].iloc[i-2]
            # store as (start idx, end idx, low, high) for FVG box
            fvg_list.append((i-2, i, bottom, top))  # invert low/high for plotting consistency

    return fvg_list

def price_in_fvg(price, fvg):
    """
    check if price is inside a fvg range (low < price < high)
    fvg tuple: (start_idx, end_idx, low, high)
    """
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
                fillcolor='rgba(255, 165, 0, 0.3)',  # orange transparent
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
    tickers_input = st.text_input("enter tickers (comma separated)", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    bullish_stocks = []
    for ticker in tickers:
        interval, score = TIMEFRAMES["6M"]
        df = fetch_data(ticker, interval)
        if df is None or df.empty:
            continue
        last_close = df['Close'].iloc[-1]
        fvg_list = find_fvg(df)
        # check if price has entered any FVG below current price
        fvg_below = [fvg for fvg in fvg_list if price_in_fvg(last_close, fvg) and fvg[3] < last_close]
        if fvg_below:
            bullish_stocks.append({
                'ticker': ticker,
                'score': score,
                'df': df,
                'fvg': fvg_below
            })

    st.subheader("top stocks with price inside 6M FVG below close")
    if not bullish_stocks:
        st.info("no stocks found with price inside a 6M FVG below current price")
    else:
        cols = st.columns(min(3, len(bullish_stocks)))
        for i, stock in enumerate(bullish_stocks[:3]):
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
