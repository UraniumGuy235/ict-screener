import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

TIMEFRAMES = {
    "1M": ("1mo", 3),
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
        return df
    except Exception:
        return None

def find_equals(df, tolerance=0.01):
    highs = df['High'].values
    lows = df['Low'].values
    equals = []
    n = len(df)
    for i in range(n):
        hi1 = highs[i]
        lo1 = lows[i]
        for j in range(i+1, n):
            hi2 = highs[j]
            lo2 = lows[j]

            if j - i > 1:
                inter_highs = highs[i+1:j]
                inter_lows = lows[i+1:j]
                max_high = np.max(inter_highs)
                min_low = np.min(inter_lows)
            else:
                max_high = -np.inf
                min_low = np.inf

            # equals high check: difference within tolerance and no higher high between candles
            if abs(hi1 - hi2) <= tolerance and max_high < min(hi1, hi2):
                equals.append(('high', i, j, (hi1 + hi2) / 2))
            # equals low check: difference within tolerance and no lower low between candles
            elif abs(lo1 - lo2) <= tolerance and min_low > max(lo1, lo2):
                equals.append(('low', i, j, (lo1 + lo2) / 2))
    return equals

def plot_candles_with_equals(df, equals=None, title=""):
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

    if equals:
        for eq_type, start, end, level in equals:
            color = 'orange' if eq_type == 'high' else 'cyan'
            fig.add_shape(
                type='line',
                x0=start, x1=end,
                y0=level, y1=level,
                xref='x', yref='y',
                line=dict(color=color, width=3, dash='solid'),
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
        best_setup = None
        best_score = 0
        best_gap = None
        best_df = None
        best_tf = None
        for tf_label, (interval, score) in TIMEFRAMES.items():
            df = fetch_data(ticker, interval)
            if df is None or df.empty:
                continue
            last_close = df['Close'].iloc[-1]
            equals = find_equals(df, tolerance=0.01)
            highs_eq = [e for e in equals if e[0] == 'high' and e[3] > last_close]
            if highs_eq:
                closest_level = min(highs_eq, key=lambda x: x[3])
                gap = closest_level[3] - last_close
                if score > best_score or (score == best_score and (best_gap is None or gap < best_gap)):
                    best_score = score
                    best_gap = gap
                    best_setup = highs_eq
                    best_df = df
                    best_tf = tf_label
        if best_setup:
            bullish_stocks.append({
                'ticker': ticker,
                'score': best_score,
                'gap': best_gap,
                'setup': best_setup,
                'df': best_df,
                'tf': best_tf
            })

    bullish_stocks = sorted(bullish_stocks, key=lambda x: (-x['score'], x['gap']))
    st.subheader("top 3 bullish stocks with equals above price")
    if not bullish_stocks:
        st.info("no bullish setups found with equals above current price")
    else:
        cols = st.columns(min(3, len(bullish_stocks)))
        for i, stock in enumerate(bullish_stocks[:3]):
            with cols[i]:
                plot_candles_with_equals(
                    stock['df'],
                    equals=stock['setup'],
                    title=f"{stock['ticker']} bullish equals above price ({stock['tf']} timeframe)"
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
            equals = find_equals(df, tolerance=0.01)
            plot_candles_with_equals(df, equals=equals, title=f"{ticker} {tf_label} chart with equals")
