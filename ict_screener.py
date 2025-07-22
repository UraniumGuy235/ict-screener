import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

TIMEFRAMES = {
    "1M": ("1mo", 3),
    "1W": ("1wk", 2),
    "1D": ("1d", 1),
    "1H": ("60m", 0.5),  # lower priority
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
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)  # patch: ensure 'Date' column exists
        return df
    except Exception:
        return None

def find_equal_levels(df, price_col='Low', tol=0.02):
    levels = []
    n = len(df)
    for i in range(n):
        base_price = df.loc[i, price_col]
        for j in range(i+1, n):
            test_price = df.loc[j, price_col]
            if abs(test_price - base_price) / base_price <= tol:
                inter_slice = df.loc[i+1:j-1] if j - i > 1 else pd.DataFrame()
                if price_col == 'Low':
                    if not inter_slice.empty and (inter_slice['Low'] < min(base_price, test_price)).any():
                        continue
                else:
                    if not inter_slice.empty and (inter_slice['High'] > max(base_price, test_price)).any():
                        continue
                levels.append((i, j, (base_price + test_price)/2))
            else:
                if price_col == 'Low' and test_price > base_price * (1 + tol):
                    break
                if price_col == 'High' and test_price < base_price * (1 - tol):
                    break
    unique_levels = []
    seen = set()
    for s, e, lvl in levels:
        if (s, e) not in seen and (e, s) not in seen:
            unique_levels.append((s, e, lvl))
            seen.add((s, e))
    return unique_levels

def plot_candles_with_equals(df, equals_highs=None, equals_lows=None, title=""):
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

    if equals_highs:
        for s, e, lvl in equals_highs:
            fig.add_shape(
                type='line',
                x0=s, x1=e,
                y0=lvl, y1=lvl,
                xref='x', yref='y',
                line=dict(color='orange', width=3, dash='solid'),
                name='equals high'
            )
    if equals_lows:
        for s, e, lvl in equals_lows:
            fig.add_shape(
                type='line',
                x0=s, x1=e,
                y0=lvl, y1=lvl,
                xref='x', yref='y',
                line=dict(color='cyan', width=3, dash='solid'),
                name='equals low'
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
            highs_eq = find_equal_levels(df, 'High', tol=0.02)
            bullish_eq = [(s, e, lvl) for (s, e, lvl) in highs_eq if lvl > last_close]
            if bullish_eq:
                closest_level = min(bullish_eq, key=lambda x: x[2])
                gap = closest_level[2] - last_close
                if score > best_score or (score == best_score and (best_gap is None or gap < best_gap)):
                    best_score = score
                    best_gap = gap
                    best_setup = bullish_eq
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
            plot_candles_with_equals(
                stock['df'],
                equals_highs=stock['setup'],
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
            highs_eq = find_equal_levels(df, 'High', tol=0.02)
            lows_eq = find_equal_levels(df, 'Low', tol=0.02)
            plot_candles_with_equals(df, equals_highs=highs_eq, equals_lows=lows_eq, title=f"{ticker} {tf_label} chart with equals")
