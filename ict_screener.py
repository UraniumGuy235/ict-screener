import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("ict bullish stock screener")

# multi timeframe config: label: (yf_interval, priority_score)
TIMEFRAMES = {
    "1M": ("1mo", 3),
    "1W": ("1wk", 2),
    "1D": ("1d", 1),
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
    except Exception as e:
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

def get_bullish_equals_above_price(df, last_close):
    highs_eq = find_equal_levels(df, 'High', tol=0.02)
    # filter equals above current price
    bullish_eq = [(s, e, lvl) for (s, e, lvl) in highs_eq if lvl > last_close]
    return bullish_eq

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
        bullish_eq = get_bullish_equals_above_price(df, last_close)
        if bullish_eq:
            # get closest equals level above price
            closest_level = min(bullish_eq, key=lambda x: x[2])
            gap = closest_level[2] - last_close
            # update best if better timeframe or smaller gap
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

# sort bullish stocks by score desc then gap asc
bullish_stocks = sorted(bullish_stocks, key=lambda x: (-x['score'], x['gap']))

st.subheader("top 3 bullish stocks with equals above price")

if not bullish_stocks:
    st.info("no bullish setups found with equals above current price")
else:
    cols = st.columns(min(3, len(bullish_stocks)))
    for i, stock in enumerate(bullish_stocks[:3]):
        df = stock['df']
        ticker = stock['ticker']
        tf = stock['tf']
        last_close = df['Close'].iloc[-1]
        eqs = stock['setup']

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

        # plot bullish equals highs (orange)
        for start_idx, end_idx, level in eqs:
            fig.add_shape(
                type='line',
                x0=start_idx, x1=end_idx,
                y0=level, y1=level,
                xref='x', yref='y',
                line=dict(color='orange', width=3, dash='solid')
            )

        fig.update_layout(
            title=f"{ticker} bullish equals above price ({tf} timeframe)",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            xaxis=dict(type='category', tickangle=-45),
        )

        with cols[i]:
            st.plotly_chart(fig, use_container_width=True)
