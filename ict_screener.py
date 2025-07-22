import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("ict stock screener dashboard")

TIMEFRAMES = {
    "6M": "1d",
    "3M": "1d",
    "1M": "1d",
    "1W": "1d",
    "1D": "1h",
    "1H": "5m"
}

# map timeframes to yfinance intervals and appropriate start date offsets
TIMEFRAME_OFFSETS = {
    "6M": 180,
    "3M": 90,
    "1M": 30,
    "1W": 7,
    "1D": 1,
    "1H": 1  # 1 day with 5m interval
}

interval_option = st.selectbox("select timeframe for analysis:", list(TIMEFRAMES.keys()), index=2)
interval = TIMEFRAMES[interval_option]

# auto set start date based on timeframe offset, override with input if desired
default_start = datetime.today() - timedelta(days=TIMEFRAME_OFFSETS[interval_option])
start_date = st.date_input("start date", default_start)
end_date = st.date_input("end date", datetime.today())

tickers_input = st.text_input("enter tickers (comma separated)", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end, interval):
    try:
        df = yf.download(ticker, start=start, end=end + timedelta(days=1), interval=interval, progress=False)
        if df.empty:
            return None
        # fix column names if multiindex returned
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return df
    except Exception as e:
        st.error(f"error fetching data for {ticker}: {e}")
        return None

def find_equal_levels(df, price_col='Low', tol=0.01):
    levels = []
    n = len(df)
    for i in range(n):
        base_price = df.iloc[i][price_col]
        cluster = [i]
        for j in range(i+1, n):
            test_price = df.iloc[j][price_col]
            if abs(test_price - base_price) / base_price < tol:
                # check no break in between
                inter_df = df.iloc[i+1:j]
                if price_col == 'Low':
                    if (inter_df['Low'] < base_price).any():
                        break
                else:
                    if (inter_df['High'] > base_price).any():
                        break
                cluster.append(j)
            else:
                break
        if len(cluster) > 1:
            levels.append((cluster[0], cluster[-1], base_price))

    # remove duplicates by start-end index
    unique_levels = []
    seen = set()
    for s, e, lvl in levels:
        if (s, e) not in seen:
            unique_levels.append((s, e, lvl))
            seen.add((s, e))
    return unique_levels

st.subheader("scan results")
found_setups = False
cols = st.columns(len(tickers))

for idx, ticker in enumerate(tickers):
    df = fetch_data(ticker, start_date, end_date, interval)
    if df is None or df.empty:
        with cols[idx % len(cols)]:
            st.warning(f"no data for {ticker}")
        continue

    eq_lows_levels = find_equal_levels(df, price_col='Low', tol=0.005)  # 2% tolerance
    eq_highs_levels = find_equal_levels(df, price_col='High', tol=0.005)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='price'
    )])

    # plot equal lows horizontal lines
    for start_idx, end_idx, level in eq_lows_levels:
        x0 = df.index[start_idx]
        x1 = df.index[end_idx]
        fig.add_shape(type='line',
                      x0=x0, x1=x1,
                      y0=level, y1=level,
                      line=dict(color='lime', width=2))

    # plot equal highs horizontal lines
    for start_idx, end_idx, level in eq_highs_levels:
        x0 = df.index[start_idx]
        x1 = df.index[end_idx]
        fig.add_shape(type='line',
                      x0=x0, x1=x1,
                      y0=level, y1=level,
                      line=dict(color='orange', width=2))

    fig.update_layout(
        title=f"{ticker} price with ICT equal highs/lows",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    with cols[idx % len(cols)]:
        st.plotly_chart(fig, use_container_width=True)
        found_setups = True

if not found_setups:
    st.info("no setups found in selected tickers/timeframe. try adjusting date range or tickers.")
