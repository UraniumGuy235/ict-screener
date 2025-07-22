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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df.reset_index(inplace=True)  # keep Date as column
        return df
    except Exception as e:
        st.error(f"error fetching data for {ticker}: {e}")
        return None

def find_equal_levels(df, price_col='Low', tol=0.02):
    # find pairs of candles that have lows/highs within tol and no intervening breaks above/below
    levels = []
    n = len(df)
    for i in range(n):
        base_price = df.loc[i, price_col]
        for j in range(i+1, n):
            test_price = df.loc[j, price_col]
            if abs(test_price - base_price) / base_price <= tol:
                # check no breaks between i and j
                inter_slice = df.loc[i+1:j-1] if j - i > 1 else pd.DataFrame()
                if price_col == 'Low':
                    if not inter_slice.empty and (inter_slice['Low'] < min(base_price, test_price)).any():
                        continue
                else:
                    if not inter_slice.empty and (inter_slice['High'] > max(base_price, test_price)).any():
                        continue
                levels.append((i, j, (base_price + test_price)/2))
            else:
                # break early because prices diverge
                if price_col == 'Low' and test_price > base_price * (1 + tol):
                    break
                if price_col == 'High' and test_price < base_price * (1 - tol):
                    break
    # remove duplicates, keep unique pairs only
    unique_levels = []
    seen = set()
    for s, e, lvl in levels:
        if (s, e) not in seen and (e, s) not in seen:
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

    eq_lows_levels = find_equal_levels(df, price_col='Low', tol=0.005)
    eq_highs_levels = find_equal_levels(df, price_col='High', tol=0.005)

    # x axis as category to skip non trading days
    x_vals = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()

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

    # draw equals lows lines (solid lime)
    for start_idx, end_idx, level in eq_lows_levels:
        fig.add_shape(
            type='line',
            x0=start_idx, x1=end_idx,
            y0=level, y1=level,
            xref='x', yref='y',
            line=dict(color='lime', width=3, dash='solid')
        )

    # draw equals highs lines (solid orange)
    for start_idx, end_idx, level in eq_highs_levels:
        fig.add_shape(
            type='line',
            x0=start_idx, x1=end_idx,
            y0=level, y1=level,
            xref='x', yref='y',
            line=dict(color='orange', width=3, dash='solid')
        )

    fig.update_layout(
        title=f"{ticker} price with ICT equal highs/lows",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='category',
            tickangle=-45,
            tickmode='auto',
            tickfont=dict(size=10)
        )
    )

    with cols[idx % len(cols)]:
        st.plotly_chart(fig, use_container_width=True)
        found_setups = True

if not found_setups:
    st.info("no setups found in selected tickers/timeframe. try adjusting date range or tickers.")
