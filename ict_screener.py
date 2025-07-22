import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

TIMEFRAMES = {
    "1Y": ("1d", 1),  # fetch daily, resample to yearly
}

def fetch_yearly_data(ticker):
    try:
        df = yf.download(ticker, period="10y", interval="1d", progress=False)
        if df.empty:
            st.write(f"no daily data for {ticker}")
            return None
        df.index = pd.to_datetime(df.index)
        yearly = df.resample('Y').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        yearly.dropna(inplace=True)
        yearly.reset_index(inplace=True)
        if yearly.empty:
            st.write(f"no yearly data after resampling for {ticker}")
            return None
        return yearly
    except Exception as e:
        st.write(f"error fetching {ticker}: {e}")
        return None

def find_equals(df, price_col='Low', tol=0.02):
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

def find_fvgs(df):
    fvg_list = []
    for i in range(2, len(df)):
        high0, low0 = df.loc[i-2, 'High'], df.loc[i-2, 'Low']
        high1, low1 = df.loc[i-1, 'High'], df.loc[i-1, 'Low']
        high2, low2 = df.loc[i, 'High'], df.loc[i, 'Low']

        # bearish FVG
        if low0 > high2:
            fvg_list.append(('bear', i-2, i, high2, low0))
        # bullish FVG
        elif high0 < low2:
            fvg_list.append(('bull', i-2, i, high0, low2))
    return fvg_list

def plot_candles_with_equals_fvg(df, equals_highs=None, equals_lows=None, fvg_list=None, title=""):
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

    # plot equals highs
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
    # plot equals lows
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

    # plot FVG boxes, extend to right by 5 bars
    if fvg_list:
        for fvg_type, start_idx, end_idx, low_lvl, high_lvl in fvg_list:
            # convert idx to x axis value
            start_x = x_vals[start_idx]
            end_x = x_vals[end_idx]
            # extend 5 bars to right
            extend_x = x_vals[-1]  # last date
            fig.add_shape(
                type="rect",
                x0=start_x,
                x1=extend_x,
                y0=low_lvl,
                y1=high_lvl,
                xref="x",
                yref="y",
                fillcolor="rgba(255, 0, 0, 0.2)" if fvg_type == 'bear' else "rgba(0, 255, 0, 0.2)",
                line=dict(width=0),
                layer="below"
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

mode = st.radio("select mode", ("single ticker",))

if mode == "single ticker":
    ticker_input = st.text_input("enter ticker symbol", "EOSE").upper()
    if ticker_input:
        df = fetch_yearly_data(ticker_input)
        if df is None:
            st.warning(f"no yearly data for {ticker_input}")
        else:
            df.rename(columns={'Date': 'Date'}, inplace=True)
            equals_highs = find_equals(df, 'High', tol=0.02)
            equals_lows = find_equals(df, 'Low', tol=0.02)
            fvg_list = find_fvgs(df)
            latest_close = df['Close'].iloc[-1]
            # check if price inside any fvg box below price
            inside_fvg = [f for f in fvg_list if latest_close >= f[3] and latest_close <= f[4]]
            if inside_fvg:
                st.success(f"{ticker_input} price is inside a yearly FVG zone.")
            else:
                st.info(f"{ticker_input} price NOT inside any yearly FVG zone.")
            plot_candles_with_equals_fvg(df, equals_highs, equals_lows, fvg_list, title=f"{ticker_input} yearly candles with equals and FVG")
