import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("eose yearly fvg checker with 3-candle logic and extended zones")

def fetch_data(ticker, interval):
    try:
        df = yf.download(ticker, period="10y", interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df.reset_index(inplace=True)
        return df
    except Exception:
        return None

def atr(df, length=28):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(length).mean()
    return atr

def find_fvg(df, atr_length=28, atr_multiplier=1.5):
    fvg_list = []
    atr_values = atr(df, length=atr_length)

    for i in range(2, len(df)-1):  # iterate starting at 2 to access i-2, i-1, i, i+1 safely
        # indices for the three candles forming the potential FVG: i-2, i-1, i
        c0 = df.loc[i-2, 'Close']
        c1 = df.loc[i-1, 'Close']
        c2 = df.loc[i, 'Close']

        h0 = df.loc[i-2, 'High']
        h1 = df.loc[i-1, 'High']
        h2 = df.loc[i, 'High']

        l0 = df.loc[i-2, 'Low']
        l1 = df.loc[i-1, 'Low']
        l2 = df.loc[i, 'Low']

        atr_val = atr_values.iloc[i-1] if i-1 < len(atr_values) else None
        if atr_val is None or pd.isna(atr_val):
            continue

        price_diff = h1 - l1

        # conditions from indicator logic (bear and bull conditions)
        bear_condition = (c0 <= h1) and (c2 <= c1) and (h2 < l1)
        bull_condition = (c0 >= l1) and (c2 >= c1) and (l2 > h1)

        middle_candle_volatility_condition = price_diff > (atr_val * atr_multiplier)

        if (bear_condition or bull_condition) and middle_candle_volatility_condition:
            # for bear FVG: top=low of 3rd candle, bottom=high of middle candle
            # for bull FVG: top=low of middle candle, bottom=high of 3rd candle
            if bull_condition:
                top = l1
                bottom = h2
            else:
                top = l2
                bottom = h1
            fvg_list.append((i-1, i, top, bottom))
    return fvg_list

def price_in_fvg(price, fvg):
    low, high = sorted([fvg[2], fvg[3]])
    return low <= price <= high

def plot_candles_with_fvg(df, fvg_list=None, title=""):
    x_vals = df['Date'].dt.strftime('%Y').tolist()
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

    for fvg in fvg_list:
        bar_start, bar_end, top, bottom = fvg
        low, high = sorted([top, bottom])
        x0 = x_vals[bar_start]
        x1 = x_vals[-1]  # extend to right edge (last x label)
        fig.add_shape(
            type='rect',
            x0=x0,
            x1=x1,
            y0=low,
            y1=high,
            xref='x',
            yref='y',
            fillcolor='rgba(255,165,0,0.3)',
            line=dict(color='rgba(255,165,0,0.5)', width=1),
            layer='below'
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

ticker = "EOSE"
df = fetch_data(ticker, "1y")  # yearly candles
if df is None:
    st.error(f"no data for {ticker}")
else:
    last_close = df['Close'].iloc[-1]
    fvg_list = find_fvg(df)
    inside_fvg = [fvg for fvg in fvg_list if price_in_fvg(last_close, fvg)]
    if inside_fvg:
        st.success(f"{ticker} current price IS inside {len(inside_fvg)} yearly FVG(s)")
    else:
        st.info(f"{ticker} current price NOT inside any yearly FVG")
    plot_candles_with_fvg(df, fvg_list=inside_fvg, title=f"{ticker} yearly chart with extended FVG zones")
