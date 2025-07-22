import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("eose yearly fvg checker")

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

def find_fvg(df):
    fvg_list = []
    for i in range(3, len(df)):
        c0, c1, c2, c3 = df.loc[i-3, 'Close'], df.loc[i-2, 'Close'], df.loc[i-1, 'Close'], df.loc[i, 'Close']
        h0, h1, h2, h3 = df.loc[i-3, 'High'], df.loc[i-2, 'High'], df.loc[i-1, 'High'], df.loc[i, 'High']
        l0, l1, l2, l3 = df.loc[i-3, 'Low'], df.loc[i-2, 'Low'], df.loc[i-1, 'Low'], df.loc[i, 'Low']
        
        bear_condition = (c0 <= h1) and (c2 <= c1) and (h3 < l1)
        bull_condition = (c0 >= l1) and (c2 >= c1) and (l3 > h1)
        
        if bear_condition or bull_condition:
            top = l3 if bull_condition else l1
            bottom = h1 if bull_condition else l3
            fvg_list.append((i, i-1, top, bottom))
    return fvg_list

def price_in_fvg(price, fvg):
    low, high = sorted([fvg[2], fvg[3]])
    return low <= price <= high

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
        for fvg in fvg_list:
            bar1, bar2, top, bottom = fvg
            low, high = sorted([top, bottom])
            fig.add_shape(
                type='rect',
                x0=bar2,
                x1=bar1,
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
df = fetch_data(ticker, "1mo")  # monthly bars over 1y = yearly timeframe
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
    plot_candles_with_fvg(df, inside_fvg, title=f"{ticker} yearly chart with price inside FVG zones")
