import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict fvg yearly screener + single ticker viewer")

def fetch_yearly_data(ticker):
    try:
        df = yf.download(ticker, period="10y", interval="1mo", progress=False)
        if df.empty:
            df = yf.download(ticker, period="max", interval="1mo", progress=False)
            if df.empty:
                return None
        df = df.resample('Y').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        df.reset_index(inplace=True)
        return df
    except Exception:
        return None

def find_fvgs(df, atr_length=28, atr_mult=1.5):
    # identify FVG zones based on 3 candle pattern, yearly data
    fvg_zones = []
    # ATR calc
    df['H-L'] = df['High'] - df['Low']
    df['TR'] = df[['High', 'Close'].shift(1)].max(axis=1) - df[['Low', 'Close'].shift(1)].min(axis=1)
    df['TR'] = df['TR'].fillna(df['H-L'])
    df['ATR'] = df['TR'].rolling(atr_length).mean()
    for i in range(2, len(df)-1):
        c0 = df.iloc[i-2]
        c1 = df.iloc[i-1]
        c2 = df.iloc[i]
        price_diff = c1['High'] - c1['Low']
        atr_val = df['ATR'].iloc[i-1]
        if pd.isna(atr_val):
            continue
        vol_cond = price_diff > atr_val * atr_mult
        # Bearish FVG
        bear_cond = (c0['Close'] <= c1['High']) and (c2['Close'] <= c1['Close']) and (c2['High'] < c1['Low'])
        # Bullish FVG
        bull_cond = (c0['Close'] >= c1['Low']) and (c2['Close'] >= c1['Close']) and (c2['Low'] > c1['High'])
        if vol_cond and (bear_cond or bull_cond):
            # box from c1 candle top to bottom
            top = c1['Low'] if bull_cond else c1['High']
            bottom = c1['High'] if bull_cond else c1['Low']
            # normalize top < bottom
            high_val = max(c1['High'], c1['Low'])
            low_val = min(c1['High'], c1['Low'])
            # we want the gap between candles, so use the upper low and lower high of middle candle
            if bull_cond:
                fvg_zones.append({'start': i-1, 'end': i, 'low': bottom, 'high': top, 'type': 'bullish'})
            else:
                fvg_zones.append({'start': i-1, 'end': i, 'low': top, 'high': bottom, 'type': 'bearish'})
    return fvg_zones

def price_in_fvg(price, fvg):
    return fvg['low'] <= price <= fvg['high']

def plot_yearly_candles_with_fvg(df, fvgs, ticker):
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
    # add FVG zones as rectangles extending right
    max_x = len(x_vals) - 1
    for fvg in fvgs:
        start_idx = fvg['start']
        end_idx = fvg['end']
        low = fvg['low']
        high = fvg['high']
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x_vals[start_idx],
            x1=x_vals[max_x],
            y0=low,
            y1=high,
            fillcolor="green" if fvg['type'] == 'bullish' else 'red',
            opacity=0.3,
            line_width=0,
            layer="below"
        )
        # highlight the candle range forming the FVG with border
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x_vals[start_idx],
            x1=x_vals[end_idx],
            y0=low,
            y1=high,
            line=dict(color="green" if fvg['type']=='bullish' else 'red', width=2),
            fillcolor="rgba(0,0,0,0)"
        )
    fig.update_layout(
        title=f"{ticker} yearly candles with FVG zones",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickangle=-45, type='category')
    )
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.title("ticker input")
ticker_input = st.sidebar.text_input("enter ticker", "EOSE").upper()

if ticker_input:
    df = fetch_yearly_data(ticker_input)
    if df is None or df.empty:
        st.error(f"no data for {ticker_input}")
    else:
        fvgs = find_fvgs(df)
        last_close = df['Close'].iloc[-1]
        fvg_below = [f for f in fvgs if price_in_fvg(last_close, f) and f['low'] < last_close]
        if fvg_below:
            st.success(f"{ticker_input} price inside yearly FVG below current price")
            plot_yearly_candles_with_fvg(df, fvg_below, ticker_input)
        else:
            st.info("no yearly FVG below current price containing price found")
