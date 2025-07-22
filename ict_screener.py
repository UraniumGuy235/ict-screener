import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.title("FVG Screener - 1Y Candles")

# === functions ===

def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="max", interval="1d")
        if data.empty:
            return None
        data = data.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low',
            'Close': 'Close', 'Adj Close': 'Adj Close', 'Volume': 'Volume'
        })
        expected = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not expected.issubset(data.columns):
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except Exception:
        return None

def resample_to_yearly(df):
    return df.resample('1Y').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

def find_fvg_zones(df):
    fvg_zones = []
    for i in range(2, len(df)):
        prev = df.iloc[i - 2]
        curr = df.iloc[i]
        if curr['Low'] > prev['High']:
            fvg_zones.append({
                'start_idx': i - 2,
                'end_idx': i,
                'high': prev['High'],
                'low': curr['Low']
            })
    return fvg_zones

def price_inside_any_fvg(df, fvg_zones):
    last_close = df.iloc[-1]['Close']
    for zone in fvg_zones:
        if zone['low'] <= last_close <= zone['high']:
            return zone
    return None

def plot_chart(df, fvg_zones, inside_zone=None, title="FVG Zones - 1Y Candles"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Candles"
    ))

    for zone in fvg_zones:
        color = 'rgba(255,165,0,0.3)' if zone == inside_zone else 'rgba(128,128,128,0.2)'
        fig.add_shape(
            type="rect",
            x0=df.index[zone['start_idx']],
            x1=df.index[-1],
            y0=zone['low'], y1=zone['high'],
            fillcolor=color,
            line=dict(width=0),
            layer="below"
        )

    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# === app logic ===

ticker = st.text_input("Enter Ticker (e.g., EOSE)")

if ticker:
    df = fetch_data(ticker.upper())
    if df is None:
        st.error("invalid or missing data")
    else:
        yearly_df = resample_to_yearly(df)
        fvg_zones = find_fvg_zones(yearly_df)
        inside_zone = price_inside_any_fvg(yearly_df, fvg_zones)
        if inside_zone:
            st.success(f"{ticker.upper()} is inside an FVG zone")
        else:
            st.warning("no FVG zone contains the current price")
        plot_chart(yearly_df, fvg_zones, inside_zone)
