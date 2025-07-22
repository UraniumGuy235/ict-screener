import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict fvg yearly screener - single ticker viewer")

def fetch_yearly_data(ticker):
    try:
        df = yf.download(ticker, period="10y", interval="1d", progress=False)
        if df.empty:
            st.write(f"no daily data for {ticker}")
            return None
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.write(f"missing columns in data for {ticker}")
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

def find_fvgs(df):
    # fvg needs 3 candles: check for gap between candle 1 and 3 w/ candle 2 inside gap
    fvgs = []
    n = len(df)
    for i in range(n - 2):
        c1 = df.iloc[i]
        c2 = df.iloc[i + 1]
        c3 = df.iloc[i + 2]
        # bullish FVG: gap between low of c3 and high of c1, c2 low inside gap
        if c3['Low'] > c1['High']:
            if c2['Low'] > c1['High'] and c2['High'] < c3['Low']:
                fvgs.append({
                    'start': i,
                    'end': i + 2,
                    'top': c3['Low'],
                    'bottom': c1['High']
                })
        # bearish FVG: gap between low of c1 and high of c3, c2 high inside gap
        elif c1['Low'] > c3['High']:
            if c2['High'] < c1['Low'] and c2['Low'] > c3['High']:
                fvgs.append({
                    'start': i,
                    'end': i + 2,
                    'top': c1['Low'],
                    'bottom': c3['High']
                })
    return fvgs

def price_in_fvg(price, fvg):
    return fvg['bottom'] <= price <= fvg['top']

def plot_yearly_chart(df, fvgs, price, ticker):
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
    for fvg in fvgs:
        fig.add_shape(
            type='rect',
            x0=fvg['start'], x1=len(df) - 1,  # extend to right edge
            y0=fvg['bottom'], y1=fvg['top'],
            xref='x', yref='y',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0),
            layer='below'
        )
        # mark start and end candle tops
        fig.add_shape(
            type='line',
            x0=fvg['start'], x1=fvg['end'],
            y0=fvg['top'], y1=fvg['top'],
            xref='x', yref='y',
            line=dict(color='red', width=2, dash='solid')
        )
        fig.add_shape(
            type='line',
            x0=fvg['start'], x1=fvg['end'],
            y0=fvg['bottom'], y1=fvg['bottom'],
            xref='x', yref='y',
            line=dict(color='red', width=2, dash='solid')
        )
    fig.update_layout(
        title=f"{ticker} yearly chart with FVG zones (price {price})",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    ticker_input = st.text_input("enter ticker symbol for yearly fvg scan", "EOSE").upper()
    if not ticker_input:
        st.warning("enter ticker symbol")
        return

    df = fetch_yearly_data(ticker_input)
    if df is None:
        st.warning(f"no yearly data for {ticker_input}")
        return

    latest_close = df['Close'].iloc[-1]
    fvgs = find_fvgs(df)
    price_inside_fvgs = [fvg for fvg in fvgs if price_in_fvg(latest_close, fvg)]

    if not price_inside_fvgs:
        st.info(f"no yearly FVGs found containing current price ({latest_close}) for {ticker_input}")
        return

    st.success(f"found {len(price_inside_fvgs)} yearly FVG(s) containing current price {latest_close}")
    plot_yearly_chart(df, price_inside_fvgs, latest_close, ticker_input)

if __name__ == "__main__":
    main()
