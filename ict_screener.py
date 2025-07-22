import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict fvg yearly screener + single ticker viewer")

def fetch_yearly_data(ticker):
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        yearly = df.resample('Y').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        yearly.reset_index(inplace=True)
        return yearly
    except Exception:
        return None

def find_fvg_yearly(df):
    # fvg logic: check for fair value gaps from 3 consecutive candles (yearly candles here)
    # fvg when middle candle's range doesn't overlap with previous or next candle's body
    fvg_zones = []
    for i in range(1, len(df)-1):
        prev = df.iloc[i-1]
        mid = df.iloc[i]
        nxt = df.iloc[i+1]

        # bearish FVG: middle candle high < prev candle low and next candle low
        bearish_gap = mid['High'] < prev['Low'] and mid['High'] < nxt['Low']
        # bullish FVG: middle candle low > prev candle high and next candle high
        bullish_gap = mid['Low'] > prev['High'] and mid['Low'] > nxt['High']

        if bearish_gap or bullish_gap:
            fvg_zones.append({
                'start_idx': i-1,
                'end_idx': i+1,
                'top': max(prev['High'], mid['High'], nxt['High']),
                'bottom': min(prev['Low'], mid['Low'], nxt['Low'])
            })
    return fvg_zones

def price_inside_fvg(price, fvg_zone):
    return fvg_zone['bottom'] <= price <= fvg_zone['top']

def plot_yearly_fvg(df, fvg_zones, ticker):
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

    # plot fvg zones as rectangles extending rightwards
    for zone in fvg_zones:
        start = zone['start_idx']
        end = zone['end_idx']
        top = zone['top']
        bottom = zone['bottom']

        # draw rectangle from start bar to extend 3 more bars right
        fig.add_shape(
            type='rect',
            xref='x',
            yref='y',
            x0=x_vals[start],
            y0=bottom,
            x1=x_vals[end],
            y1=top,
            fillcolor='rgba(255, 165, 0, 0.3)',  # orange translucent
            line=dict(width=0),
        )
        # extend the rectangle to the right (by adding an invisible shape beyond last candle)
        fig.add_shape(
            type='rect',
            xref='x',
            yref='y',
            x0=x_vals[end],
            y0=bottom,
            x1=str(pd.to_datetime(x_vals[end]) + pd.DateOffset(years=3)).split(' ')[0],
            y1=top,
            fillcolor='rgba(255, 165, 0, 0.1)',  # lighter orange translucent
            line=dict(width=0),
        )

    fig.update_layout(
        title=f"{ticker} Yearly Candles + FVG Zones",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.header("Select mode")
mode = st.sidebar.radio("mode", ["single ticker", "yearly fvg screener"])

if mode == "single ticker":
    ticker = st.sidebar.text_input("Ticker symbol", "EOSE").upper()
    if ticker:
        df = fetch_yearly_data(ticker)
        if df is None or df.empty:
            st.warning(f"no yearly data for {ticker}")
        else:
            fvg_zones = find_fvg_yearly(df)
            price = df['Close'].iloc[-1]
            inside_fvgs = [z for z in fvg_zones if price_inside_fvg(price, z)]

            st.write(f"current price: {price}")
            if inside_fvgs:
                st.success(f"price is inside {len(inside_fvgs)} yearly FVG zone(s) below")
            else:
                st.info("price not inside any yearly FVG zone")

            plot_yearly_fvg(df, fvg_zones, ticker)

elif mode == "yearly fvg screener":
    # for demo, keep limited tickers here, expand as needed
    sp500_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "BRK-B", "JNJ", "V", "NVDA",
        "JPM", "UNH", "HD", "PG", "BAC",
        "DIS", "MA", "PYPL", "NFLX", "ADBE",
        # ... expand to full S&P 500 as needed
    ]
    found = False
    checked = []
    for ticker in sp500_tickers:
        st.write(f"checking {ticker}...")
        df = fetch_yearly_data(ticker)
        if df is None or df.empty:
            continue
        fvg_zones = find_fvg_yearly(df)
        price = df['Close'].iloc[-1]
        inside_fvgs = [z for z in fvg_zones if price_inside_fvg(price, z)]
        if inside_fvgs:
            st.success(f"{ticker} has price inside {len(inside_fvgs)} yearly FVG zone(s) below")
            plot_yearly_fvg(df, fvg_zones, ticker)
            found = True
            break
        checked.append(ticker)

    if not found:
        st.warning("no stocks found with price inside a yearly FVG below current price yet")

