import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🌑 ICT Setup Candlestick Chart")

ticker = st.text_input("Enter ticker (e.g., AAPL, MSFT):", "AAPL")
period = st.selectbox("Select historical period:", ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'], index=3)
interval = st.selectbox("Select interval:", ['1d', '1h', '30m', '15m'], index=0)

def detect_fvg(df):
    # bullish fair value gap: low of prev bar > high of bar before that
    df['bullish_fvg'] = df['Low'].shift(1) > df['High'].shift(2)
    return df

def find_equal_lows(df, tol=0.001):
    # rolling 3 bars: lows roughly equal within tolerance
    lows = df['Low'].rolling(3).apply(lambda x: abs(x[0] - x[1]) < tol and abs(x[1] - x[2]) < tol, raw=True)
    return lows.fillna(0).astype(bool)

def find_equal_highs(df, tol=0.001):
    highs = df['High'].rolling(3).apply(lambda x: abs(x[0] - x[1]) < tol and abs(x[1] - x[2]) < tol, raw=True)
    return highs.fillna(0).astype(bool)

def open_confluence(df):
    # open within 0.1 of rolling mean of last 3 bars' opens
    return df['Open'].rolling(3).apply(lambda x: abs(x[0] - x.mean()) < 0.1, raw=True).fillna(0).astype(bool)

if st.button("Fetch and Plot"):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)

    # detect ict setups
    df = detect_fvg(df)
    df['eq_lows'] = find_equal_lows(df)
    df['eq_highs'] = find_equal_highs(df)
    df['open_confluence'] = open_confluence(df)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='cyan',
        decreasing_line_color='magenta',
        name='Price'
    )])

    # plot bullish FVG zones as translucent rectangles
    fvg_bars = df[df['bullish_fvg']]
    for i in fvg_bars.index:
        # the FVG is between bar i-2 high and bar i-1 low
        high = df.loc[i - pd.Timedelta(days=1)]['High'] if (i - pd.Timedelta(days=1)) in df.index else None
        low = df.loc[i - pd.Timedelta(days=2)]['Low'] if (i - pd.Timedelta(days=2)) in df.index else None
        if high is None or low is None:
            continue
        fig.add_shape(type="rect",
                      x0=i - pd.Timedelta(days=2), x1=i - pd.Timedelta(days=1),
                      y0=high, y1=low,
                      fillcolor="green", opacity=0.2, line_width=0)

    # plot equal lows and highs as horizontal lines spanning full x-axis range
    eq_lows_vals = df.loc[df['eq_lows'], 'Low'].unique()
    for lvl in eq_lows_vals:
        fig.add_hline(y=lvl, line=dict(color='lime', width=1, dash='dash'), annotation_text='Equal Lows', annotation_position="bottom left")

    eq_highs_vals = df.loc[df['eq_highs'], 'High'].unique()
    for lvl in eq_highs_vals:
        fig.add_hline(y=lvl, line=dict(color='orange', width=1, dash='dash'), annotation_text='Equal Highs', annotation_position="top left")

    # plot open confluence as blue markers on candles
    confluence_idx = df.index[df['open_confluence']]
    fig.add_trace(go.Scatter(
        x=confluence_idx,
        y=df.loc[confluence_idx, 'Open'],
        mode='markers',
        marker=dict(color='blue', size=8, symbol='circle'),
        name='Open Confluence'
    ))

    fig.update_layout(
        template='plotly_dark',
        title=f'{ticker} ICT Setups',
        xaxis_rangeslider_visible=True,
        xaxis_title='Date',
        yaxis_title='Price',
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True)
