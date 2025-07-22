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
    # bullish fair value gap: low of bar i-1 > high of bar i-2
    df['bullish_fvg'] = df['Low'].shift(1) > df['High'].shift(2)
    return df

def find_equal_lows(df, tol=0.01):  # 1% tolerance
    lows = df['Low'].rolling(3).apply(
        lambda x: abs(x[0] - x[1]) / x[1] < tol and abs(x[1] - x[2]) / x[1] < tol,
        raw=True)
    return lows.fillna(0).astype(bool)

def find_equal_highs(df, tol=0.01):  # 1% tolerance
    highs = df['High'].rolling(3).apply(
        lambda x: abs(x[0] - x[1]) / x[1] < tol and abs(x[1] - x[2]) / x[1] < tol,
        raw=True)
    return highs.fillna(0).astype(bool)

if st.button("Fetch and Plot"):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)

    df = detect_fvg(df)
    df['eq_lows'] = find_equal_lows(df)
    df['eq_highs'] = find_equal_highs(df)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'
    )])

    # plot bullish FVG zones as translucent green rectangles between bars i-2 and i-1
    fvg_bars = df[df['bullish_fvg']]
    for i in fvg_bars.index:
        i_minus_1 = df.index.get_loc(i) - 1
        i_minus_2 = df.index.get_loc(i) - 2
        if i_minus_1 < 0 or i_minus_2 < 0:
            continue
        x0 = df.index[i_minus_2]
        x1 = df.index[i_minus_1]
        y0 = df.loc[x1, 'High']
        y1 = df.loc[x0, 'Low']
        fig.add_shape(type="rect",
                      x0=x0, x1=x1,
                      y0=y0, y1=y1,
                      fillcolor="rgba(0,255,0,0.2)", line_width=0)

    # equal lows lines connecting the two lows forming the pattern
    eq_lows_idx = df.index[df['eq_lows']]
    for idx in eq_lows_idx:
        pos = df.index.get_loc(idx)
        if pos >= 2:
            x0 = df.index[pos-2]
            x1 = df.index[pos-1]
            y0 = df.loc[x0, 'Low']
            y1 = df.loc[x1, 'Low']
            fig.add_shape(type="line",
                          x0=x0, x1=x1,
                          y0=y0, y1=y1,
                          line=dict(color='lime', width=2))

    # equal highs lines connecting the two highs forming the pattern
    eq_highs_idx = df.index[df['eq_highs']]
    for idx in eq_highs_idx:
        pos = df.index.get_loc(idx)
        if pos >= 2:
            x0 = df.index[pos-2]
            x1 = df.index[pos-1]
            y0 = df.loc[x0, 'High']
            y1 = df.loc[x1, 'High']
            fig.add_shape(type="line",
                          x0=x0, x1=x1,
                          y0=y0, y1=y1,
                          line=dict(color='orange', width=2))

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
