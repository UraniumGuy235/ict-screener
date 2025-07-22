import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
from streamlit.components.v1 import html

st.set_page_config(layout="wide")

# === indicator logic ===
def find_equals(df, tolerance=0.015):
    highs = df['High']
    lows = df['Low']
    equals = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            hi1, hi2 = highs.iloc[i], highs.iloc[j]
            lo1, lo2 = lows.iloc[i], lows.iloc[j]
            max_high = highs.iloc[i+1:j].max() if j > i+1 else 0
            min_low = lows.iloc[i+1:j].min() if j > i+1 else float('inf')

            if abs(hi1 - hi2) <= tolerance and max_high < min(hi1, hi2):
                equals.append(('high', df.index[i], df.index[j], min(hi1, hi2)))
            elif abs(lo1 - lo2) <= tolerance and min_low > max(lo1, lo2):
                equals.append(('low', df.index[i], df.index[j], max(lo1, lo2)))
    return equals

# === FVG logic ===
def find_fvg(df):
    fvg_zones = []
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            fvg_zones.append((df.index[i-2], df.index[i], df['High'].iloc[i-2], df['Low'].iloc[i]))
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            fvg_zones.append((df.index[i-2], df.index[i], df['Low'].iloc[i-2], df['High'].iloc[i]))
    return fvg_zones

# === chart drawing ===
def plot_chart(df, equals, fvg):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 increasing_line_color='green', decreasing_line_color='red'))

    for t, i1, i2, level in equals:
        color = 'red' if t == 'high' else 'green'
        fig.add_trace(go.Scatter(x=[i1, i2], y=[level, level], mode='lines',
                                 line=dict(color=color, width=2), name=f'Equal {t.capitalize()}'))

    for start, end, hi, lo in fvg:
        fig.add_shape(type="rect",
                      x0=start, x1=end,
                      y0=lo, y1=hi,
                      fillcolor="rgba(255,165,0,0.3)",
                      line=dict(width=0))

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
    return fig

# === main analysis ===
def analyze_ticker(ticker):
    df = yf.download(ticker, period='5y', interval='1mo', progress=False)
    df = df[df['Volume'] > 0]  # drop empty rows

    equals = find_equals(df)
    fvg = find_fvg(df)
    
    has_bullish = any(t == 'high' and lvl > df['Close'].iloc[-1] for t, _, _, lvl in equals)
    has_bearish = any(t == 'low' and lvl < df['Close'].iloc[-1] for t, _, _, lvl in equals)
    return df, equals, fvg, has_bullish, has_bearish

# === layout ===
col1, col2 = st.columns([3,2])

with col1:
    st.title("ICT Screener App")

    ticker_input = st.text_input("Enter a ticker symbol", value="AAPL")
    if st.button("Analyze"):
        df, equals, fvg, bullish, bearish = analyze_ticker(ticker_input)
        fig = plot_chart(df, equals, fvg)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("TradingView Chart")
    tradingview_ticker = st.text_input("Enter ticker for TradingView embed", value="AAPL")
    tv_embed_code = f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{tradingview_ticker}&symbol={tradingview_ticker}&interval=D&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc/UTC&withdateranges=1&hidevolume=0&studies_overrides=&overrides=&enabled_features=&disabled_features=" width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """
    html(tv_embed_code, height=500)

st.markdown("---")
st.markdown("**Note**: Chart includes ICT FVG zones and Equal Highs/Lows. TradingView widget shows live data.")
