import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

def fetch_data(ticker, period="5y", interval="1mo"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def find_equals(df, tolerance=0.01):
    if not {'High', 'Low'}.issubset(df.columns):
        return []
    
    equals = []
    highs = df['High']
    lows = df['Low']
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            hi1, hi2 = highs[i], highs[j]
            low1, low2 = lows[i], lows[j]
            if abs(hi1 - hi2) <= tolerance:
                max_high = df['High'][i+1:j].max()
                if pd.isna(max_high) or max_high < min(hi1, hi2):
                    equals.append(("high", df.index[i], df.index[j], float((hi1 + hi2) / 2)))
            if abs(low1 - low2) <= tolerance:
                min_low = df['Low'][i+1:j].min()
                if pd.isna(min_low) or min_low > max(low1, low2):
                    equals.append(("low", df.index[i], df.index[j], float((low1 + low2) / 2)))
    return equals

def plot_chart(df, equals, ticker):
    fig = go.Figure()

    colors = ['green' if c >= o else 'red' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ))

    for kind, i1, i2, level in equals:
        color = 'lime' if kind == "high" else 'red'
        fig.add_trace(go.Scatter(
            x=[i1, i2],
            y=[level, level],
            mode='lines',
            line=dict(color=color, width=2),
            name=f"{kind.upper()} Equal"
        ))

    fig.update_layout(
        title=ticker,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(type='category')  # skips missing trading days
    )
    st.plotly_chart(fig, use_container_width=True)

def analyze_ticker(ticker, interval="1mo"):
    df = fetch_data(ticker, interval=interval)
    if df.empty or not {'High', 'Low', 'Close'}.issubset(df.columns):
        return None, None, (False, False)

    equals = find_equals(df, tolerance=0.01)
    latest_close = float(df['Close'].iloc[-1])
    has_bullish = any(t == 'high' and float(lvl) > latest_close for t, _, _, lvl in equals)
    has_bearish = any(t == 'low' and float(lvl) < latest_close for t, _, _, lvl in equals)

    return df, equals, (has_bullish, has_bearish)

def main():
    st.title("ICT Screener - Monthly Equals Focus")

    # search a specific stock
    with st.sidebar:
        st.subheader("Search Single Stock")
        single = st.text_input("Ticker", "AAPL")
        interval = st.selectbox("Interval", ["6mo", "3mo", "1mo", "1wk", "1d", "1h"], index=2)
        if st.button("Analyze"):
            df, equals, _ = analyze_ticker(single, interval)
            if df is not None:
                plot_chart(df, equals, single)
            else:
                st.error("No data found.")

    # screen for bullish setups
    st.subheader("Bullish Monthly Setups (Equals ABOVE price)")

    tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'NVDA', 'TSLA', 'AMZN', 'AMD', 'NFLX', 'INTC', 
               'BA', 'GE', 'LMT', 'JPM', 'BAC', 'XOM', 'CVX', 'WMT', 'TGT', 'UNH', 'JNJ', 'PFE']
    bullish = []
    bearish = []

    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        df, equals, (has_bullish, has_bearish) = analyze_ticker(ticker)
        if df is None:
            continue
        if has_bullish:
            bullish.append((ticker, df, equals))
        elif has_bearish:
            bearish.append((ticker, df, equals))
        progress.progress((i + 1) / len(tickers))

    st.subheader("📈 Bullish Charts")
    for ticker, df, equals in bullish[:3]:
        st.markdown(f"### {ticker}")
        plot_chart(df, equals, ticker)

    st.subheader("📉 Bearish Charts")
    for ticker, df, equals in bearish[:3]:
        st.markdown(f"### {ticker}")
        plot_chart(df, equals, ticker)

if __name__ == "__main__":
    main()
