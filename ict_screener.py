import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide", page_title="ICT Screener", page_icon="📈")

def fetch_data(ticker, period="5y", interval="1mo"):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def find_equals(df, tolerance=0.01):
    equals = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.index[i] >= df.index[j]:
                continue
            subset = df.iloc[i + 1:j]
            max_high = subset['High'].max() if not subset.empty else float('-inf')
            min_low = subset['Low'].min() if not subset.empty else float('inf')

            hi1 = float(df['High'].iloc[i])
            hi2 = float(df['High'].iloc[j])
            lo1 = float(df['Low'].iloc[i])
            lo2 = float(df['Low'].iloc[j])

            if abs(hi1 - hi2) <= tolerance and max_high < min(hi1, hi2):
                equals.append(('high', i, j, hi1))
            elif abs(lo1 - lo2) <= tolerance and min_low > max(lo1, lo2):
                equals.append(('low', i, j, lo1))
    return equals

def plot_chart(df, equals, title="Chart"):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'
    ))

    for eq in equals:
        typ, i, j, lvl = eq
        color = "lime" if typ == "low" else "magenta"
        fig.add_trace(go.Scatter(
            x=[df.index[i], df.index[j]],
            y=[lvl, lvl],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"Equals {typ}"
        ))

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_rangeslider_visible=False
    )
    return fig

def analyze_ticker(ticker, interval="1mo"):
    df = fetch_data(ticker, interval=interval)
    if df.empty:
        return None, None, False

    equals = find_equals(df, tolerance=0.01)

    latest_close = df['Close'].iloc[-1]
    has_bullish = any(t == 'high' and lvl > latest_close for t, _, _, lvl in equals)
    has_bearish = any(t == 'low' and lvl < latest_close for t, _, _, lvl in equals)

    return df, equals, (has_bullish, has_bearish)

def main():
    st.title("📈 ICT Screener")

    tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOG", "AMD", "META", "AMZN", "BA", "SHOP"]
    selected_ticker = st.text_input("Search a specific ticker", value="AAPL").upper()

    st.subheader("📊 Screened Setups")
    bullish, bearish = [], []

    progress = st.progress(0)
    for idx, ticker in enumerate(tickers):
        df, equals, (has_bullish, has_bearish) = analyze_ticker(ticker)
        if df is not None:
            if has_bullish and not has_bearish:
                bullish.append((ticker, df, equals))
            elif has_bearish and not has_bullish:
                bearish.append((ticker, df, equals))
        progress.progress((idx + 1) / len(tickers))

    st.subheader("🟢 Bullish Setups (Equals Above)")
    for ticker, df, equals in bullish[:3]:
        st.plotly_chart(plot_chart(df, equals, title=f"{ticker} (Bullish)"), use_container_width=True)

    st.subheader("🔴 Bearish Setups (Equals Below)")
    for ticker, df, equals in bearish[:3]:
        st.plotly_chart(plot_chart(df, equals, title=f"{ticker} (Bearish)"), use_container_width=True)

    st.subheader("🔍 Detailed View")
    df, equals, _ = analyze_ticker(selected_ticker)
    if df is not None:
        st.plotly_chart(plot_chart(df, equals, title=f"{selected_ticker} - Detailed"), use_container_width=True)
    else:
        st.warning(f"No data found for {selected_ticker}")

if __name__ == "__main__":
    main()
