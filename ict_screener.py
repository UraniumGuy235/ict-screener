import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import random

st.set_page_config(layout="wide", page_title="ICT Screener", page_icon="📈")

def fetch_data(ticker, interval="1mo"):
    try:
        df = yf.download(ticker, period="5y", interval=interval, progress=False)
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def find_equals(df, tolerance=0.01):
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

def plot_chart(df, equals):
    fig = go.Figure()

    for i in range(len(df)):
        open_price = df['Open'][i]
        close_price = df['Close'][i]
        color = "green" if close_price >= open_price else "red"
        fig.add_trace(go.Candlestick(
            x=[df.index[i]],
            open=[open_price],
            high=[df['High'][i]],
            low=[df['Low'][i]],
            close=[close_price],
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=False
        ))

    for typ, t1, t2, lvl in equals:
        fig.add_trace(go.Scatter(
            x=[t1, t2],
            y=[lvl, lvl],
            mode="lines",
            line=dict(color="yellow", width=2),
            name=f"{typ.capitalize()} Equal"
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category'),
        margin=dict(l=10, r=10, t=30, b=30)
    )
    return fig

def analyze_ticker(ticker, interval="1mo"):
    df = fetch_data(ticker, interval=interval)
    if df.empty:
        return None, None, False

    equals = find_equals(df, tolerance=0.01)

    latest_close = float(df['Close'].iloc[-1])
    has_bullish = any(t == 'high' and float(lvl) > latest_close for t, _, _, lvl in equals)
    has_bearish = any(t == 'low' and float(lvl) < latest_close for t, _, _, lvl in equals)

    return df, equals, (has_bullish, has_bearish)

def main():
    st.title("📈 ICT Screener - Monthly Highs & Lows")

    tab1, tab2 = st.tabs(["Screener", "Single Ticker View"])

    with tab1:
        stock_list = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'NVDA', 'META', 'AMZN', 'BABA', 'BA', 'NIO',
                      'PLTR', 'AMD', 'INTC', 'GME', 'AMC', 'RIOT', 'MARA', 'F', 'T', 'KO',
                      'PFE', 'XOM', 'CVX', 'WMT', 'JPM', 'GS', 'BAC', 'SOFI', 'BBBY', 'BB', 'TLRY']
        random.shuffle(stock_list)
        st.write("Screening monthly charts for bullish/bearish equal highs/lows...")
        bullish = []
        bearish = []
        for ticker in stock_list[:20]:
            df, equals, (has_bullish, has_bearish) = analyze_ticker(ticker)
            if has_bullish:
                bullish.append((ticker, df, equals))
            elif has_bearish:
                bearish.append((ticker, df, equals))

        st.subheader("📈 Bullish Setups")
        for ticker, df, equals in bullish[:3]:
            st.write(f"**{ticker}**")
            fig = plot_chart(df, equals)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Bearish Setups")
        for ticker, df, equals in bearish[:3]:
            st.write(f"**{ticker}**")
            fig = plot_chart(df, equals)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        ticker = st.text_input("Enter ticker:", value="AAPL")
        interval = st.selectbox("Interval:", ["1mo", "1w", "1d", "1h"])

        if st.button("Fetch Chart"):
            df, equals, _ = analyze_ticker(ticker, interval)
            if df is not None:
                st.plotly_chart(plot_chart(df, equals), use_container_width=True)
            else:
                st.warning("Could not fetch data for this ticker.")

if __name__ == "__main__":
    main()
