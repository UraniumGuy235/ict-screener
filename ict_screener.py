import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide")

# --- TIMEFRAME MAPPING ---
TIMEFRAMES = {
    "1M": "1mo",
    "1W": "1wk",
    "1D": "1d",
    "1H": "60m",
    "3M": "3mo",
    "6M": "6mo"
}

# --- FETCH DATA ---
def fetch_data(ticker, interval):
    try:
        data = yf.download(ticker, period="5y", interval=interval, auto_adjust=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# --- FIND EQUALS ---
def find_equals(df, tolerance=0.01):
    equals = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if df.index[i] >= df.index[j]:
                continue
            subset = df.iloc[i+1:j]
            max_high = subset['High'].max() if not subset.empty else float('-inf')
            min_low = subset['Low'].min() if not subset.empty else float('inf')

            hi1, hi2 = df['High'].iloc[i], df['High'].iloc[j]
            lo1, lo2 = df['Low'].iloc[i], df['Low'].iloc[j]

            if abs(hi1 - hi2) <= tolerance and max_high < min(hi1, hi2):
                equals.append(('high', i, j, hi1))
            elif abs(lo1 - lo2) <= tolerance and min_low > max(lo1, lo2):
                equals.append(('low', i, j, lo1))
    return equals

# --- PLOT CHART ---
def plot_chart(df, equals, title):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'))

    for eq in equals:
        kind, i, j, level = eq
        fig.add_trace(go.Scatter(
            x=[df.index[i], df.index[j]],
            y=[level, level],
            mode="lines",
            line=dict(color="cyan" if kind=="high" else "magenta", width=1),
            name=f"Equals {kind.capitalize()}"
        ))

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        dragmode='pan',
    )

    return fig

# --- MAIN SCREENING ---
def main():
    st.title("ICT Screener")

    sample_tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "GOOGL", "NFLX", "AMD", "INTC"]

    selected_interval = "1M"  # fixed
    bullish_setups = []
    bearish_setups = []

    for ticker in sample_tickers:
        df = fetch_data(ticker, TIMEFRAMES[selected_interval])
        if df is None or len(df) < 10:
            continue

        equals = find_equals(df, tolerance=0.01)
        if not equals:
            continue

        current_price = df['Close'].iloc[-1]
        highs_above = [eq for eq in equals if eq[0] == 'high' and eq[3] > current_price]
        lows_below = [eq for eq in equals if eq[0] == 'low' and eq[3] < current_price]

        if highs_above and not lows_below:
            bullish_setups.append((ticker, df, equals))
        elif lows_below and not highs_above:
            bearish_setups.append((ticker, df, equals))

    st.subheader("Bullish Setups")
    for ticker, df, equals in bullish_setups:
        st.plotly_chart(plot_chart(df, equals, f"Bullish: {ticker}"), use_container_width=True)

    st.subheader("Bearish Setups")
    for ticker, df, equals in bearish_setups:
        st.plotly_chart(plot_chart(df, equals, f"Bearish: {ticker}"), use_container_width=True)

    st.subheader("Custom Chart")
    custom_ticker = st.text_input("Enter Ticker", value="AAPL")
    custom_interval = st.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=0)

    if st.button("Fetch Custom Chart"):
        df = fetch_data(custom_ticker, TIMEFRAMES[custom_interval])
        if df is not None:
            equals = find_equals(df, tolerance=0.01)
            st.plotly_chart(plot_chart(df, equals, f"Custom Chart: {custom_ticker}"), use_container_width=True)
        else:
            st.error("Failed to fetch data.")

if __name__ == '__main__':
    main()
