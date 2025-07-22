import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import datetime


def fetch_yearly_data(ticker):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365 * 10)
    df = yf.download(ticker, start=start_date, end=end_date, interval='1mo', auto_adjust=True)
    if df.empty or not set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns):
        st.write(f"invalid or missing data for {ticker}")
        return None

    df['Year'] = df.index.year
    yearly_data = df.groupby('Year').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    yearly_data.dropna(inplace=True)
    yearly_data.reset_index(inplace=True)
    return yearly_data


def find_fvg_zones(df):
    zones = []
    for i in range(1, len(df) - 1):
        prev_high = df.loc[i - 1, 'High']
        prev_low = df.loc[i - 1, 'Low']
        next_high = df.loc[i + 1, 'High']
        next_low = df.loc[i + 1, 'Low']

        curr_open = df.loc[i, 'Open']
        curr_close = df.loc[i, 'Close']

        upper = min(prev_high, next_high)
        lower = max(prev_low, next_low)

        if lower < curr_low and upper > curr_high:
            zones.append((df.loc[i, 'Year'], lower, upper))

    return zones


def is_price_inside_any_fvg(df, zones):
    if df.empty:
        return False
    last_close = df['Close'].iloc[-1]
    for zone in zones:
        lower, upper = zone[1], zone[2]
        if lower <= last_close <= upper:
            return zone
    return None


def plot_chart_with_fvg(df, fvg_zone, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=df['Year'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    if fvg_zone:
        fvg_year, fvg_low, fvg_high = fvg_zone
        fig.add_shape(
            type='rect',
            x0=fvg_year,
            x1=df['Year'].max(),
            y0=fvg_low,
            y1=fvg_high,
            line=dict(color='blue', width=1),
            fillcolor='blue',
            opacity=0.2
        )

    fig.update_layout(title=f"{ticker} with Yearly FVG Highlighted", xaxis_title="Year", yaxis_title="Price")
    st.plotly_chart(fig)


def main():
    st.title("FVG Screener - Yearly Timeframe")

    ticker = st.text_input("Enter ticker (e.g. EOSE)", "EOSE")
    df = fetch_yearly_data(ticker)
    if df is None:
        return

    zones = find_fvg_zones(df)
    match = is_price_inside_any_fvg(df, zones)

    if match:
        st.write(f"{ticker} is currently inside a yearly FVG: {match}")
        plot_chart_with_fvg(df, match, ticker)
    else:
        st.write(f"{ticker} is NOT currently inside any yearly FVG.")
        plot_chart_with_fvg(df, None, ticker)


if __name__ == "__main__":
    main()
