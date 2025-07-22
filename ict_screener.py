import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

def fetch_yearly_resampled(ticker):
    try:
        df = yf.download(ticker, period="max", interval="1mo", progress=False)
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

def find_fvgs(df):
    # FVG logic simplified for yearly resampled data
    fvgs = []
    # need at least 3 candles to form FVG
    if len(df) < 3:
        return fvgs
    for i in range(2, len(df)):
        # candle i-2, i-1, i form the gap check on candle i-1
        high2 = df.loc[i-2, 'High']
        low2 = df.loc[i-2, 'Low']
        high1 = df.loc[i-1, 'High']
        low1 = df.loc[i-1, 'Low']
        high0 = df.loc[i, 'High']
        low0 = df.loc[i, 'Low']

        # bearish FVG condition
        bearish_fvg = (df.loc[i-1, 'Close'] <= high2) and (df.loc[i-1, 'Close'] <= df.loc[i-2, 'Close']) and (high0 < low2)
        # bullish FVG condition
        bullish_fvg = (df.loc[i-1, 'Close'] >= low2) and (df.loc[i-1, 'Close'] >= df.loc[i-2, 'Close']) and (low0 > high2)

        if bearish_fvg or bullish_fvg:
            # store tuple: start_index, end_index, top, bottom, is_bullish
            top = low1 if bullish_fvg else low2
            bottom = high2 if bullish_fvg else high1
            # but better to store top as max, bottom as min of the gap
            top = max(high1, high2)
            bottom = min(low1, low2)
            fvgs.append((i-2, i, bottom, top, bullish_fvg))
    return fvgs

def plot_yearly_with_fvg(df, fvgs, ticker):
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
    # plot FVG boxes extending to the right
    for start_i, end_i, bottom, top, is_bullish in fvgs:
        start_x = x_vals[start_i]
        end_x = x_vals[-1]  # extend all the way to last candle

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=start_x,
            y0=bottom,
            x1=end_x,
            y1=top,
            fillcolor="rgba(0,255,0,0.2)" if is_bullish else "rgba(255,0,0,0.2)",
            line=dict(width=0),
            layer="below"
        )

    fig.update_layout(
        title=f"{ticker} yearly chart with FVG zones",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- main ---

ticker_input = st.text_input("enter ticker symbol (for yearly FVG scan)", "EOSE").upper()

if ticker_input:
    df_yearly = fetch_yearly_resampled(ticker_input)
    if df_yearly is None or df_yearly.empty:
        st.warning(f"no yearly data available for {ticker_input}")
    else:
        fvgs = find_fvgs(df_yearly)
        last_close = df_yearly['Close'].iloc[-1]

        # check if current price is inside any FVG zone below price
        inside_fvg = False
        for _, _, bottom, top, _ in fvgs:
            if bottom < last_close < top:
                inside_fvg = True
                break

        if inside_fvg:
            st.success(f"{ticker_input} price is inside a yearly FVG zone")
            plot_yearly_with_fvg(df_yearly, fvgs, ticker_input)
        else:
            st.info(f"no yearly FVG zone contains the current price for {ticker_input}")
            plot_yearly_with_fvg(df_yearly, fvgs, ticker_input)
