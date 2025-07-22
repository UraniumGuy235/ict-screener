import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import random

st.set_page_config(layout="wide")
st.title("automatic ict 6M FVG screener (auto stops on hit)")

TIMEFRAMES = {
    "6M": ("1mo", 3),
}

# bigger universe (100 tickers, mostly US large caps + some tech + energy + financials)
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "DIS", "NFLX", "ADBE", "PYPL", "CSCO", "INTC", "CMCSA", "PEP",
    "KO", "XOM", "BA", "CRM", "ABT", "NKE", "WMT", "T", "CVX", "MCD",
    "COST", "ACN", "IBM", "TXN", "QCOM", "ORCL", "MDT", "AMGN", "HON", "UPS",
    "UNH", "LOW", "CAT", "AXP", "SBUX", "MMM", "GS", "GE", "F", "MO",
    "BLK", "DE", "LMT", "GM", "ZTS", "NOW", "ADI", "SYK", "MDLZ", "PLD",
    "SPGI", "CCI", "CHTR", "FIS", "BKNG", "BDX", "DUK", "C", "ADP", "CSX",
    "SO", "MS", "ETN", "TGT", "VRTX", "CB", "SHW", "USB", "CL", "EMR",
    "ISRG", "DHR", "AON", "EW", "ITW", "REGN", "PNC", "NSC", "EW", "MNST",
    "APD", "MCO", "ICE", "MET", "BSX", "ECL", "SNPS", "DXCM", "HCA", "PGR"
]

def fetch_data(ticker, interval):
    try:
        df = yf.download(ticker, period="1y", interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        return df
    except Exception:
        return None

def find_fvg(df):
    fvg_list = []
    atr = df['High'] - df['Low']
    atr_ma = atr.rolling(28).mean()
    atr_mult = 1.5

    for i in range(3, len(df)):
        if i >= len(atr_ma) or pd.isna(atr_ma.iloc[i-2]):
            continue
        price_diff = df['High'].iloc[i-2] - df['Low'].iloc[i-2]
        middle_candle_vol = price_diff
        if middle_candle_vol <= atr_ma.iloc[i-2] * atr_mult:
            continue

        bear_cond = (df['Close'].iloc[i-3] <= df['High'].iloc[i-2] and
                     df['Close'].iloc[i-1] <= df['Close'].iloc[i-2] and
                     df['High'].iloc[i] < df['Low'].iloc[i-2])

        bull_cond = (df['Close'].iloc[i-3] >= df['Low'].iloc[i-2] and
                     df['Close'].iloc[i-1] >= df['Close'].iloc[i-2] and
                     df['Low'].iloc[i] > df['High'].iloc[i-2])

        if bear_cond or bull_cond:
            is_up_candle = df['Open'].iloc[i-1] <= df['Close'].iloc[i-1]
            top = df['Low'].iloc[i-2] if is_up_candle else df['Low'].iloc[i]
            bottom = df['High'].iloc[i] if is_up_candle else df['High'].iloc[i-2]
            fvg_list.append((i-2, i, bottom, top))

    return fvg_list

def price_in_fvg(price, fvg):
    _, _, low, high = fvg
    return low < price < high

def plot_candles_with_fvg(df, fvg_list=None, title=""):
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

    if fvg_list:
        for start, end, low, high in fvg_list:
            fig.add_shape(
                type='rect',
                x0=start, x1=end,
                y0=low, y1=high,
                xref='x', yref='y',
                fillcolor='rgba(255, 165, 0, 0.3)',
                line=dict(color='rgba(255,165,0,0.8)', width=2, dash='dash')
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

# session state for iteration index and results
if 'scan_index' not in st.session_state:
    st.session_state.scan_index = 0
if 'hits' not in st.session_state:
    st.session_state.hits = []

batch_size = 10
interval, _ = TIMEFRAMES["6M"]

while st.session_state.scan_index < len(UNIVERSE) and not st.session_state.hits:
    batch = UNIVERSE[st.session_state.scan_index:st.session_state.scan_index + batch_size]
    st.session_state.scan_index += batch_size
    for ticker in batch:
        df = fetch_data(ticker, interval)
        if df is None or df.empty:
            continue
        last_close = df['Close'].iloc[-1]
        fvg_list = find_fvg(df)
        fvg_below = [fvg for fvg in fvg_list if price_in_fvg(last_close, fvg) and fvg[3] < last_close]
        if fvg_below:
            st.session_state.hits.append({'ticker': ticker, 'df': df, 'fvg': fvg_below})
            break  # stop on first hit in batch

if st.session_state.hits:
    st.subheader("Found stocks with price inside 6M FVG below current price")
    for stock in st.session_state.hits:
        plot_candles_with_fvg(stock['df'], fvg_list=stock['fvg'], title=f"{stock['ticker']} - 6M FVG")
else:
    st.info("No stocks found with price inside 6M FVG below current price yet. Scanning...")

