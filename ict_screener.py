import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict bullish stock screener + single ticker viewer")

TIMEFRAMES = {
    "6M": ("1mo", 3),
}

SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "BRK-B", "JPM", "JNJ", "V",
    "WMT", "PG", "MA", "NVDA", "DIS", "HD", "BAC", "XOM", "PFE", "ADBE",
    "CMCSA", "NFLX", "T", "KO", "INTC", "CSCO", "MRK", "PEP", "ABT", "CVX",
    "CRM", "NKE", "ORCL", "ACN", "MDT", "AVGO", "MCD", "COST", "TXN", "QCOM",
    "DHR", "NEE", "LLY", "UNP", "LIN", "BMY", "LOW", "IBM", "PM", "AMGN",
    "SBUX", "HON", "CAT", "MMM", "GS", "GE", "NOW", "LMT", "BLK", "BKNG",
    "ADI", "ZTS", "CB", "FIS", "SYK", "CL", "ISRG", "PLD", "ANTM", "TMO",
    "SPGI", "MO", "CSX", "USB", "MDLZ", "CI", "CCI", "PNC", "ADP", "INTU",
    "MMC", "SCHW", "GILD", "APD", "VRTX", "EMR", "DUK", "ICE", "BDX",
    "MCO", "SO", "SHW", "CME", "COP", "ATVI", "ECL", "EW", "MET", "PGR",
    "HUM", "PSA", "DG", "TJX", "C", "F", "DLR", "AEP", "ETN", "KMB",
    "LRCX", "ROST", "AIG", "WBA", "EXC", "EL", "KR", "OXY", "ALL", "BSX",
    "VLO", "ORLY", "DLTR", "KLAC", "DD", "ITW", "MCK", "HCA", "CMG", "FCX",
    "SNPS", "REGN", "NOC", "WM", "ADM", "XEL", "KMI", "BIIB", "PEG", "D",
    "CTSH", "WFC", "BAX", "COF", "CTAS", "EBAY", "STZ", "VTR", "ESS", "A",
    "DOW", "GLW", "TRV", "FTNT", "TSN", "MS", "ZBH", "MNST", "KEY", "MPC",
    "HES", "CNC", "PAYX", "TEL", "EXPE", "ANSS", "ILMN", "AKAM", "CTRA",
    # you can add remaining tickers as needed
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
        return df
    except Exception:
        return None

def find_fvg(df):
    fvg_list = []
    for i in range(2, len(df)):
        high2, low2 = df.loc[i-2, 'High'], df.loc[i-2, 'Low']
        high1, low1 = df.loc[i-1, 'High'], df.loc[i-1, 'Low']
        high0, low0 = df.loc[i, 'High'], df.loc[i, 'Low']
        # bullish gap
        if low0 > high2:
            fvg_list.append((i-2, i, low0, high2))
        # bearish gap
        elif high0 < low2:
            fvg_list.append((i-2, i, low2, high0))
    return fvg_list

def price_in_fvg(price, fvg):
    lower, upper = fvg[2], fvg[3]
    return lower < price < upper if lower < upper else upper < price < lower

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
        for start_i, end_i, lower, upper in fvg_list:
            fig.add_shape(
                type='rect',
                x0=start_i, x1=end_i,
                y0=lower, y1=upper,
                xref='x', yref='y',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='rgba(255, 0, 0, 0.6)'),
                layer='below'
            )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

if "scan_index" not in st.session_state:
    st.session_state.scan_index = 0
if "hits" not in st.session_state:
    st.session_state.hits = []

batch_size = 10
interval, _ = TIMEFRAMES["6M"]

while st.session_state.scan_index < len(SP500_TICKERS) and not st.session_state.hits:
    batch = SP500_TICKERS[st.session_state.scan_index:st.session_state.scan_index + batch_size]
    st.session_state.scan_index += batch_size
    for ticker in batch:
        st.write(f"scanning {ticker}...")
        df = fetch_data(ticker, interval)
        if df is None or df.empty:
            continue
        last_close = df['Close'].iloc[-1]
        fvg_list = find_fvg(df)
        fvg_below = [fvg for fvg in fvg_list if price_in_fvg(last_close, fvg) and max(fvg[2], fvg[3]) < last_close]
        if fvg_below:
            st.session_state.hits.append({'ticker': ticker, 'df': df, 'fvg': fvg_below})
            break

if st.session_state.hits:
    hit = st.session_state.hits[0]
    st.subheader(f"found ticker {hit['ticker']} with 6M FVG below current price")
    plot_candles_with_fvg(hit['df'], hit['fvg'], title=f"{hit['ticker']} 6M chart with FVG below price")
else:
    st.info("no stocks found with price inside a 6M FVG below current price yet, scanning more...")

