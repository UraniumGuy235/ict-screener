
import streamlit as st
import yfinance as yf
import pandas as pd
import pytz
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 ICT Screener – Bullish FVGs w/ Liquidity")

# --- top 100 US stocks by market cap (approx, stable as of mid-2025)
TOP_100 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "UNH", "JPM", "V", "LLY", "XOM", "JNJ",
    "MA", "PG", "HD", "CVX", "MRK", "ABBV", "AVGO", "KO", "PEP", "COST", "WMT", "BAC", "MCD", "TMO", "DIS",
    "ADBE", "PFE", "WFC", "CSCO", "LIN", "ORCL", "NFLX", "ACN", "ABT", "NKE", "DHR", "INTC", "CRM", "AMD",
    "TXN", "CMCSA", "PM", "BMY", "MDT", "BA", "NEE", "AMGN", "HON", "IBM", "SBUX", "RTX", "GS", "LOW", "CAT",
    "GE", "NOW", "INTU", "ISRG", "QCOM", "LMT", "SPGI", "USB", "AMT", "CVS", "BLK", "PLD", "T", "MDLZ", "ZTS",
    "AXP", "CI", "ELV", "DE", "ADP", "MO", "BKNG", "ADI", "MMC", "SYK", "BDX", "APD", "TJX", "ETN", "GILD",
    "CB", "CSX", "REGN", "PNC", "CL", "VRTX", "FISV", "C", "SO", "EOG", "HCA", "PGR", "TRV", "ITW"
]

# === functions ===
def fetch_data(ticker):
    df = yf.download(ticker, period="7d", interval="15m", progress=False)
    df.dropna(inplace=True)
    df = df.tz_localize("UTC").tz_convert("America/New_York")
    return df

def detect_bullish_fvgs(df):
    fvgs = []
    for i in range(2, len(df)):
        if df['Low'].iloc[i-2] > df['High'].iloc[i]:
            ts = df.index[i]
            fvgs.append(ts)
    return fvgs

def detect_equal_lows(df, tolerance=0.001):
    lows = df['Low'].round(2)
    return lows[lows.duplicated(keep=False)].index.tolist()

def detect_equal_highs(df, tolerance=0.001):
    highs = df['High'].round(2)
    return highs[highs.duplicated(keep=False)].index.tolist()

def fvg_in_ny_open(fvg_times):
    return [ts for ts in fvg_times if 9 <= ts.hour <= 11]

def screen_ticker(ticker):
    try:
        df = fetch_data(ticker)
        fvgs = detect_bullish_fvgs(df)
        ny_fvgs = fvg_in_ny_open(fvgs)
        if not ny_fvgs:
            return None

        eq_lows = detect_equal_lows(df)
        eq_highs = detect_equal_highs(df)
        last_price = df['Close'].iloc[-1]

        return {
            "Ticker": ticker,
            "Last Price": round(last_price, 2),
            "FVG Time": ny_fvgs[-1].strftime("%Y-%m-%d %H:%M"),
            "Equal Lows": "✅" if eq_lows else "❌",
            "Equal Highs": "✅" if eq_highs else "❌"
        }
    except:
        return None

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Candles'
    ))
    fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False)
    return fig

# === UI ===
col1, col2 = st.columns([1, 2])

with col1:
    manual_input = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,TSLA")
    do_top100 = st.checkbox("Scan Top 100 US Companies")

with col2:
    start_button = st.button("🚀 Run Screener")

tickers = set()
if manual_input:
    tickers.update([x.strip().upper() for x in manual_input.split(",") if x.strip()])
if do_top100:
    tickers.update(TOP_100)

if start_button and tickers:
    st.info(f"Scanning {len(tickers)} tickers... grab a coffee ☕")
    results = []
    for t in sorted(tickers):
        res = screen_ticker(t)
        if res:
            results.append(res)

    if results:
        st.success(f"Found {len(results)} bullish FVG setups.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("📈 Charts")
        for r in results[:5]:  # limit to 5 charts max
            df = fetch_data(r["Ticker"])
            st.plotly_chart(plot_chart(df, r["Ticker"]), use_container_width=True)
    else:
        st.warning("No setups detected today.")
