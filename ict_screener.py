import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ict fvg below price screener + single ticker viewer")

TIMEFRAMES = {
    "6M": ("3mo", 6),  # approx 6 months, score high to prioritize
}

# simplified sp500 tickers list for example, expand with full list below
SP500_TICKERS = [
    # truncated sample - replace with full sp500 list
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK-B", "JNJ", "V", "PG", "NVDA",
    # add all 500 tickers here
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
    # iterate over bars with enough history (i from 3 to len-1)
    for i in range(3, len(df)):
        c0, c1, c2, c3 = df.loc[i-3, 'Close'], df.loc[i-2, 'Close'], df.loc[i-1, 'Close'], df.loc[i, 'Close']
        h0, h1, h2, h3 = df.loc[i-3, 'High'], df.loc[i-2, 'High'], df.loc[i-1, 'High'], df.loc[i, 'High']
        l0, l1, l2, l3 = df.loc[i-3, 'Low'], df.loc[i-2, 'Low'], df.loc[i-1, 'Low'], df.loc[i, 'Low']
        
        # bearish fvg check
        bear_condition = (c0 <= h1) and (c2 <= c1) and (h3 < l1)
        # bullish fvg check
        bull_condition = (c0 >= l1) and (c2 >= c1) and (l3 > h1)
        
        # atr or volatility filter omitted for simplicity
        
        if bear_condition or bull_condition:
            # fvg box coords: bar index, top, bar index, bottom
            top = l3 if bull_condition else l1
            bottom = h1 if bull_condition else l3
            fvg_list.append( (i, i-1, top, bottom) )
    return fvg_list

def price_in_fvg(price, fvg):
    # fvg = (bar1, bar2, top, bottom)
    low, high = sorted([fvg[2], fvg[3]])
    return low <= price <= high

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
        for fvg in fvg_list:
            bar1, bar2, top, bottom = fvg
            low, high = sorted([top, bottom])
            fig.add_shape(
                type='rect',
                x0=bar2,
                x1=bar1,
                y0=low,
                y1=high,
                xref='x',
                yref='y',
                fillcolor='rgba(255,165,0,0.3)',  # orange transparent
                line=dict(color='rgba(255,165,0,0.5)', width=1),
                layer='below'
            )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(type='category', tickangle=-45),
    )
    st.plotly_chart(fig, use_container_width=True)

def analyze_ticker(ticker):
    df = fetch_data(ticker, TIMEFRAMES["6M"][0])
    if df is None or df.empty:
        return None, None
    last_close = df['Close'].iloc[-1]
    fvg_list = find_fvg(df)
    fvg_below = []
    for fvg in fvg_list:
        low, high = sorted([fvg[2], fvg[3]])
        if price_in_fvg(last_close, fvg) and high < last_close:
            fvg_below.append(fvg)
    return df, fvg_below

st.sidebar.title("settings")
search_mode = st.sidebar.radio("mode", ["screener", "single ticker"])

if search_mode == "screener":
    st.subheader("screening S&P 500 for price inside 6M FVG below price")
    batch_size = 10
    sp500 = SP500_TICKERS  # full list pls
    bullish_found = False
    i = 0
    bullish_stocks = []
    while not bullish_found and i < len(sp500):
        batch = sp500[i:i+batch_size]
        for ticker in batch:
            st.write(f"scanning {ticker}...")
            df, fvg_below = analyze_ticker(ticker)
            if df is not None and fvg_below:
                st.write(f"found {ticker} with price inside 6M FVG below current price!")
                bullish_stocks.append({
                    'ticker': ticker,
                    'df': df,
                    'fvg': fvg_below
                })
                bullish_found = True
                break
        i += batch_size
        if not bullish_found and i >= len(sp500):
            st.write("no stocks found with price inside a 6M FVG below current price yet, scanned all S&P 500.")

    if bullish_stocks:
        for stock in bullish_stocks:
            plot_candles_with_fvg(stock['df'], stock['fvg'], title=f"{stock['ticker']} price inside 6M FVG below price")
    else:
        st.info("no bullish setups found")

elif search_mode == "single ticker":
    ticker = st.text_input("enter ticker symbol", "AAPL").upper()
    if ticker:
        df, fvg_below = analyze_ticker(ticker)
        if df is None:
            st.warning(f"no data for {ticker}")
        else:
            plot_candles_with_fvg(df, fvg_below, title=f"{ticker} 6M chart price inside FVG below price")
