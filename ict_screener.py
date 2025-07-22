import streamlit as st

# app title
st.set_page_config(page_title="TradingView Live Chart", layout="wide")
st.title("📈 TradingView Live Chart Viewer")

# user input for ticker symbol
st.markdown("Enter a ticker symbol (format: `EXCHANGE:TICKER`, e.g. `NASDAQ:AAPL`, `NYSE:TSLA`, `TSE:SU`):")
ticker = st.text_input("Ticker", value="NASDAQ:AAPL")

# embed function
def embed_tradingview_chart(symbol):
    st.components.v1.html(f"""
        <div class="tradingview-widget-container">
          <iframe src="https://s.tradingview.com/widgetembed/?symbol={symbol}&interval=D&theme=dark&style=1&toolbar_bg=333&withdateranges=1&hide_side_toolbar=0&allow_symbol_change=1&save_image=1"
            width="100%" height="600" frameborder="0" allowfullscreen></iframe>
        </div>
    """, height=600)

# render chart
if ticker:
    embed_tradingview_chart(ticker.strip().upper())
