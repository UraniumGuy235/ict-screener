import streamlit as st
import yfinance as yf

st.title("yfinance data debug")

ticker = st.text_input("ticker", "EOSE")

if st.button("fetch data"):
    df = yf.download(ticker, period="6mo", interval="1d")
    st.write("columns:", df.columns.tolist())
    st.write("index dtype:", df.index.dtype)
    st.write("index sample:", df.index[:5].tolist())
    st.write("dtypes:")
    st.write(df.dtypes)
    st.write("data sample:")
    st.write(df.head())
