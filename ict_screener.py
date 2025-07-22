import streamlit as st
import yfinance as yf

st.title("yfinance data debug")

ticker = st.text_input("Enter ticker symbol:", "EOSE")

if st.button("Fetch Data"):
    df = yf.download(ticker, period="6mo", interval="1d")
    st.write("columns:", df.columns.tolist())
    st.write("index type:", type(df.index))
    st.write("index sample:", df.index[:5])
    st.write("dtypes:")
    st.write(df.dtypes)
    st.write("data sample:")
    st.write(df.head())
