for idx, ticker in enumerate(tickers):
    df = fetch_data(ticker, start=start_date, end=end_date, interval=interval)
    if df is None or df.empty:
        continue
    df = detect_fvg(df)
    df['eq_lows'] = find_equal_lows(df)
    df['eq_highs'] = find_equal_highs(df)
    df['open_confluence'] = open_confluence(df)

    # setup can be any one indicator firing
    df['setup'] = df['bullish_fvg'] | df['eq_lows'] | df['eq_highs'] | df['open_confluence']

    if df['setup'].any():
        found_setups = True
        with cols[idx % len(cols)]:
            st.markdown(f"### {ticker}")
            fig = go.Figure()

            # candlestick
            fig.add_trace(go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name='Price'))

            # horizontal lines for equal lows
            eq_lows_levels = df.loc[df['eq_lows'], 'Low'].unique()
            for level in eq_lows_levels:
                fig.add_hline(y=level, line=dict(color='blue', dash='dash'), annotation_text='Equal Low', annotation_position='bottom left')

            # horizontal lines for equal highs
            eq_highs_levels = df.loc[df['eq_highs'], 'High'].unique()
            for level in eq_highs_levels:
                fig.add_hline(y=level, line=dict(color='red', dash='dash'), annotation_text='Equal High', annotation_position='top left')

            # markers for bullish FVG
            bullish_fvg_points = df[df['bullish_fvg']]
            fig.add_trace(go.Scatter(x=bullish_fvg_points.index,
                                     y=bullish_fvg_points['Low'] * 0.995,
                                     mode='markers',
                                     marker=dict(color='green', size=10, symbol='triangle-up'),
                                     name='Bullish FVG'))

            # markers for open confluence
            open_conf_points = df[df['open_confluence']]
            fig.add_trace(go.Scatter(x=open_conf_points.index,
                                     y=open_conf_points['Open'],
                                     mode='markers',
                                     marker=dict(color='purple', size=10, symbol='circle'),
                                     name='Open Confluence'))

            fig.update_layout(height=600,
                              title=f"{ticker} Price Chart with Indicators",
                              xaxis_title="Date",
                              yaxis_title="Price")

            st.plotly_chart(fig, use_container_width=True)

if not found_setups:
    st.info("no setups found today. try adjusting timeframe or start date.")
