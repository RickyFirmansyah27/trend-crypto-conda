import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import talib
import util

def main():
    ticker_data = util.get_ticker_data(ticker_symbol, data_periode, data_interval)

    if len(ticker_data) != 0:
        ticker_data['fast_ma'] = talib.EMA(ticker_data['Close'], int(ma1))
        ticker_data['slow_ma'] = talib.EMA(ticker_data['Close'], int(ma2))
        ticker_data['rsi'] = talib.RSI(ticker_data['Close'], timeperiod=14)
        ticker_data['sar'] = talib.SAR(ticker_data['High'], ticker_data['Low'], acceleration=0.02, maximum=0.2)
        
        
        
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)        
        fig = util.candle_chart(fig, ticker_data)        
        fig = util.row_trace(fig, ticker_data.index, ticker_data['fast_ma'], 'fast MA', 'yellow', 1)
        fig = util.row_trace(fig, ticker_data.index, ticker_data['slow_ma'], 'slow MA', 'blue', 1)
        fig = util.row_trace(fig, ticker_data.index, ticker_data['sar'], 'Parabolic SAR', 'black', 1, mode='markers')
        st.write(fig)


         
        #Parabolic SAR
        ticker_data['sar_diff'] = ticker_data['sar'].diff()
        ticker_data.dropna(subset=['sar_diff'], inplace=True)
        util.join_frame(ticker_data['sar'],ticker_data['sar_diff'])

        #EMA Cross
        ticker_data['ma_diff'] = ticker_data['fast_ma'] - ticker_data['slow_ma']
        util.join_frame_MA(ticker_data['fast_ma'], ticker_data['slow_ma'], ticker_data['ma_diff'])


        #Data Cryptocurrency
        st.subheader('Data Cryptocurrency')
        ticker_data.dropna(subset=['sar_diff','ma_diff'], inplace=True)
        dataset = util.create_trendlist(ticker_data)
       

        #Support and Resistance
        st.subheader('Support/Resistance')
        SR = util.SR(ticker_data)
        st.write(SR)
        util.join_df_to_ticker_data(dataset,ticker_data)
        X = ticker_data[['ma_diff','sar_diff']]
        Y = dataset['trend']
        util.join_frame(X,Y)

        if is_training_mode:
            #training_ML
            st.write()
            #util.train_ml_model(X,Y)
        else:
            st.header('Data Testing')
            util.model(ticker_data, X,"naive_bayes_model.p")
            util.komparasi_data(ticker_data,Y)     
       
           
if __name__ == '__main__':
    ticker_symbol = st.sidebar.text_input(
    "Please enter the stock symbol", 'BTC-USD')
    is_training_mode = st.sidebar.checkbox('Training Mode', value=False)
    data_periode = st.sidebar.text_input('Periode', '30d')
    data_interval = st.sidebar.radio('Interval', ['1h','30m','15m','1d'])
    ma1 = st.sidebar.text_input('EMA 1', 5)
    ma2 = st.sidebar.text_input('EMA 2', 20)
 

    st.header("Visualisasi Data")
    st.write("---")

    main()
