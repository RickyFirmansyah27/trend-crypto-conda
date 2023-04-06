import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB



def moving_average(ticker_data, period):
    ma = ticker_data['Close'].ewm(span=period).mean()
    column_name = 'ma_' + str(period)
    ticker_data[column_name] = ma
    return ticker_data

def get_ticker_data(ticker_symbol, data_period, data_interval):
    ticker_data = yf.download(tickers=ticker_symbol, period=data_period, interval=data_interval)
    if len(ticker_data) == 0:
        st.write('Cryptocurrency tidak ditemukan atau period terlalu panjang')
    else:
        ticker_data.index = ticker_data.index.strftime("%d-%m-%Y %H:%M")
    return ticker_data

def candle_chart(candle_fig, ticker_data):
    candle_fig.add_trace(
        go.Candlestick(x=ticker_data.index,
        open=ticker_data['Open'],
        close=ticker_data['Close'],
        low=ticker_data['Low'],
        high=ticker_data['High'],
        name='Market Data'
        )
    )
    candle_fig.update_layout(
        height=800,
    )
    return candle_fig




def row_trace(candle_fig, x_value, y_value, trace_name, color, row_num, mode = 'lines'):
    candle_fig.add_trace(
        go.Scatter(
            x = x_value,
            y = y_value,
            name = trace_name,
            line = dict(color=color),
            mode = mode
        ),
        row = row_num,
        col = 1
    )
    return candle_fig



    
def join_frame(df1,df2):
    dataframe = pd.concat([df1,df2], axis=1, join='outer')
    dataframe.dropna(axis=1,how='all')
    #st.write(dataframe)
    return dataframe

    
def join_frame_MA(df1,df2,df3):
    dataframe = pd.concat([df1,df2,df3], axis=1, join='outer')
    #st.write(dataframe)   

def data(ticker_data):
    df1 = ticker_data['Open']
    df2 = ticker_data['High']
    df3 = ticker_data['Low']
    df4 = ticker_data['Close']
    ticker_data.dropna(inplace=True)
    data = pd.concat([df1,df2,df3,df4], axis=1, join='outer')
    st.write(data)
    



def SR(ticker_data):
    levels = []
    for i in range(2,ticker_data.shape[0]-2):
        if is_Support(ticker_data,i):
            levels.append((i,ticker_data['Low'][i]))
        elif is_Resistance(ticker_data,i):
            levels.append((i,ticker_data['High'][i]))

    mean = np.mean(ticker_data['High']-ticker_data['Low'])
    
    
    levels = []
    state = []
    for i in range(2,ticker_data.shape[0]-2):
        if is_Support(ticker_data,i):
            x = ticker_data['Low'][i]
            if np.sum([abs(x-y)< mean for y in levels]) == 0:
                levels.append((x))
                state.append('Support')
             
             
        elif is_Resistance(ticker_data,i):
            x = ticker_data['High'][i]
            if np.sum([abs(x-y)< mean for y in levels]) == 0:
                levels.append((x))
                state.append('Resistance')

    
    SR = pd.DataFrame(
         {'S/R' : levels,
          'Status' : state
         }
        )
   
    return SR

def level_sr(x,levels):
    np.sum([abs(x-y)< mean for y in levels]) == 0
   


def is_Support(ticker_data,i):
    support = ticker_data['Low'][i] < ticker_data['Low'][i-1] and ticker_data['Low'][i] < ticker_data['Low'][i+1] and ticker_data['Low'][i+1] < ticker_data['Low'][i+2] and ticker_data['Low'][i-1] < ticker_data['Low'][i-2]
    return support 

def is_Resistance(ticker_data,i):
    resistance = ticker_data['High'][i] > ticker_data['High'][i-1] and ticker_data['High'][i] > ticker_data['High'][i+1] and ticker_data['High'][i+1] > ticker_data['High'][i+2] and ticker_data['High'][i-1] > ticker_data['High'][i-2]
    return resistance


   
def create_trendlist(ticker_data):
    ticker_data['ma_diff'] = ticker_data['fast_ma'] - ticker_data['slow_ma']
    ticker_data['sar_diff'] = ticker_data['sar'].diff()
    trend = []
    for i in range(len(ticker_data)):
        if ticker_data['ma_diff'][i] >= 0:
            if ticker_data['sar_diff'][i] >= 0:   
                state = 'Uptrend'
                trend.append(state)
            else:
                state = 'Uptrend'
                trend.append(state)
                
        if ticker_data['ma_diff'][i] <= 0:
            if ticker_data['sar_diff'][i] <= 0:
                state = 'Downtrend'
                trend.append(state)
            else:
                state = 'Downtrend'
                trend.append(state)

       
    
    df1 = ticker_data['Open']
    df2 = ticker_data['High']
    df3 = ticker_data['Low']
    df4 = ticker_data['Close']
    df5 = ticker_data['ma_diff']
    df6 = ticker_data['sar_diff']
    ticker_data.dropna(subset=["sar_diff"], inplace=True)
    dataset =  pd.DataFrame(
         {'Open' : df1,
          'High' : df2,
          'Low' : df3,
          'Close' : df4,
          'ma_diff' : df5,
          'sar_diff' : df6,
          'trend' : trend
         }
        )
    dataset.dropna(subset=['sar_diff','ma_diff'], inplace=True)
    st.write(dataset)
    #dataset.to_csv('dataset.csv')
    return dataset


def train_ml_model(X,Y):
    clf = GaussianNB()
    clf = clf.fit(X,Y)
    pickle.dump(clf, open("naive_bayes_model.p", "wb"))  
    
def model(ticker_data, X, pickle_filename):
    clf = pickle.load(open(pickle_filename,"rb"))
    ticker_data['trend_pred'] = clf.predict(X)
    return ticker_data

    

def join_df_to_ticker_data(df, ticker_data):
    ticker_data = pd.concat([ticker_data, df], axis=1, join='outer')
    return ticker_data

def komparasi_data(ticker_data,dataset):
    df1 = ticker_data['Open']
    df2 = ticker_data['High']
    df3 = ticker_data['Low']
    df4 = ticker_data['Close']
    df5 = dataset
    df6 = ticker_data['trend_pred']
    dataframe = pd.concat([df1,df2,df3,df4,df5,df6], axis=1, join='outer')
    dataframe.dropna(axis=1,how='all')
    st.write(dataframe)

    acc = 0
    index = 0
    for i in range(len(ticker_data)):
        if df5[i] == 'Uptrend' and ticker_data['trend_pred'][i] == 'Uptrend':
            acc = acc + 1
            index = index +1
            
        elif df5[i] == 'Downtrend' and ticker_data['trend_pred'][i] == 'Downtrend':
            acc = acc + 1
            index = index +1
            
        else:
            acc = acc + 0
            index = index +1
    hasil = acc * (100 / index)
    hasil = round(hasil,3)
    st.write('Accurasi: '+str(hasil)+'%')
