import math
import os
import math
import torch
import numpy as np
import torch.nn as nn
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outputpath = 'D:/Experiment/meta_learning/data/meta_sample'
k = [1111,2222,3333,4444,5555]
meta = pd.DataFrame(columns=['cell', 'mean', 'median', 'range', 'var', 'std', 'CV', 'waverate', 'skewness', 'kurtosis',
                             'trend', 'seasonality', 'size', 'window', 'lr', 'layer', 'head', 'dmodel', 'perf'])
i = 0
for cell in k:
    path = 'D:/Experiment/meta_learning/data/'+str(cell)+'/'+str(cell)+'.txt'
    alldata = pd.read_table(path)
    data1 = alldata.fillna(0, inplace=False)
    data2 = data1.loc[data1['Internet'] > 0]
    df = data2.loc[data2['Country'] == 39]
    df3 = df.drop(columns=['Cell', 'Country', 'ReceivedSMS', 'SendSMS', 'IncomingCall', 'OutgoingCall'])
    df4 = df3.set_index(['Time'], inplace=False)
    df4 = df4.sort_values('Time')
    df4 = df4.reset_index()
    df5 = df4.drop(columns=['Time'])
    df6 = df5

    mean = float(df6.mean())
    median = float(df6.median())
    maxV = df6.max()
    minV = df6.min()
    R = float(maxV-minV)
    var = float(df6.var())
    std = float(df6.std())
    cv = float(std/mean)
    waverate = float((df6.quantile(0.9)-df6.quantile(0.1)+1)/(df6.quantile(0.1)+1))
    sk = (df6-mean)**3
    skewness = float(sk.sum()/(len(df6)*(std**3)))
    ku = (df6-mean)**4
    kurtosis = float(sk.sum()/(len(df6)*(std**4)))-3

    STL = df5['Internet'].tolist()
    rd = sm.tsa.seasonal_decompose(STL, period=144)
    train = pd.DataFrame(columns=['value', 'trend', 'seasonal', 'residual', 'add'])
    train['value'] = df5['Internet']
    train['trend'] = rd.trend
    train['seasonal'] = rd.seasonal
    train['residual'] = rd.resid
    train['add'] = train['trend']+train['seasonal']+train['residual']

    detrend = train['value'] - train['trend']
    vardetrend = float(detrend.var())
    varItn = float(train['value'].var())
    trend = 1-(vardetrend/varItn)

    deseasonality = train['value'] - train['seasonal']
    vardeseasonality = float(deseasonality.var())
    seasonality = 1-(vardeseasonality/varItn)

    path2 = 'D:/Experiment/meta_learning/data/meta_sample/cell' + str(cell) + '/cell' + str(cell) + '.csv'
    ybj = pd.read_csv(path2)
    yuanbj = ybj.loc[ybj['perf'] == min(ybj['perf'])]
    size = float(yuanbj['size'])
    win = int(yuanbj['window'])
    lr = float(yuanbj['lr'])
    layer = int(yuanbj['layer'])
    head = int(yuanbj['head'])
    d = int(yuanbj['dmodel'])
    p = float(yuanbj['perf'])
    meta.loc[i] = [cell, mean, median, R, var, std, cv, waverate, skewness, kurtosis, trend, seasonality, size, win, lr, layer, head, d, p]
    i += 1

meta.to_csv(outputpath + '/meta_sample.csv', index=False)



