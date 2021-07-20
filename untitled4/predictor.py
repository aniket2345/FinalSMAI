#miscellaneous
import os
from os import path
import pickle
import math
from tqdm import tqdm
import math
from collections import Counter
#data management
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#machine learning

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow.keras.optimizers

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
os.system('clear')


data_path = os.getcwd()+'/Daily_x/today/'
tickers_full = [ticker for ticker in os.listdir(data_path)]
tickers = [ticker[:-7] for ticker in os.listdir(data_path)]
print(tickers)
if(".D" in tickers) :tickers.remove(".D")
if("" in tickers): tickers.remove("")
tickers.remove("ITC")
tickers.remove("JSWSTEEL")
"""tickers.remove(".D")
tickers.remove("")"""
print(tickers)
Price_Dict = {}
for count,ticker in enumerate(tickers):
    print("TICKER : ",ticker)
    model_location = 'Stock_Deep_Models/'+ticker
    data_location = 'Daily_x/today/'+tickers_full[count]
    """command = "cd "+model_location+"; ls"
    os.sys(command)"""
    model = keras.models.load_model(model_location)
    data = pd.read_csv(data_location)
    data = data[['Adj Close']]
    data.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    scaler_object = scaler.fit(np.array(data).reshape(-1, 1))
    data_scaled_list = [i[0] for i in data_scaled]
    data_scaled_list = data_scaled_list[-100:]
    temp_file = []
    temp_file.append(data_scaled_list)
    data_scaled_np = np.array(temp_file)
    print(data_scaled_np)
    print(data_scaled_np.shape)
    data_scaled_np = data_scaled_np.reshape(data_scaled_np.shape[0],data_scaled_np.shape[1],1)
    prediction = model.predict(data_scaled_np)
    prediction = scaler_object.inverse_transform(prediction)
    Price_Dict[ticker]=prediction
for i in Price_Dict:
    print(i, ' ', Price_Dict[i])


