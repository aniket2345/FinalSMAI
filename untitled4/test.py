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

def delete_classifieres(path):
    os.system('rm -rf '+path)
def save_to_pickle(data, path):
    file_to_write = open(path, "wb")
    pickle.dump(data, file_to_write)
    file_to_write.close()
def read_from_pickle(path):
    infile = open(path, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data
def save_adjusted_dataframe(file_path, adj_DataFrame):
    if path.exists(file_path):
        print("--Initiating Deleting--")
        delete_command = 'rm -rf '+file_path
        os.system(delete_command)
        adj_DataFrame.to_csv(file_path)
    else:
        adj_DataFrame.to_csv(file_path)

def make_adjusted_df(tickers):

    # CREATES A DATAFRAME WITH ONLY THE ADJUSTED CLOSE
    Yahoo_path = os.getcwd()+"/Yahoo_Data" # path for data
    adj_DataFrame = pd.DataFrame() # empty dataframe
    for count,ticker in enumerate(tickers):
        adj_close = pd.read_csv(Yahoo_path+"/"+ticker, index_col=0, parse_dates=True) #loads yahoo dataset for ticker
        adj_close = adj_close.rename(columns={'Adj Close': ticker}) # renames Adj close
        adj_close = adj_close.drop(['Open','High','Low','Close','Volume'], 1) # keep ONLY adj close
        if adj_DataFrame.empty:
            adj_DataFrame = adj_close # if first ticker, then set to adj_DataFrame
            print(adj_DataFrame)
        else:
            adj_DataFrame = adj_DataFrame.join(adj_close, how='outer')
            print(adj_DataFrame)

    save_adjusted_dataframe('adjusted_dataframe/adj_df.csv', adj_DataFrame)

def n_day_percent_change(ticker):

    # creates 7 day percentage change

    n_day=7

    percent_change_dataframe = pd.DataFrame() # create a empty dataframe
    percent_change_dataframe=adj_DataFrame # set empty dataframe
    percent_change_dataframe = percent_change_dataframe[[ticker]]

    for i in range(1, n_day+1):
        column_name = str(i)+"days"
        #percent_change_dataframe[str(i)] = ((adjusted_price.shift(-i) - adjusted_price) / adjusted_price)
        percent_change_dataframe[column_name] = (adj_DataFrame[[ticker]].shift(i) - adj_DataFrame[[ticker]])/adj_DataFrame[[ticker]]
        percent_change_dataframe.fillna(0,inplace=True)
    return percent_change_dataframe

def buy_sell_hold(*args):
    cols = [col for col in args]
    #print("cols :",cols)
    requirement = 0.02
    for col in cols:
        if col>requirement:
            #print(col ,' returning 1\n')
            return 1
        if col<-requirement:
            #print(col ,'returning -1\n')
            return -1
    #print('returning 0\n')
    return 0
def creating_featuresets(ticker):
    normalized_adjusted_price = adj_DataFrame.pct_change()
    normalized_adjusted_price = normalized_adjusted_price.replace([np.inf, -np.inf], 0)
    normalized_adjusted_price.fillna(0, inplace=True)
    x = normalized_adjusted_price.values
    target_dataframe = pd.DataFrame()
    target_dataframe[ticker] = list(map(buy_sell_hold,
                                percent_change_dictionary[ticker]['1days'],
                                percent_change_dictionary[ticker]['2days'],
                                percent_change_dictionary[ticker]['3days'],
                                percent_change_dictionary[ticker]['4days'],
                                percent_change_dictionary[ticker]['5days'],
                                percent_change_dictionary[ticker]['6days'],
                                percent_change_dictionary[ticker]['7days'],
                                ))
    target_list=[i[0] for i in target_dataframe.values]
    count = Counter(target_list[4750:])

    target_dataframe.fillna(0, inplace=True)
    target_dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    target_dataframe.dropna(inplace=True)
    y = target_dataframe.values
    return x,y,target_dataframe,target_list[4750:]

def machine_learning(ticker,which_ticker):
    x,y,_,real_list = creating_featuresets(ticker)
    """clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])"""
    clf = neighbors.KNeighborsClassifier()
    #clf = svm.LinearSVC()
    #clf = SVC(kernel="linear", C=0.025)

    x_train = x[:4750]
    x_test = x[4750:]
    y_train = y[:4750]
    y_test = y[4750:]

    which_ticker =which_ticker+1
    print(which_ticker, ' -- Processing -- ', ticker)
    clf.fit(x_train,y_train.ravel())
    prediction = clf.predict(x_test)
    prediction_list = [i for i in prediction]
    predictions_count = Counter(prediction)
    #print("Predicted class counts : ", Counter(predictions))
    file_path ='stock_market_classifiers/stc-'+ticker[0:-7]+'.pickle'
    if (path.exists(file_path)):
        os.system('cd stock_market_classifiers; rm -rf stc-'+ticker[0:-7]+'.pickle')
        #save_to_pickle(clf, file_path)
    else:
        pass
        #save_to_pickle(clf,file_path)
    return real_list,prediction_list



def create_nday_featureset(ticker, adj_DataFrame):
    n_day = 100
    #DataFrame = adj_DataFrame[ticker].pct_change()
    DataFrame_unscaled = adj_DataFrame[ticker]
    DataFrame_unscaled.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(-1,1))
    DataFrame = scaler.fit_transform(np.array(DataFrame_unscaled).reshape(-1,1))
    scaler_object = scaler.fit(np.array(DataFrame_unscaled).reshape(-1,1))
    DataFrame_list = [i[0] for i in DataFrame]
    print(DataFrame_list[0])
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    DataFrame_list_train = DataFrame_list[:int(len(DataFrame_list)*0.75)]
    DataFrame_list_test = DataFrame_list[int(len(DataFrame_list)*0.75):]
    for i in range (0, len(DataFrame_list_train)-(n_day+1)):
        x_train.append(DataFrame_list_train[i:i+n_day])
        y_train.append(DataFrame_list_train[i+n_day])
    for i in range (0, len(DataFrame_list_test)-(n_day+1)):
        x_test.append(DataFrame_list_test[i:i+n_day])
        y_test.append(DataFrame_list_test[i+n_day])
    print(x_train)
    x_train_np = np.array(x_train)
    y_train_np = np.array(y_train)
    x_test_np = np.array(x_test)
    y_test_np = np.array(y_test)
    print(x_train_np)
    print(x_train_np.shape)
    x_train_np = x_train_np.reshape(x_train_np.shape[0],x_train_np.shape[1],1)
    print(x_train_np)
    x_test_np = x_test_np.reshape(x_test_np.shape[0],x_test_np.shape[1],1)

    return x_train_np, y_train_np, x_test_np, y_test_np, scaler_object

def Deep_model():
    model = Sequential()
    model.add(LSTM(50,return_sequences=True, input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def deep_learning(ticker, adj_DataFrame):
    x_train,y_train,x_test,y_test,_=create_nday_featureset(ticker, adj_DataFrame)
    model = Deep_model()
    model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=100, batch_size=64, verbose=1)
    #os.system('rm -rf Deep_Models')
    save_location = 'Stock_Deep_Models/'+ticker[0:-7]
    model.save(save_location)
    print("FINISHED CONFIGURING SEQUENTIAL LSTM MODEL FOR ",ticker)

#scaler = MinMaxScaler(feature_range=(-1,1))
Yahoo_path = os.getcwd()+"/Yahoo_Data/"
tickers = [ticker for ticker in os.listdir(Yahoo_path) if not ticker == '.DS_Store']
#make_adjusted_df(tickers) # To Create and Save Adjusted DataFrame
adj_DataFrame = pd.read_csv('adjusted_dataframe/adj_df.csv', index_col=0, parse_dates=True)
"""tickers_batch_one = tickers[1:10]
for ticker in tickers_batch_one:
    deep_learning(ticker,adj_DataFrame)"""
create_nday_featureset(tickers[0],adj_DataFrame)


"""save_location = 'Stock_Deep_Models/'+tickers[0][0:-7]

model = keras.models.load_model(save_location)
x_train,y_train,x_test,y_test,scaler_object = create_nday_featureset(tickers[0],adj_DataFrame)
prediction = model.predict(x_test)
prediction = scaler_object.inverse_transform(prediction)
y_test = scaler_object.inverse_transform(y_test.reshape(-1,1))
error = math.sqrt(mean_squared_error(y_test, prediction))
for i in range(len(prediction)):
    print((y_test[i]-1400),' || ',prediction[i])
print(error)"""

"""print(type(prediction))
print(type(y_test))
error = 0
for i in range(0,683):
    error = error + abs(y_test[i]-prediction[i][0])
    print(y_test[i],' == ',prediction[i][0])
print(error/683)
"""
"""percent_change_dictionary = {}
target_dictionary={}

adjusted_price = pd.read_csv('adjusted_dataframe/adj_df.csv', index_col=0, parse_dates=True)
adjusted_price.fillna(0, inplace=True)
"""





"""for ticker in tickers:
    percent_change_dictionary[ticker] = n_day_percent_change(ticker)
save_to_pickle(percent_change_dictionary,'percent_change/percent_change_dictionary.pickle')"""




"""
percent_change_dictionary = read_from_pickle('percent_change/percent_change_dictionary.pickle')

accuracy_dictionary={}
train_size = 4750
test_size = 1597
which_ticker = 0
average_accuracy = 0
for ticker in tickers :
    real_list,prediction_list = machine_learning(ticker,which_ticker)
    correct=0
    for i in range(len(real_list)):
        if real_list[i]==prediction_list[i]:
            correct= correct+1
    accuracy = (correct/test_size)*100
    average_accuracy=average_accuracy+accuracy
    accuracy_dictionary[ticker]=accuracy
    which_ticker=which_ticker+1
average_accuracy=average_accuracy/45
print(average_accuracy)
"""



# --------------------------------------------------------------------------------------------------------------------------------
# CODE I DON'T NEED CURRENTLY

"""

def initial():
    path = os.getcwd()+"/Yahoo_Data"
    tickers = [ ticker for ticker in os.listdir(path) if not ticker=='.DS_Store']
    return tickers
    data = {}
    for ticker in tickers :
        data[ticker]=pd.read_csv(path+"/"+ticker, index_col=0, parse_dates=True)
    # LOOK INTO RESAMPLING 'https://www.youtube.com/watch?v=19yyasfGLhk&list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ&index=4' Allows you to add up min min data into data of one hour by either adding ( sum() useful for volume of a day found by adding data of single days) or taking average (mean() used for price by taking mean of price over days so that we can know what the average price was for that week) it is upto your understanding to figure out which resample will be more beneficial depending on your reqirements


"""
# --------------------------------------------------------------------------------------------------------------------------------
