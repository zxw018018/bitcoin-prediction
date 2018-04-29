import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import time

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from urllib2 import Request, urlopen, URLError

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[0], inputs.shape[1])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def get_response():
    request = Request("https://apiv2.bitcoinaverage.com/indices/global/ticker/BTCUSD")
    response = urlopen(request)
    return response


table = list()

window_len = 10
predict_len = 2

#build LSTM 
LSTM_training_inputs = [[0 for col in range(4)] for row in range(window_len)]
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
global model
model = build_model(LSTM_training_inputs, output_size=1, neurons = 10)

def predict_model(LSTM_data):
    print LSTM_data
    LSTM_training_inputs = []
    for i in range(window_len):
        temp_set = LSTM_data[i][:]
        print LSTM_data[0]
        for j in range(0, 4):
            print LSTM_data[0][j]
            temp_set[j] = temp_set[j] / LSTM_data[0][j] - 1
        LSTM_training_inputs.append(temp_set)
    LSTM_training_outputs = LSTM_data[window_len + predict_len - 1][2] / LSTM_data[0][2] - 1
    LSTM_training_outputs = np.array([LSTM_training_outputs])
    LSTM_test_inputs = []
    for i in range(predict_len, window_len+predict_len):
        temp_set = LSTM_data[i][:]
        for j in range(0, 4):
            temp_set[j] = temp_set[j] / LSTM_data[0][j] - 1
        LSTM_test_inputs.append(temp_set)
    LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
    LSTM_training_inputs = np.array([LSTM_training_inputs])
    LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
    LSTM_test_inputs = np.array([LSTM_test_inputs])
    model.fit(LSTM_training_inputs[0:1], LSTM_training_outputs[0:1], epochs=3, batch_size=1, verbose=2, shuffle=True)
    print("predict_num:%d"%(i+window_len+predict_len))    
    predict_res = float((np.transpose(model.predict(LSTM_test_inputs))+1)* LSTM_data[0][2])
    print("predict_res:%f"%(predict_res))
    return predict_res

while True:
    actual_response = get_response().read()
    actual_last = float(actual_response.split("last")[1].split("high")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    actual_bid = float(actual_response.split("bid")[1].split("last")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    actual_ask = float(actual_response.split("ask")[1].split("bid")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    actual_volume = float(actual_response.split("volume")[1].split("changes")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    table.append([actual_ask, actual_bid, actual_last, actual_volume])
    print len(table)
    table = table[-(window_len + predict_len):]
    if len(table) == window_len+predict_len:
        predict_data = predict_model(table)
    time.sleep(1)



# while len(LSTM_data) < window_len+predict_len:
#     time.sleep(0.5)




