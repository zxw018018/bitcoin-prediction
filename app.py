import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import plotly.plotly as py
from plotly.graph_objs import *
from scipy.stats import rayleigh
from flask import Flask
import numpy as np
import pandas as pd
import os
import sqlite3
import datetime as dt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
import time
import seaborn as sns
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import datetime
from urllib2 import Request, urlopen, URLError, HTTPError
import time

app = dash.Dash('bitcoin-prediction')
server = app.server

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
    while True:
        try: 
            response = urlopen(request)
            break
        except HTTPError as e:
            print 'Error code: ', e.code
            continue
        except URLError as e:
            print 'Error code: ', e.code
            continue
    return response

    # last = response.read().split("last")[1].split("high")[0].split(' ')[1].split(',')[0]
    # return float(last)
global table
table = list()

window_len = 50
predict_len = 10
slide_window = 20

#build LSTM 
LSTM_training_inputs = [[0 for col in range(4)] for row in range(window_len)]
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)
global model
model = build_model(LSTM_training_inputs, output_size=1, neurons = 10)

def predict_model(LSTM_data):
    #print LSTM_data
    LSTM_training_inputs = []
    for i in range(window_len):
        temp_set = LSTM_data[i][:]
        #print LSTM_data[0]
        for j in range(0, 4):
            #print LSTM_data[0][j]
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
    model.train_on_batch(LSTM_training_inputs[0:1], LSTM_training_outputs[0:1])
#    model.fit(LSTM_training_inputs[0:1], LSTM_training_outputs[0:1], epochs=3, batch_size=1, verbose=2, shuffle=True)
    print("predict_num:%d"%(i+window_len+predict_len))    
    predict_res = float((np.transpose(model.predict(LSTM_test_inputs))+1)* LSTM_data[0][2])
    print("predict_res:%f"%(predict_res))
    return predict_res

global last_list
global predict_list
global show_last_list
last_list = list()
predict_list = list()
predict_data_list = list()
show_last_list = list()

for i in range(0, window_len+predict_len):
    last_list.append(float(get_response().read().split("last")[1].split("high")[0].split(' ')[1].split(',')[0]))




app.layout = html.Div([
    html.Div([
        html.H2("Bitcoin Prediction"),
        html.Img(src="https://bitcoin.org/img/icons/opengraph.png"),
    ], className='banner'),
    html.Div([
        html.Div([
            html.H3("Bitcoin Price (USD)")
        ], className='Title'),
        html.Div([
            dcc.Graph(id='wind-speed'),
        ], className='twelve columns wind-speed'),
        dcc.Interval(id='wind-speed-update', interval=1000, n_intervals=0),
    ], className='row wind-speed-row'),
], style={'padding': '0px 10px 15px 10px',
          'marginLeft': 'auto', 'marginRight': 'auto', "width": "900px",
          'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'})






@app.callback(Output('wind-speed', 'figure'), [Input('wind-speed-update', 'n_intervals')])
def gen_wind_speed(interval):
    
    actual_response = get_response().read()
    actual_last = float(actual_response.split("last")[1].split("high")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    actual_bid = float(actual_response.split("bid")[1].split("last")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    actual_ask = float(actual_response.split("ask")[1].split("bid")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    actual_volume = float(actual_response.split("volume")[1].split("changes")[0].replace('"','').\
                    replace(':','').replace(',','').replace(' ','').replace('\n',''))
    global table
    table.append([actual_ask, actual_bid, actual_last, actual_volume])

    table = table[-(window_len + predict_len):]
    global predict_data_list
    
    if len(table) == window_len+predict_len:
        predict_data = predict_model(table)
        predict_data_list.append(predict_data)
        predict_data_list = predict_data_list[-slide_window:]
        total_data = 0.0
        for i in range(len(predict_data_list)):
            total_data = total_data + predict_data_list[i]
        predict_data = total_data / len(predict_data_list)
    else:
        predict_data = actual_last

    # last_price = float(get_response().split("last")[1].split("high")[0].split(' ')[1].split(',')[0])
    global last_list
    global show_last_list
    last_list.append(actual_last)
    last_list = last_list[-(200+window_len+predict_len+slide_window/2):]
    show_last_list = last_list[:][-(200+window_len+predict_len+slide_window/2):-(window_len+predict_len+slide_window/2)]

    global predict_list
    predict_list.append(predict_data)
    predict_list = predict_list[-200:]
    
    
    trace_actual = Scatter(
        #y=df['Speed'],
        #y=last_list[-(200+window_len+predict_len+10):-(window_len+predict_len+10)],
        x=range(200-slide_window/2),
        y=show_last_list[:][-200+slide_window/2:],
        line=Line(
            color='#fab915'
        ),
        hoverinfo='skip',
        mode='lines',
        name='actual price'
    )

    trace_predict = Scatter(
        y=predict_list,
        line=Line(
            color='#4286f4'
        ),
        hoverinfo='skip',
        mode='lines',
        name='predict price'
    )

    lastest_predict = 0.0
    lastest_actual = 0.0
    if len(predict_list) > 0 and len(show_last_list) > 0:
        lastest_predict = predict_list[-1]
        lastest_actual = show_last_list[-1]


    predict_arrow = 0
    last_arrow = 0
    if len(show_last_list) > 10 and len(predict_list) > 10:
        if predict_list[-slide_window/2] > lastest_actual:
            predict_arrow = -60
            last_arrow = 60
        else:
            predict_arrow = 60
            last_arrow = -60

    total_error = 0.0
    average_error = 0.0
    for i in range(len(show_last_list)):
        total_error = total_error + abs(predict_list[i]-show_last_list[i])/show_last_list[i]
    if len(show_last_list) > 0:
        average_error = total_error/len(show_last_list)
    print "total error: " + str(total_error)
    print "average: "+str(average_error)

    layout = Layout(
        height=450,
        annotations=[
            dict(
                x=200,
                y=lastest_predict,
                xref='x',
                yref='y',
                text=("%.2f" % lastest_predict),
                showarrow=True,
                arrowhead=7,
                arrowcolor='#4286f4',
                ax=0,
                ay=predict_arrow
            ),
            dict(
                x=200-slide_window/2,
                y=lastest_actual,
                xref='x',
                yref='y',
                text=("%.2f" % lastest_actual),
                showarrow=True,
                arrowhead=7,
                arrowcolor='#fab915',
                ax=0,
                ay=last_arrow
            )
        ],
        title='average error: {:.4%}'.format(average_error),
        xaxis=dict(
            range=[0, 200],
            showgrid=True,
            showline=True,
            zeroline=True,
            fixedrange=True,
            tickvals=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
            ticktext=['200', '180', '160', '140', '120', '100', '80', '60', '40', '20', '0'],
            title='Time Elapsed (sec)'
        ),
        yaxis=dict(
            showline=False,
            fixedrange=True,
            zeroline=False,
        ),
        margin=Margin(
            t=45,
            l=50,
            r=50,
            b=45,
        )
    )

    return Figure(data=[trace_actual,trace_predict], layout=layout)





external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "https://zxw018018.github.io/bitcoin.css",
                "https://fonts.googleapis.com/css?family=Raleway:400,400i,700,700i",
                "https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i"]


for css in external_css:
    app.css.append_css({"external_url": css})

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

if __name__ == '__main__':
    #app.run_server(os.environ['PORT'])
    app.run_server(port=8050)
