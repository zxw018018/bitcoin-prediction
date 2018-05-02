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
from math import sqrt
from sklearn.metrics import mean_squared_error


app = dash.Dash()



app.layout = html.Div([
    html.Div([
        html.H2("Bitcoin Prediction"),
        html.Img(src="https://bitcoin.org/img/icons/opengraph.png"),
    ], className='banner'),
    html.Div([
        html.Div([
            html.H3("Next Day Bitcoin Price Pridiction(USD)")
        ], className='Title'),
        html.Div([
            dcc.Graph(id='wind-speed'),
        ], className='twelve columns wind-speed'),
        
    ], className='row wind-speed-row'),
    html.Div([ 
      html.H5("Start Date", id='start-date-css', style={'display': 'inline',
        'margin-bottom': 0, 'margin-right': '30px', 'margin-left': '45px'}),
      html.H5("End Date", style={'display': 'inline', 'margin-bottom': 0, 'margin-right': '37px', 'margin-left': '5px'}),
      html.H5("Epochs", style={'display': 'inline', 'margin-bottom': 0, 'margin-right': '50px', 'margin-left': '8px'}),
      html.H5("Window", style={'display': 'inline', 'margin-bottom': 0, 'margin-right': '30px', 'margin-left': '5px'}),
      html.H5("Neurons", style={'display': 'inline', 'margin-bottom': 0, 'margin-right': '44px', 'margin-left': '5px'}),
      html.H5("Dropout", style={'display': 'inline', 'margin-bottom': 0, 'margin-right': '30px', 'margin-left': '5px'}),

      ]),
    dcc.Input(id='start-date', type='text', value='2013-12-29', style={'width': '15%', 'margin-left': '30px'}),
    dcc.Input(id='end-date', type='text', value='2018-04-01', style={'width': '15%'}),
    dcc.Input(id='epoch', type='text', value='2', style={'width': '15%'}),
    dcc.Input(id='window-length', type='text', value='10', style={'width': '15%'}),
    dcc.Input(id='neurons', type='text', value='10', style={'width': '15%'}),
    dcc.Input(id='dropout', type='text', value='0.25', style={'width': '15%'}),
    html.Button(id='submit-button', n_clicks=0, children='Submit', style={'margin-left':'40%', 'color': 'white', 
      'background-color': 'forestgreen', 'font-size': '14px'}),
  
], style={'padding': '0px 10px 15px 10px',
          'marginLeft': 'auto', 'marginRight': 'auto', "width": "900px",
          'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'})




def build_model(inputs, output_size, neurons, dropout, activ_func="linear",
                loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


@app.callback(Output('wind-speed', 'figure'),
              [Input('submit-button', 'n_clicks')],
              [State('start-date', 'value'),
               State('end-date', 'value'),
               State('epoch', 'value'),
               State('window-length', 'value'),
               State('neurons', 'value'),
               State('dropout','value')])
def update_output(n_clicks, start_date_value, end_date_value, epoch_value, window_len_value, neuron_value, dropout_value):
  print 'start date is: ' + start_date_value
  print 'end date is: ' + end_date_value
  print 'epoch value is: ' + epoch_value
  print 'window length is: ' + window_len_value
  print 'neurons is: ' + neuron_value
  print 'dropout is: ' + dropout_value

  print 'start getting data'
  # get market info for bitcoin from the start of 2014 to the current day
  bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20131227&end="+time.strftime("%Y%m%d"))[0]
  # convert the date string to the correct date format
  bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
  # convert to int
  bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
  # look at the first few rows
  bitcoin_market_info.head()
  # load dataset
  #bitcoin_market_info = read_csv('bitcoin_data.csv', header=0, index_col=0)

  model_data = bitcoin_market_info[['Date','Open','High','Low','Close','Volume']]
  # need to reverse the data frame so that subsequent rows represent later timepoints
  model_data = model_data.sort_values(by='Date')
  model_data.head()

  print 'start washing data'


  model_data = bitcoin_market_info.sort_values(by='Date')
  model_data.head()
  start_date = start_date_value
  end_date = end_date_value
  model_data = model_data[model_data['Date']>=start_date]
  model_data = model_data[model_data['Date']<=end_date]
  model_data2 = model_data
  model_data = model_data.drop('Date', 1)



  print 'start spliting data'

  split = int(len(model_data)/10)
  training_set, test_set = model_data[0:split*9], model_data[split:]

  print 'start training data'

  window_len = int(window_len_value)
  norm_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
  LSTM_training_inputs = []
  for i in range(len(training_set)-window_len):
      temp_set = training_set[i:(i+window_len)].copy()
      for col in norm_cols:
          temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
      LSTM_training_inputs.append(temp_set)
  LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
  LSTM_test_inputs = []
  for i in range(len(test_set)-window_len):
      temp_set = test_set[i:(i+window_len)].copy()
      for col in norm_cols:
          temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
      LSTM_test_inputs.append(temp_set)
  LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1

  LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
  LSTM_training_inputs = np.array(LSTM_training_inputs)

  LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
  LSTM_test_inputs = np.array(LSTM_test_inputs)

  print 'start building model'

  model = build_model(LSTM_training_inputs, output_size=1, neurons = int(neuron_value), dropout = float(dropout_value))

  predict_len=1

  LSTM_test_inputs = []
  for i in range(len(test_set)-window_len-predict_len):
      temp_set = test_set[i:(i+window_len)].copy()
      for col in norm_cols:
          temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
      LSTM_test_inputs.append(temp_set)
  LSTM_test_outputs = (test_set['Close'][window_len+predict_len:].values/test_set['Close'][:-window_len-predict_len].values)-1

  LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
  LSTM_training_inputs = np.array(LSTM_training_inputs)

  LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
  LSTM_test_inputs = np.array(LSTM_test_inputs)

  print LSTM_training_inputs.shape

  history = model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=int(epoch_value), batch_size=1, verbose=2, shuffle=True, validation_data = (LSTM_test_inputs,LSTM_test_outputs))

  y1=training_set['Close'][window_len:]
  y2=((np.transpose(model.predict(LSTM_training_inputs))+1) * training_set['Close'].values[:-window_len])[0]
  rmse = sqrt(mean_squared_error(y1, y2))
  print('Test RMSE: %.3f' % rmse)
  print 'start ploting'

  trace_actual = Scatter(
    x=model_data2['Date'][window_len:split*9].astype(datetime.datetime),
    y=y1,
    line=Line(
        color='#fab915'
    ),
    hoverinfo='skip',
    mode='lines',
    name='actual price'
  )

  trace_predict = Scatter(
      x=model_data2['Date'][window_len+1:split*9+1].astype(datetime.datetime),
      y=y2,
      line=Line(
          color='#4286f4'
      ),
      hoverinfo='skip',
      mode='lines',
      name='predict price'
  )

  layout = Layout(
          title='Test RMSE: {:.3f}'.format(rmse)
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
    app.run_server(port=9000)
