# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import datetime


import plotly
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

from dash.dependencies import Input, Output
from urllib2 import Request, urlopen, URLError
import time

def get_last():
	request = Request("https://apiv2.bitcoinaverage.com/indices/global/ticker/BTCUSD")
	response = urlopen(request)
	last = response.read().split("last")[1].split("high")[0].split(' ')[1].split(',')[0]
	return last


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H2("Bitcoin Price Prediction")
    ], className='banner'),
    html.Div([
        html.Div([
            html.H3("Bitcoin Realtime Price")
        ], className='Title'),
        html.Div(id='live-update-text'),
        html.Div([
            dcc.Graph(id='prediction'),
        ], className='twelve columns prediction'),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    ], className='row prediction-row'),
])


# , style={'padding': '0px 10px 15px 10px',
#           'marginLeft': 'auto', 'marginRight': 'auto', "width": "900px",
#           'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'})




@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(interval):
    last = get_last()
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Last: {}'.format(last), style=style),
    ]



@app.callback(Output('prediction', 'figure'), [Input('interval-component', 'n_intervals')])
def gen_bitcoin(interval):
	trace = Scatter(
        y=get_last(),
        line=Line(
            color='#42C4F7'
        ),
        hoverinfo='skip',
        # error_y=ErrorY(
        #     type='data',
        #     array=df['SpeedError'],
        #     thickness=1.5,
        #     width=2,
        #     color='#B4E8FC'
        # ),
        mode='lines'
    )

	layout = Layout(
    	height=450,
        # xaxis=dict(
        #     range=[0, 200],
        #     showgrid=False,
        #     showline=False,
        #     zeroline=False,
        #     fixedrange=True,
        #     tickvals=[0, 50, 100, 150, 200],
        #     ticktext=['200', '150', '100', '50', '0'],
        #     title='Time Elapsed (sec)'
        # ),
        # yaxis=dict(
        #     range=[min(0, min(df['Speed'])),
        #            max(45, max(df['Speed'])+max(df['SpeedError']))],
        #     showline=False,
        #     fixedrange=True,
        #     zeroline=False,
        #     nticks=max(6, round(df['Speed'].iloc[-1]/10))
        # ),
        margin=Margin(
            t=45,
            l=50,
            r=50
        )
    )

	return Figure(data=[trace], layout=layout)

	
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

if __name__ == '__main__':
    app.run_server(debug=True)