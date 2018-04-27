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

from urllib2 import Request, urlopen, URLError
import time

app = dash.Dash('streaming-wind-app')
server = app.server

global last_list
last_list = list()

def get_last():
    request = Request("https://apiv2.bitcoinaverage.com/indices/global/ticker/BTCUSD")
    response = urlopen(request)
    last = response.read().split("last")[1].split("high")[0].split(' ')[1].split(',')[0]
    return float(last)

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
    
    global last_list
    last_list.append(get_last())
    last_list = last_list[-200:]

    
    trace = Scatter(
        #y=df['Speed'],
        y=last_list,
        line=Line(
            color='#42C4F7'
        ),
        hoverinfo='skip',
        mode='lines'
    )

    layout = Layout(
        height=450,
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
            r=50
        )
    )

    return Figure(data=[trace], layout=layout)



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
    app.run_server()
