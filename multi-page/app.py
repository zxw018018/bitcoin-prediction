import dash

app = dash.Dash('bitcoin-prediction')
server = app.server
app.config.suppress_callback_exceptions = True


