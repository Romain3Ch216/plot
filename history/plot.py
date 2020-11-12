import json
import pdb
import sys
import numpy as np
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

json_file = sys.argv[1]

with open(json_file, 'r') as outfile:
    logs = outfile.read()
    logs = json.loads(logs)
    metrics = list(logs['1'].keys())
    history = {}
    logs = logs.values()
    logs = [list(item.values()) for item in logs]
    logs = np.array(logs)
    for i, key in enumerate(metrics):
        history[key] = logs[:,i]
    grad = history['grad']
    grad = np.array([list(g.values()) for g in grad])


fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=history['train_loss'],
                    mode='lines',
                    name='train loss'))
fig1.add_trace(go.Scatter(y=history['val_loss'],
                    mode='lines',
                    name='val loss'))
fig1.update_layout(xaxis_title='Epochs',
                   yaxis_title='Loss')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=history['train_accuracy'],
                    mode='lines',
                    name='train accuracy'))
fig2.add_trace(go.Scatter(y=history['val_accuracy'],
                    mode='lines',
                    name='val accuracy'))
fig2.update_layout(xaxis_title='Epochs',
                   yaxis_title='Accuracy')

fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=history['train_IoU'],
                    mode='lines',
                    name='train IoU'))
fig3.add_trace(go.Scatter(y=history['val_IoU'],
                    mode='lines',
                    name='val IoU'))
fig3.update_layout(xaxis_title='Epochs',
                   yaxis_title='Intersect over Union')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
        ]),

        html.Div([
            dcc.RadioItems(
                id='depth-selector',
                options=[{'label': 'Depth {}'.format(i), 'value': i} for i in range(grad.shape[1])],
                value=grad.shape[1]-1,
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='grad'),
        ])
    ], style={'margin': '40px 100px'})
])


@app.callback(
    Output('grad', 'figure'),
    Input('depth-selector', 'value'))
def update_graph(depth):
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=grad[:,depth],
                        mode='lines',
                        name='depth {}'.format(depth)))
    fig4.update_layout(xaxis_title='Epochs',
                       yaxis_title='Gradient abslute mean')
    return fig4

if __name__ == '__main__':
    app.run_server(debug=True)
