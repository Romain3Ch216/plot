import pickle as pkl
import sys
import numpy as np
import pandas as pd
from metrics import confusion_matrix_analysis

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table

file_name = sys.argv[1]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

with open(file_name, 'rb') as f:
    data = pkl.load(f)
    global cm
    cm = data[0]
    labels = data[1]
    per_class, overall = confusion_matrix_analysis(cm)
    per_class = pd.DataFrame(per_class)
    per_class = per_class.T
    per_class.insert(0, 'Class', labels)
    overall = pd.DataFrame.from_dict(overall, orient='index')
    overall = overall.T
    overall.round(4)
    per_class.round(4)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(children=[
        html.Div([
            dcc.Dropdown(
                id='normalize',
                options=[{'label': 'Raw confusion matrix', 'value': 'raw'},
                         {'label': 'Normalized over predictions', 'value': 'pred'},
                         {'label': 'Normalized over ground truth', 'value': 'gt'}],
                value='raw',
                style={'width': '20em'}
            )]),
        dcc.Graph(id='conf_mat'),
        html.Div(
            children=[dash_table.DataTable(
                    id='per_class',
                    columns=[{"name": i, "id": i} for i in per_class.columns],
                    data=per_class.to_dict('records'))], style={'margin': '50px 0px'}),
        dash_table.DataTable(
            id='overall',
            columns=[{"name": i, "id": i} for i in overall.columns],
            data=overall.to_dict('records'))
    ], style={'margin': '50px 100px'})
])

@app.callback(
    Output('conf_mat', 'figure'),
    Input('normalize', 'value'))
def update_graph(mode):
    if mode == 'raw':
        data = cm
    elif mode == 'pred':
        data = 100 * cm / (np.sum(cm, axis=0)+1e-20)
        data = np.array(data).astype(int)
    elif mode == 'gt':
        data = 100 * cm.T / (np.sum(cm, axis=1)+1e-20)
        data = np.array(data.T).astype(int)

    fig = px.imshow(img=data,
                    labels=dict(x="Prediction", y="Ground Truth", z='Metric'),
                    x=labels,
                    y=labels,
                    height=750)
    fig.update_xaxes(side="top")
    return fig


if __name__ == '__main__':

    app.run_server(debug=True)
