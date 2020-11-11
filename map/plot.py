import argparse
import plotly.express as px
import pickle as pkl

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import torch
import numpy as np

import pdb

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

parser = argparse.ArgumentParser(description="Plot confusion_matrix")
parser.add_argument('--map', type=str,
                    help="Probability map file")

args = parser.parse_args()
config = parser.parse_args()
config = vars(config)

with open(args.map, 'rb') as f:
    data = pkl.load(f)
    probs, labels_, colors, rgb = data[0], data[1], data[2], data[3]

softmax = torch.nn.Softmax(dim=-1)
probs = torch.from_numpy(probs)
probs = softmax(probs)
probs = probs.numpy()
prob = np.max(probs, axis=-1)
map  = np.argmax(probs, axis=-1)
classes = np.unique(map).reshape(1,-1)
map  = convert_to_color_(map, colors)
legend = convert_to_color_(classes, colors)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = px.imshow(img=map, height=700)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    margin=dict(l=5, r=5, t=5, b=5),
)

fig_prob = px.imshow(img=prob, height=700)
fig_prob.update_xaxes(showticklabels=False)
fig_prob.update_yaxes(showticklabels=False)
fig_prob.update_layout(
    margin=dict(l=5, r=5, t=5, b=5),
)

fig_rgb = px.imshow(img=rgb, height=700)
fig_rgb.update_xaxes(showticklabels=False)
fig_rgb.update_yaxes(showticklabels=False)
fig_rgb.update_layout(
    margin=dict(l=5, r=5, t=5, b=5),
)

leg_fig = px.imshow(img=legend, width=600)
leg_fig.update_xaxes(showticklabels=False)
leg_fig.update_yaxes(showticklabels=False)
leg_fig.update_layout(annotations=[dict(x=i, y=0, text=labels_[i], showarrow=False, font={'size': 20}) for i in range(len(labels_))])


app.layout = html.Div([
    html.Div(children=[
        html.Div(className='column', children=[dcc.Graph(id='rgb', figure= fig_rgb)]),
        html.Div(className='column', children=[dcc.Graph(id='map', figure= fig)]),
        html.Div(className='column', children=[dcc.Graph(id='prob', figure= fig_prob)])],
        style={'display': 'flex', 'justify-content': 'space-around'}),
    dcc.Graph(figure=leg_fig)
    ])


if __name__ == '__main__':

    app.run_server(debug=True)
