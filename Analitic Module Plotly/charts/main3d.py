import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from charts import main3d
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import flask
import plotly.plotly as py
from plotly import graph_objs as go
import math
import uuid
def layout(df):
    df = df[df.x != -1]

    layout = [ dcc.Graph(
            id='main_graph3d' + str(uuid.uuid4()),
            figure={
                'data': [
                    go.Scatter3d(
                        x=df[df['label'] == i]['time'],
                        y=df[df['label'] == i]['x'],
                        z=df[df['label'] == i]['y'],
                        text=df[df['label'] == i]['y'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 5
                        },
                        name=i
                    ) for i in df.label.unique()
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Time'},
                    yaxis={'title': 'X-id'},
                    margin={'l': 40, 'b': 40, 't': 50, 'r': 10},
                    showlegend=True,
                    hovermode='closest'
                )
            })]
    return layout