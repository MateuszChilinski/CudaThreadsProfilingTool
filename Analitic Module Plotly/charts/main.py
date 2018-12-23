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
    df_kernels = df[df.x == -1]
    df = df[df.x != -1]
    maxY = df['x'].max()*1.1
    kernels = []
    for index, row in df_kernels.iterrows():
        time = row['time']
        if row['label'].startswith("start_"):
            color = 'rgb(50, 171, 96)'
        else:
            color = 'rgb(220, 20, 60)'
        kernels.append({
                'type': 'line',
                'yref': 'y',
                'x0': time,
                'x1': time,
                'y0': 0,
                'y1': maxY,
                'line': {
                    'color': color,
                    'width': 4,
                    'dash': 'dashdot',
                },
            })

    layout = [dcc.Graph(
            id='main_graph' + str(uuid.uuid4()),
            figure={
                'data': [
                    go.Scattergl(
                        x=df[df['label'] == i]['time'],
                        y=df[df['label'] == i]['x'],
                        text=df[df['label'] == i]['y'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in df.label.unique()
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Time'},
                    yaxis={'title': 'X-id'},
                    margin={'l': 40, 'b': 40, 't': 50, 'r': 10},
                    shapes= kernels,
                    showlegend=True,
                    hovermode='closest'
                )
            }
        )]
    return layout