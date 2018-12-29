import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
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


class Main3d():
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.text = []
        self.labels = []

    def set_data(self, data):
        data = data[data.x != -1]
        self.labels = data.label.unique()
        for label in self.labels:
            self.x.append(data[data['label'] == label]['time'])
            self.y.append(data[data['label'] == label]['x'])
            self.z.append(data[data['label'] == label]['y'])
            self.text.append(data[data['label'] == label]['y'])

    def get_content(self,):
        layout = dcc.Graph(
            id='main_graph',
            style={"height": "78vh"},
            figure={
                'data': [
                    go.Scatter3d(
                        x=self.x[index],
                        y=self.y[index],
                        z=self.z[index],
                        text=self.text[index],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 5
                        },
                        name=label
                    ) for index, label in enumerate(self.labels)
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Time'},
                    yaxis={'title': 'X-id'},
                    margin={'l': 40, 'b': 40, 't': 50, 'r': 10},
                    showlegend=True,
                    hovermode='closest'
                )
            })
        return layout
