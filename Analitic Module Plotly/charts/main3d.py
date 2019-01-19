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
        internal_data = data[data.x != -1]
        self.labels = data.label.unique()
        self.x.clear()
        self.y.clear()
        self.z.clear()
        self.text.clear()

        for label in self.labels:
            time = internal_data[internal_data['label'] == label]['time']
            self.x.append(time)

            x_id = internal_data[internal_data['label'] == label]['x']
            self.y.append(x_id)

            y_id = internal_data[internal_data['label'] == label]['y']
            self.z.append(y_id)

            text = internal_data[internal_data['label'] == label]['label']
            self.text.append(text)

    def get_content(self,):
        layout = dcc.Graph(
            id='main_graph' + str(uuid.uuid4()),
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
                'layout': go.Layout(scene = dict(xaxis = dict(title='Time'),
                    yaxis = dict(title='X-id'),
                    zaxis = dict(title='Y-id')),
                    margin={'l': 180, 'b': 150, 't': 150, 'r': 10},
                    showlegend=True,
                    hovermode='closest',
                    font=dict(family='Courier New, monospace', size=22, color='#7f7f7f')
                )
            })
        return layout
