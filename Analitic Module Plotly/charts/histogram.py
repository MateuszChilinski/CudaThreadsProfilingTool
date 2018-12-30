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
import numpy as np
import uuid


class Histogram():
    def __init__(self):
        self.kernels = []
        self.labels = []
        self.x = []
        self.y = []
        self.text = []
        self.kernels_lines = []

    def set_data(self, data):
        self.kernels = data[data.x == -1]
        data = data[data.x == 1]
        grouped = data.groupby(
            ['time', 'label']).size().reset_index(name='count')
        self.labels = grouped.label.unique()

        for label in self.labels:
            self.x.append(grouped[grouped['label'] == label]['time'])
            self.y.append(
                np.cumsum(grouped[grouped['label'] == label]['count']))
            self.text.append(grouped[grouped['label'] == label]['label'])

        self.__generate_kernel_lines()

    def __generate_kernel_lines(self):
        self.kernel_lines.clear()
        for _, row in self.kernels.iterrows():
            time = row['time']
            if row['label'].startswith("start_"):
                color = 'rgb(50, 171, 96)'
            else:
                color = 'rgb(220, 20, 60)'

            self.kernels_lines.append({
                'type': 'line',
                'yref': 'paper',
                'x0': time,
                'x1': time,
                'y0': 0,
                'y1': 1,
                'line': {
                        'color': color,
                        'width': 4,
                        'dash': 'dashdot',
                },
            })

    def get_content(self):
        layout = dcc.Graph(
            id='main_graph',
            style={"height": "78vh"},
            config={"scrollZoom": True},
            figure={
                'data': [
                    go.Scattergl(
                        x=self.x[index],
                        y=self.y[index],
                        text=self.text[index],
                        opacity=0.7,
                        name=label,
                        mode = 'lines+markers',
                    ) for index, label in enumerate(self.labels)
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Time'},
                    yaxis={'title': 'Count'},
                    margin={'l': 40, 'b': 40, 't': 50, 'r': 10},
                    shapes=self.kernels_lines,
                    showlegend=True,
                    hovermode='closest'
                )
            })

        return layout
