import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from charts import main3d, main, histogram
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import flask
import plotly.plotly as py
from plotly import graph_objs as go
import math
import base64
import datetime
import io
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div([
    html.Div([

        html.Span("CUDA Threads Profiling Tool Analitic Module v2", className='app-title')]),

    dcc.Upload(
        id='upload-csv',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='original_data', style={"display": "none"}),


    # tabs
    html.Div([

        dcc.Tabs(
            id="tabs",
            style={"height": "20", "verticalAlign": "middle"},
            children=[
                dcc.Tab(
                    label='Main',
                    value="main_tab",
                    children=[
                        html.Div(children=[''], style={"width": "20px", "height": "20px"}), # a little hack to make label visible
                        html.Div(id="main-graph")
                    ]),
                dcc.Tab(
                    label="Main chart 3d",
                    value="main3d_tab",
                    children=[
                        html.Div(children=[''], style={"width": "20px", "height": "20px"}),
                        html.Div(id="main3d-graph")
                    ]),
                dcc.Tab(
                    label="Histogram",
                    value="histogram_tab",
                    children=[
                        html.Div(children=[''], style={"width": "20px", "height": "20px"}),
                        html.Div(id="histogram-graph")
                    ])
            ],
            value="main_tab",
        )

    ],
        className="row tabs_div"
    ),
])


def parse_contents(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    return df


@app.callback(Output("original_data", "children"), [Input('upload-csv', 'contents')])
def load_files(list_of_contents):
    if not list_of_contents:
        return None
    try:
        loaded = [parse_contents(c) for c in list_of_contents]
        df = pd.concat(loaded)
        return df.to_json()
    except Exception as ex:
        print(ex)
        return "Error"


@app.callback(Output("main-graph", "children"), [Input("original_data", "children")])
def load_graph_for_main(data):
    result, data = validate_graph_data(data)
    if result:
        return main.layout(data)
    return data


@app.callback(Output("main3d-graph", "children"), [Input("original_data", "children")])
def load_graph_for_main3d(data):
    result, data = validate_graph_data(data)
    if result:
        return main3d.layout(data)
    return data


@app.callback(Output("histogram-graph", "children"), [Input("original_data", "children")])
def load_graph_for_histogram(data):
    result, data = validate_graph_data(data)
    if result:
        return histogram.layout(data)
    return data


def validate_graph_data(data):
    if not data:
        return False, dcc.Graph()
    try:
        data = pd.read_json(data)
        return True, data
    except:
        return False,  html.Div([
            'There was an error processing this file.',
            dcc.Graph()
        ], style={"margin": "10px"})


if __name__ == '__main__':
    app.run_server(debug=True)
