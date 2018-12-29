import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from charts import *
import dash
from dash.dependencies import Input, Output, State
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

    # tabs
    html.Div([
        dcc.Tabs(
            id="tabs",
            style={"height": "20", "verticalAlign": "middle"},
            children=[
                dcc.Tab(label='Main', value="main-tab"),
                dcc.Tab(label="Main chart 3d", value="main3d-tab"),
                dcc.Tab(label="Histogram", value="histogram-tab")
            ],
            value="main-tab",
        )

    ],
        className="row tabs_div"
    ),
    html.Div(id='intermediate-content-div', style={"display": "none"}),
    html.Div(id="tab-content", children=[dcc.Graph()], style={"margin": "20px"})
])

app_state = AppState()


def parse_contents(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    return df


@app.callback(Output("intermediate-content-div", "children"), [Input('upload-csv', 'contents')], [State("intermediate-content-div", "children")])
def load_files(list_of_contents, old_state):
    if list_of_contents is None:
        return None
    try:
        loaded = [parse_contents(c) for c in list_of_contents]
        df = pd.concat(loaded)
        app_state.set_data(df)
    except Exception as ex:
        print(ex)
        app_state.set_data_loading_error()
    return old_state


@app.callback(Output("tab-content", "children"), [Input("intermediate-content-div", "children"), Input("tabs", "value")])
def update_content(intermediate_value, current_tab):
    return app_state.get_content(current_tab)


if __name__ == '__main__':
    app.run_server(debug=True)
