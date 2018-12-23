import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
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
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.scripts.config.serve_locally = True
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
    ),        # tabs
        html.Div([

            dcc.Tabs(
                id="tabs",
                style={"height":"20","verticalAlign":"middle"},
                children=[
                    dcc.Tab(label="Main", value="main_tab"),
                    dcc.Tab(label="Main chart 3d", value="main3d_tab"),
                    dcc.Tab(label="Histogram", value="histogram_tab"),
                ],
                value="main_tab",
            )

            ],
            className="row tabs_div"
            ),
        html.Div(id="tab_content", className="row", style={"margin": "2% 3%"})
    ])


def parse_contents(contents, filename, date):
    df = contents


@app.callback([Input('upload-csv', 'contents')],
              [State('upload-csv', 'filename'),
               State('upload-csv', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output("tab_content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "main_tab":
        return main.layout
    elif tab == "main3d_tab":
        return main3d.layout
    elif tab == "histogram_tab":
        return histogram.layout
    else:
        return main.layout

if __name__ == '__main__':
    app.run_server(debug=True)