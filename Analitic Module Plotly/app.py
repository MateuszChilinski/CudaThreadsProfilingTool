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
                dcc.Tab(label="Main", value="main_tab"),
                dcc.Tab(label="Main chart 3d", value="main3d_tab"),
                dcc.Tab(label="Histogram", value="histogram_tab"),
            ],
            value="main_tab",
        )

    ],
        className="row tabs_div"
    ),


    # divs that save dataframe for each tab
    # html.Div(
    #        sf_manager.get_opportunities().to_json(orient="split"),  # opportunities df
    #        id="opportunities_df",
    #        style={"display": "none"},
    #    ),
    # html.Div(sf_manager.get_leads().to_json(orient="split"), id="leads_df", style={"display": "none"}), # leads df
    # html.Div(sf_manager.get_cases().to_json(orient="split"), id="cases_df", style={"display": "none"}), # cases df



    # Tab content
    html.Div(
        id="tab_content",
        className="row",
        style={"margin": "2% 3%"},
        children=[
            dcc.Graph(
                id='main_graph',
                config={"scrollZoom": True}
            )
        ]),
])


def parse_contents(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


@app.callback(Output("main_graph", "figure"), [Input("tabs", "value"), Input('upload-csv', 'contents')])
def render_content(tab, list_of_contents):
    children = []
    if list_of_contents is not None:
        children = [
            parse_contents(c) for c in
            list_of_contents]
    if not children and not 'df' in vars():
        return ""
    elif children:
        df = pd.concat(children)
    if tab == "main_tab":
        return main.figure(df)
    elif tab == "main3d_tab":
        return main3d.figure(df)
    elif tab == "histogram_tab":
        return histogram.figure(df)
    else:
        return main.figure(df)


if __name__ == '__main__':
    app.run_server(debug=True)
