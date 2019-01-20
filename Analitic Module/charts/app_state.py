from .main import Main
from .main3d import Main3d
from .histogram import Histogram
import dash_html_components as html
import dash_core_components as dcc


class AppState():
    def __init__(self):
        self.data_loaded = False
        self.data_loading_error = False
        self.main = Main()
        self.main3d = Main3d()
        self.histogram = Histogram()

    def set_data(self, data):
        self.main.set_data(data)
        self.main3d.set_data(data)
        self.histogram.set_data(data)
        self.data_loaded = True
        self.data_loading_error = False

    def set_data_loading_error(self):
        self.data_loading_error = True

    def get_content(self, tab_name):
        if self.data_loading_error:
            return html.Div(children=[
                "Error during of loading files",
                dcc.Graph()])

        if self.data_loaded == False:
            return html.Div(children=[
                "No files uploaded",
                dcc.Graph()
            ])

        if tab_name == "main-tab":
            return self.main.get_content()
        elif tab_name == "main3d-tab":
            return self.main3d.get_content()
        elif tab_name == "histogram-tab":
            return self.histogram.get_content()
