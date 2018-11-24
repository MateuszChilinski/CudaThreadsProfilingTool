import pandas as pd
import os
from .label import Label


class Plot():
    def __init__(self, filepath, title=None):
        self.subplots = {}
        data = pd.read_csv(filepath)

        for label_title, label_data in data.groupby('label'):
            l = Label(label_title)
            self.subplots[l] = label_data[['id', 'timestamp']]

        if title is None:
            _, title = os.path.split(filepath)

        self.title = title

    def __str__(self):
        return self.title

    def get_labels(self):
        return self.subplots.keys()

    def get_label(self, index):
        return self.subplots.keys()[index]

    def add_to_axis(self, axis):
        for label in self.subplots.keys():
            if not label.hidden:
                data = self.subplots[label]
                data.plot.scatter(
                    x='timestamp',
                    y='id',
                    color=label.get_color(),
                    ax=axis,
                    label=label)
