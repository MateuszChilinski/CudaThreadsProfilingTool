import pandas as pd
import os
from .label import Label


class Plot():
    def __init__(self, filepath, axis, title=None):
        self.subplots = {}
        self.axis = axis
        self.hidden = False
        self.scatters = {}
        data = pd.read_csv(filepath, names=["id", "timestamp", "label"])

        for label_title, label_data in data.groupby('label'):
            l = Label(label_title)
            self.subplots[l] = label_data[['id', 'timestamp']]

        if title is None:
            _, title = os.path.split(filepath)
            title, _ = os.path.splitext(title)

        self.title = title

    def __str__(self):
        if self.hidden:
            return "{} (Hidden)".format(self.title)
        return self.title

    def get_labels(self):
        return self.subplots.keys()

    def get_label(self, index):
        return self.subplots.keys()[index]

    def get_title(self):
        return self.title

    def set_title(self, value):
        self.title = value

    def change_hidden(self):
        self.hidden = self.hidden == False
        if self.hidden:
            self.remove_plot()
        else:
            self.add_plot()

    def get_max_timestamp(self):
        max = 0
        for label in self.subplots.keys():
            value = self.subplots[label]['timestamp'].max()
            if max < value:
                max = value
        return max

    def set_hidden(self, value):
        self.hidden = value

    def get_hidden(self):
        return self.hidden

    def add_plot(self):
        for label in self.subplots.keys():
            if not label.hidden:
                self.add_label(label)

    def delete_plot(self):
        self.remove_plot()
        self.subplots.clear()

    def remove_plot(self):
        for label in self.scatters.keys():
            scatter = self.scatters[label]
            scatter.remove()
        self.scatters.clear()

    def remove_label(self, label):
        scatter = self.scatters[label]
        scatter.remove()
        label.set_hidden(True)
        del self.scatters[label]

    def add_label(self, label):
        data = self.subplots[label]
        label.set_hidden(False)
        scatter = self.axis.scatter(
            'timestamp',
            'id',
            data=data,
            color=label.get_color(),
            label="{} {}".format(self.title, label))
        color = self.convert_to_rgb(scatter.get_facecolor()[0])
        label.set_color(color)

        self.scatters[label] = scatter

    def convert_to_rgb(self, face_color):
        red = face_color[0]
        green = face_color[1]
        blue = face_color[2]
        alpha = face_color[3]
        return (red, green, blue, alpha)
