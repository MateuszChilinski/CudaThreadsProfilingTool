import pandas as pd
import os
from .label import Label
import matplotlib.colors as mcolors


class Plot():
    def __init__(self, filepath, axis, title=None):
        self.subplots = {}
        self.axis = axis
        self.hidden = False
        self.scatters = {}
        self.data = pd.read_csv(filepath, names=["id", "timestamp", "label"])
        if title is None:
            _, title = os.path.split(filepath)
            title, _ = os.path.splitext(title)

        self.title = title
    
    def create_plot(self, update=False):
        self.type = 1
        for label_title, label_data in self.data.groupby('label'):
            if update == False:
                l = Label(label_title)
            else:
                l = next(x for x in self.subplots if x.get_title() == label_title)
            self.subplots[l] = label_data[['id', 'timestamp']]
            
        if update == False:
            self.add_plot()
        else:
            self.ez_update_plot()

    def create_histogram(self, update=False):
        self.type = 2
        mydata = self.data.groupby(['label', 'timestamp']).size().reset_index().sort_values(['label','timestamp'])
        mydata.columns = ['label', 'timestamp', 'size']
        mydata = mydata.reset_index(drop=True)
        nmydata = mydata.values
        for idx, val in enumerate(nmydata):
            if idx == 0 :
                continue
            if val[0] == nmydata[idx-1][0]:
                val[2] += nmydata[idx-1][2]
        mydata = pd.DataFrame(data=nmydata[0:,0:],
        index=enumerate(nmydata),
        columns=['label', 'timestamp', 'size'])
        mydata = mydata.groupby(['label'])
        for label_title, label_data in mydata:
            if update == False :
                l = Label(label_title)
            else:
                l = next(x for x in self.subplots if x.get_title() == label_title)
            self.subplots[l] = label_data[['size', 'timestamp']]

        if update == False:
            self.add_plot()
        else:
            self.ez_update_plot()

    def __str__(self):
        if self.hidden:
            return "{} (Hidden)".format(self.title)
        return self.title

    def get_labels(self):
        return self.subplots.keys()

    def set_title(self, value):
        self.title = value
        for label in self.scatters.keys():
            self.update_label(label)

    def get_title(self):
        return self.title

    def change_hidden(self):
        self.hidden = self.hidden == False
        if self.hidden:
            self.remove_plot()
        else:
            self.add_plot()

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
            if(isinstance(scatter, list)):
                scatter[0].remove()
            else:
                scatter.remove()
        self.scatters.clear()

    def remove_label(self, label):
        scatter = self.scatters[label]
        if(isinstance(scatter, list)):
            scatter[0].remove()
        else:
            scatter.remove()
        label.set_hidden(True)
        del self.scatters[label]
    def ez_update_plot(self):
        for label in self.subplots.keys():
            if label in self.scatters:
                scatter = self.scatters[label]
                if(isinstance(scatter, list)):
                    scatter[0].remove()
                else:
                    scatter.remove()
                del self.scatters[label]
                self.ez_update_label(label)
    def ez_update_label(self, label):
        data = self.subplots[label]          
        if self.type == 1:
            scatter = self.axis.scatter(
                'timestamp',
                'id',
                data=data,
                color=label.get_color(),
                label="{} {}".format(self.title, label))
            color = self.convert_to_rgb(scatter.get_facecolor()[0])
        elif self.type == 2:        
            data = self.subplots[label]
            label.set_hidden(False)
            scatter = self.axis.plot(
            'timestamp',
            'size',
            data=data,
            color=label.get_color(),
            label="{} {}".format(self.title, label))
            color = mcolors.to_rgba(scatter[0].get_markerfacecolor())
        label.set_color(color)

        self.scatters[label] = scatter

    def add_label(self, label):
        data = self.subplots[label]
        label.set_hidden(False)                
        if self.type == 1:
            scatter = self.axis.scatter(
                'timestamp',
                'id',
                data=data,
                color=label.get_color(),
                label="{} {}".format(self.title, label))
            color = self.convert_to_rgb(scatter.get_facecolor()[0])
        elif self.type == 2:        
            data = self.subplots[label]
            label.set_hidden(False)
            scatter = self.axis.plot(
            'timestamp',
            'size',
            data=data,
            color=label.get_color(),
            label="{} {}".format(self.title, label))
            color = mcolors.to_rgba(scatter[0].get_markerfacecolor())
        label.set_color(color)

        self.scatters[label] = scatter

    def update_label(self, label):
        if not label.get_hidden():
            self.scatters[label].set_color(label.get_color())
            self.scatters[label].set_label(
                "{} {}".format(self.title, label.get_title()))

    def convert_to_rgb(self, face_color):
        red = face_color[0]
        green = face_color[1]
        blue = face_color[2]
        alpha = face_color[3]
        return (red, green, blue, alpha)
