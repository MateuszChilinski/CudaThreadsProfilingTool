from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from .plot import Plot
from .object_listbox import ObjectListbox
from tkinter.colorchooser import *


class MainWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.minsize(width=1024, height=768)
        self.plots = []
        fig = Figure()
        self.axis = fig.add_subplot(111)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=7)
        self.master.grid_columnconfigure(1, weight=4)

        left_panel = Frame(master=self.master)
        left_panel.grid(row=0, column=0, sticky='nswe')
        self.figure = FigureCanvasTkAgg(fig, master=left_panel)
        self.figure.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.right_panel = Frame(self.master)
        self.right_panel.grid(row=0, column=1, sticky='nswe')

        Label(self.right_panel, text="Plots").pack()

        self.plots_listbox = ObjectListbox(self.right_panel)
        self.plots_listbox.pack(fill=X, padx=10)
        self.plots_listbox.bind('<<ListboxSelect>>', self.plot_selected)

        add_plot_button = Button(
            self.right_panel,
            text="Add plot",
            command=self.add_plot_click)
        add_plot_button.pack(fill=X, padx=10, pady=1)

        delete_plot_button = Button(
            self.right_panel,
            text="Delete plot",
            command=self.delete_plot_click)
        delete_plot_button.pack(fill=X, padx=10, pady=1)

        Label(self.right_panel, text="Labels").pack()
        self.labels_listbox = ObjectListbox(self.right_panel)
        self.labels_listbox.pack(fill=X, padx=10, pady=1)

        hide_label_button = Button(
            self.right_panel,
            text="Hide/Show label",
            command=self.change_label_hidden)
        hide_label_button.pack(fill=X, padx=10, pady=1)

        change_label_color_button = Button(
            master=self.right_panel,
            text='Select Color',
            command=self.change_current_label_color)
        change_label_color_button.pack(fill=X, padx=10, pady=1)

    def change_current_label_color(self):
        current_label = self.labels_listbox.get_current()
        if current_label:
            color = askcolor()
            current_label.set_color(color[1])
            self.refresh_plot()

    def change_label_hidden(self):
        selected_label = self.labels_listbox.get_current()
        if selected_label:
            selected_label.change_hidden()
            self.labels_listbox.refresh_element(selected_label)
            self.refresh_plot()

    def plot_selected(self, evt):
        current_plot = self.plots_listbox.get_current()
        if current_plot:
            self.labels_listbox.clear()
            self.labels_listbox.add_elements(current_plot.get_labels())

    def delete_plot_click(self):
        current_plot = self.plots_listbox.get_current()
        if current_plot:
            self.plots.remove(current_plot)
            self.plots_listbox.delete_element(current_plot)
            self.labels_listbox.clear()
            self.refresh_plot()

    def add_plot_click(self):
        filepath = filedialog.askopenfilename(
            initialdir=".",
            title="Select file with data",
            filetypes=(("txt files", "*.txt"), ("csv files", "*.csv"), ("all files", "*.*")))
        if filepath:
            new_plot = Plot(filepath)
            self.plots.append(new_plot)
            new_plot.add_to_axis(self.axis)
            self.plots_listbox.add_element(new_plot)
            self.figure.draw()

    def refresh_plot(self):
        self.axis.clear()
        # TODO Find better way to refresh changes
        for plot in self.plots:
            plot.add_to_axis(self.axis)
        self.figure.draw()
