from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
from .plot import Plot
from .object_listbox import ObjectListbox
from tkinter.colorchooser import *


class MainWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.minsize(width=1024, height=768)
        fig = Figure()
        self.axis = fig.add_subplot(111)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=7)
        self.master.grid_columnconfigure(1, weight=4)

        left_panel = Frame(master=self.master)
        left_panel.grid(row=0, column=0, sticky=NSEW)
        self.figure = FigureCanvasTkAgg(fig, master=left_panel)
        new_window_button = Button(
            left_panel,
            text="Open in new window",
            command=self.open_in_new_window)
        new_window_button.pack(side=TOP, anchor=W)
        self.figure.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.right_panel = Frame(self.master)
        self.right_panel.grid(row=0, column=1, sticky=NSEW)

        Label(self.right_panel, text="Plots").pack()

        self.plots_listbox = ObjectListbox(self.right_panel)
        self.plots_listbox.pack(fill=X, padx=10)
        self.plots_listbox.bind('<<ListboxSelect>>', self.plot_selected)

        add_plot_button = Button(
            self.right_panel,
            text="Add plot",
            command=self.add_plot_click)
        add_plot_button.pack(fill=X, padx=10, pady=1)

        hide_label_button = Button(
            self.right_panel,
            text="Hide/Show plot",
            command=self.change_plot_hidden)
        hide_label_button.pack(fill=X, padx=10, pady=1)

        delete_plot_button = Button(
            self.right_panel,
            text="Delete plot",
            command=self.delete_plot_click)
        delete_plot_button.pack(fill=X, padx=10, pady=1)

        self.labels_label = Label(self.right_panel, text="Labels")
        self.labels_label.pack()
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

    def open_in_new_window(self):
        ax = plt.subplot(111)
        for element in self.plots_listbox.get_elements():
            element.add_to_axis(ax)
        plt.show()

    def change_plot_hidden(self):
        current_plot = self.plots_listbox.get_current()
        if current_plot:
            current_plot.change_hidden()
            self.plots_listbox.refresh_element(current_plot)
            self.refresh_plot()
        else:
            self.show_plot_not_selected_info()

    def change_current_label_color(self):
        current_label = self.labels_listbox.get_current()
        if current_label:
            color = askcolor()
            current_label.set_color(color[1])
            self.refresh_plot()
        else:
            self.show_label_not_selected_info()

    def change_label_hidden(self):
        selected_label = self.labels_listbox.get_current()
        if selected_label:
            selected_label.change_hidden()
            self.labels_listbox.refresh_element(selected_label)
            self.refresh_plot()
        else:
            self.show_label_not_selected_info()

    def show_label_not_selected_info(self):
        messagebox.showinfo(
            title="Label not selected",
            message="Select label first")

    def show_plot_not_selected_info(self):
        messagebox.showinfo(
            title="Plot not selected",
            message="Select plot first")

    def plot_selected(self, evt):
        current_plot = self.plots_listbox.get_current()
        if current_plot and current_plot:
            self.labels_label.config(text="Labels: {}".format(current_plot))
            self.labels_listbox.clear()
            self.labels_listbox.add_elements(current_plot.get_labels())

    def delete_plot_click(self):
        current_plot = self.plots_listbox.get_current()
        # TODO ask user before deletion
        if current_plot:
            self.plots_listbox.delete_element(current_plot)
            self.labels_listbox.clear()
            self.refresh_plot()
        else:
            self.show_plot_not_selected_info()

    def add_plot_click(self):
        filepath = filedialog.askopenfilename(
            initialdir=".",
            title="Select file with data",
            filetypes=(("txt files", "*.txt"), ("csv files", "*.csv"), ("all files", "*.*")))
        if filepath:
            try:
                new_plot = Plot(filepath)
                new_plot.add_to_axis(self.axis)
                self.plots_listbox.add_element(new_plot)
                self.figure.draw()
            except:
                messagebox.showerror(
                    title="Invalid file",
                    message="Could not load data from file {}".format(filepath))

    def refresh_plot(self):
        self.axis.clear()
        # TODO Find better way to refresh changes
        for plot in self.plots_listbox.get_elements():
            if not plot.get_hidden():
                plot.add_to_axis(self.axis)
        self.figure.draw()
