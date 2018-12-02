from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
from tkinter.colorchooser import *
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
from .plot import Plot
from .object_listbox import ObjectListbox
from .number_entry import NumberEntry


class PlotWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.last_folder_selected = '.'
        fig = Figure()
        self.axis = fig.add_subplot(111)
        self.axis.set_xlabel('timestamp')
        self.axis.set_ylabel('id')

        # window grid
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=7)
        self.master.grid_columnconfigure(1, weight=4)

        left_panel = Frame(master=self.master)
        left_panel.grid(row=0, column=0, sticky=NSEW)
        self.figure = FigureCanvasTkAgg(fig, master=left_panel)
        NavigationToolbar2Tk(self.figure, left_panel)
        self.figure.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        right_panel = Frame(self.master)
        right_panel.grid(row=0, column=1, sticky=NSEW)

        Label(right_panel, text="Plots").pack()

        self.plots_listbox = ObjectListbox(right_panel)
        self.plots_listbox.pack(fill=X, padx=10)
        self.plots_listbox.bind('<<ListboxSelect>>', self.plot_selected)
        self.plots_listbox.bind('<Double-1>', self.change_plot_title)

        self.add_plot_button = Button(
            right_panel,
            text="Add plot",
            command=self.add_plot_click)
        self.add_plot_button.pack(fill=X, padx=10, pady=1)

        self.hide_plot_button = Button(
            right_panel,
            text="Hide/Show plot",
            command=self.change_plot_hidden,
            state=DISABLED)
        self.hide_plot_button.pack(fill=X, padx=10, pady=1)

        self.delete_plot_button = Button(
            right_panel,
            text="Delete plot",
            command=self.delete_plot_click,
            state=DISABLED)
        self.delete_plot_button.pack(fill=X, padx=10, pady=1)

        Label(right_panel, text="Labels").pack()
        self.labels_listbox = ObjectListbox(right_panel)
        self.labels_listbox.bind('<<ListboxSelect>>', self.label_selected)
        self.labels_listbox.bind(
            '<Double-1>',
            self.change_label_title)
        self.labels_listbox.pack(fill=X, padx=10, pady=1)

        self.hide_label_button = Button(
            right_panel,
            text="Hide/Show label",
            command=self.change_label_hidden,
            state=DISABLED)
        self.hide_label_button.pack(fill=X, padx=10, pady=1)

        self.change_label_color_button = Button(
            master=right_panel,
            text='Select Color',
            command=self.change_label_color,
            state=DISABLED)
        self.change_label_color_button.pack(fill=X, padx=10, pady=1)

        Label(right_panel, text="Timestamps range").pack()
        f = Frame(right_panel)
        f.pack(fill=X, padx=10)
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        self.min_range_entry = NumberEntry(f)
        self.max_range_entry = NumberEntry(f)

        Label(f, text="Min: ").grid(row=0, column=0)
        self.min_range_entry.grid(row=0, column=1, sticky=EW)
        Label(f, text="Max: ").grid(row=1, column=0)
        self.max_range_entry.grid(row=1, column=1, sticky=EW)

        self.apply_range_button = Button(
            right_panel,
            text="Apply range",
            command=self.apply_timestamp_range,
            state=DISABLED)
        self.apply_range_button.pack(fill=X, padx=10, pady=1)

        self.reset_range_button = Button(
            right_panel,
            text="Reset range",
            command=self.reset_timestamp_range,
            state=DISABLED)
        self.reset_range_button.pack(fill=X, padx=10, pady=1)

    ######################### PLOT REGION ######################################

    def change_plot_title(self, event):
        plot = self.plots_listbox.get_current()
        new_title = simpledialog.askstring(
            title="Change plot Title",
            prompt="New plot title",
            initialvalue=plot.get_title())

        if new_title is None:
            return
        elif new_title == '':
            self.show_empty_value_error()
        else:
            plot.set_title(new_title)
            self.plots_listbox.refresh_element(plot)

    def plot_selected(self, evt):
        current_plot = self.plots_listbox.get_current()
        if current_plot:
            self.labels_listbox.clear()
            self.labels_listbox.add_elements(current_plot.get_labels())
            self.refresh_plot()
            self.delete_plot_button['state'] = NORMAL
            self.hide_plot_button['state'] = NORMAL
        else:
            self.delete_plot_button['state'] = DISABLED
            self.hide_plot_button['state'] = DISABLED

    def add_plot_click(self):
        filepath = filedialog.askopenfilename(
            initialdir=self.last_folder_selected,
            title="Select file with data",
            filetypes=[("csv files", "*.csv")])
        if filepath:
            try:
                new_plot = Plot(filepath, self.axis)
                new_plot.add_plot()
                self.plots_listbox.add_element(new_plot, True)
                self.refresh_plot()
                self.validate_range_button_state()
                self.last_folder_selected, _ = os.path.split(filepath)
            except:
                messagebox.showerror(
                    title="Invalid file",
                    message="Could not load data from file {}".format(filepath))

    def change_plot_hidden(self):
        current_plot = self.plots_listbox.get_current()
        if current_plot:
            current_plot.change_hidden()
            self.plots_listbox.refresh_element(current_plot)
            self.refresh_plot()
            self.validate_range_button_state()
        else:
            self.show_plot_not_selected_info()

    def delete_plot_click(self):
        current_plot = self.plots_listbox.get_current()
        answer = messagebox.askquestion(
            "Delete plot",
            "Are You Sure?",
            icon='warning')
        if(answer != messagebox.YES):
            return
        if current_plot:
            self.labels_listbox.clear()
            self.plots_listbox.delete_element(current_plot)
            current_plot.delete_plot()
            self.refresh_plot()
            self.validate_range_button_state()
        else:
            self.show_plot_not_selected_info()

    ############################################################################

    ######################### LABELS REGION ####################################

    def change_label_title(self, event):
        label = self.labels_listbox.get_current()
        new_title = simpledialog.askstring(
            title="Change label Title",
            prompt="New label title",
            initialvalue=label.get_title())
        if new_title is None:
            return
        elif new_title == '':
            self.show_empty_value_error()
        else:
            label.set_title(new_title)
            self.labels_listbox.refresh_element(label)
            current_plot = self.plots_listbox.get_last_selected()
            current_plot.update_label(label)
            self.refresh_plot()

    def label_selected(self, evt):
        selected_label = self.labels_listbox.get_current()
        if selected_label:
            self.hide_label_button['state'] = NORMAL
            self.change_label_color_button['state'] = NORMAL
        else:
            self.hide_label_button['state'] = DISABLED
            self.change_label_color_button['state'] = DISABLED

    def change_label_hidden(self):
        selected_label = self.labels_listbox.get_current()
        if selected_label:
            plot = self.plots_listbox.get_last_selected()
            if selected_label.get_hidden():
                plot.add_label(selected_label)
            else:
                plot.remove_label(selected_label)
            self.labels_listbox.refresh_element(selected_label)
            self.refresh_plot()
        else:
            self.show_label_not_selected_info()

    def change_label_color(self):
        current_label = self.labels_listbox.get_current()
        if current_label:
            color = askcolor(initialcolor=current_label.get_255_color())
            if color[0] != None:
                color = tuple(map(int, color[0]))
                current_label.set_255_color(color)
                plot = self.plots_listbox.get_last_selected()
                if plot:
                    plot.update_label(current_label)
                self.refresh_plot()
        else:
            self.show_label_not_selected_info()

    ############################################################################

    ######################### RANGE REGION #####################################

    def validate_range_button_state(self):
        elements = filter(lambda x: not x.get_hidden(),
                          self.plots_listbox.get_elements())
        if any(elements):
            self.apply_range_button['state'] = NORMAL
            self.reset_range_button['state'] = NORMAL
        else:
            self.apply_range_button['state'] = DISABLED
            self.reset_range_button['state'] = DISABLED

    def apply_timestamp_range(self):
        if not self.plots_listbox.get_elements():
            messagebox.showinfo(
                title="No plot on figure",
                message="First add at least one plot")
            return
        range_min = self.convert_to_float(self.min_range_entry.get())
        range_max = self.convert_to_float(self.max_range_entry.get())
        message, valid = self.validate_range(range_min, range_max)
        if valid:
            self.axis.set_xlim(range_min, range_max)
            self.refresh_plot()
        else:
            messagebox.showerror(
                title='Invalid values',
                message=message
            )

    def reset_timestamp_range(self):
        self.axis.autoscale()
        self.refresh_plot()

    ############################################################################

    def refresh_plot(self):
        self.axis.legend()
        self.figure.draw()

    def show_empty_value_error(self):
        messagebox.showerror(
            title="Empty value",
            message="Value cannot be empty")

    def show_label_not_selected_info(self):
        messagebox.showinfo(
            title="Label not selected",
            message="Select label first")

    def show_plot_not_selected_info(self):
        messagebox.showinfo(
            title="Plot not selected",
            message="Select plot first")

    def validate_range(self, min, max):
        if min is None and max is None:
            return "", True
        if max == 0:
            return "Max cannot be zero", False
        if min is not None and max is not None and min >= max:
            return "Max must be greater than min", False
        return "", True

    def convert_to_float(self, value):
        try:
            return float(value)
        except:
            return None
