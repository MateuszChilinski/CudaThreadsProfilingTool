from tkinter import *
from tkinter import ttk
from .plot_window import PlotWindow


class MainWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        tabControl = ttk.Notebook(master)
        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)
        PlotWindow(tab1)
        tabControl.add(tab1, text='Plot')
        tabControl.pack(expand=1, fill=BOTH)
        tabControl.add(tab2, text='Histogram')
        tabControl.pack(expand=1, fill=BOTH)
