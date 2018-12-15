from tkinter import *
from tkinter import ttk
from .plot_window import PlotWindow


class MainWindow(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        PlotWindow(master)