from tkinter import Tk
from application import MainWindow

if __name__ == "__main__":
    root = Tk()
    MainWindow(master=root)
    root.minsize(width=1024, height=768)
    root.mainloop()
