from tkinter import *


class ObjectListbox(Listbox):
    def __init__(self, master, cnf={}, **kv):
        Listbox.__init__(self, master, cnf)
        self.master = master
        self.elements = []

    def add_elements(self, elements):
        for element in elements:
            self.add_element(element)

    def add_element(self, element):
        super(ObjectListbox, self).insert(END, element)
        self.elements.append(element)

    def clear(self):
        super(ObjectListbox, self).delete(0, END)
        self.elements.clear()

    def refresh_element(self, element):
        element_index = self.elements.index(element)
        super(ObjectListbox, self).delete(element_index)
        super(ObjectListbox, self).insert(element_index, element)

    def delete_element(self, element):
        element_index = self.elements.index(element)
        super(ObjectListbox, self).delete(element_index)
        del self.elements[element_index]

    def get_current(self):
        current_selection = super(ObjectListbox, self).curselection()
        if current_selection:
            return self.elements[current_selection[0]]
        return None

    def get_elements(self):
        return self.elements
