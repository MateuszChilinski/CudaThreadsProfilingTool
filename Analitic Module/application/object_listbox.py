from tkinter import *


class ObjectListbox(Listbox):
    def __init__(self, master, cnf={}, **kv):
        Listbox.__init__(self, master, cnf)
        self.master = master
        self.elements = []
        self.last_selected = None

    def add_elements(self, elements):
        for element in elements:
            self.add_element(element)

    def add_element(self, element, change_selection=False):
        super(ObjectListbox, self).insert(END, element)
        self.elements.append(element)
        if change_selection:
            self.change_selection()

    def change_selection(self, index=END):
        super(ObjectListbox, self).select_clear(0, END)
        super(ObjectListbox, self).select_set(index)
        super(ObjectListbox, self).event_generate("<<ListboxSelect>>")

    def clear(self):
        super(ObjectListbox, self).delete(0, END)
        self.elements.clear()

    def refresh_element(self, element):
        element_index = self.elements.index(element)
        super(ObjectListbox, self).delete(element_index)
        super(ObjectListbox, self).insert(element_index, element)
        self.change_selection(element_index)

    def delete_element(self, element):
        element_index = self.elements.index(element)
        super(ObjectListbox, self).delete(element_index)
        del self.elements[element_index]
        self.change_selection()

    def get_current(self):
        current_selection = super(ObjectListbox, self).curselection()
        if current_selection:
            self.last_selected = self.elements[current_selection[0]]
            return self.last_selected
        return None

    def get_last_selected(self):
        return self.last_selected

    def get_elements(self):
        return self.elements
