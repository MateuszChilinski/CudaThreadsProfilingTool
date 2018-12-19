import unittest
from ..object_listbox import ObjectListbox

class TestObjectListbox(unittest.TestCase):

    def test_addElements_shouldAddAllElementsToList(self):
       listbox = ObjectListbox(None)
       expect = ["1","2","3","4"]
       result =listbox.add_elements(expect)
       self.assertEqual(listbox.elements,expect)

    def test_addElement_shouldAddElementToList(self):
       listbox = ObjectListbox(None)
       listbox.elements = ["1","2","3","4"]
       result =listbox.add_elements("5")
       self.assertEqual(listbox.elements, ["1","2","3","4","5"])

    def test_clear_shouldRemoveAllElements(self):
        listbox = ObjectListbox(None)
        listbox.elements = ["1","2","3","4"]
        result =listbox.clear()
        self.assertEqual(listbox.elements, [])

    def test_deleteExistingElement_shouldRemoveOneElement(self):
        listbox = ObjectListbox(None)
        listbox.elements = ["1","2","3","4"]
        result =listbox.delete_element("2")
        self.assertEqual(listbox.elements, ["1","3","4"])

    def test_deleteNonExistingElement_shouldRaiseError(self):
        listbox = ObjectListbox(None)
        listbox.elements = ["1","2","3","4"]
        self.assertRaises(ValueError, listbox.delete_element, "5")
        self.assertEqual(listbox.elements, ["1","2","3","4"])

    def test_getCurrend_shouldRemoveOneElement(self):
        listbox = ObjectListbox(None)
        listbox.elements = ["1","2","3","4"]
        self.assertRaises(ValueError, listbox.delete_element, "5")

    def test_getCurrentOnNotSelectedListBox_ShouldReturnNone(self):
        listbox = ObjectListbox(None)
        current = listbox.get_current()
        self.assertEqual(current,None)

if __name__ == '__main__':
    unittest.main()