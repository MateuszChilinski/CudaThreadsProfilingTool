import unittest
from ..number_entry import NumberEntry

class TestNumberEntry(unittest.TestCase):

    def test_validate_actionRemove_shouldReturnTrue(self):
       entry = NumberEntry(None)
       result =entry.validate('0', None, None, None, None, None, None, None)
       self.assertTrue(result)

    def test_validate_actionFocusInOut_shouldReturnTrue(self):
       entry = NumberEntry(None)
       result =entry.validate('-1', None, None, None, None, None, None, None)
       self.assertTrue(result)

    def test_validate_actionInsertValueText_shouldReturnFalse(self):
       entry = NumberEntry(None)
       result =entry.validate('1', None, None, None, "test", None, None, None)
       self.assertFalse(result)

    def test_validate_actionInsertValueInt_shouldReturnTrue(self):
       entry = NumberEntry(None)
       result =entry.validate('1', None, "42", "4", "2", None, None, None)
       self.assertTrue(result)

    def test_validate_actionInsertIncorrectFloat_shouldReturnFalse(self):
       entry = NumberEntry(None)
       result =entry.validate('1', None, "42.-", "42.", "-", None, None, None)
       self.assertFalse(result)

    def test_validate_actionInsertCorrectFloat_shouldReturnTrue(self):
       entry = NumberEntry(None)
       result =entry.validate('1', None, "42.5", "42.", "5", None, None, None)
       self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()