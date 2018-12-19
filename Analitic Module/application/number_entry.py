from tkinter import Entry,Frame


class NumberEntry(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        validate_cmd = (self.register(self.validate),
                        '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.entry = Entry(
            self,
            validate='key',
            validatecommand=validate_cmd)
        self.entry.pack()

    def validate(self, action, index, value_if_allowed,
                 prior_value, text, validation_type, trigger_type, widget_name):
        if(action == '1'): #text was added
            if text in '0123456789.-+':
                try:
                    float(value_if_allowed)
                    return True
                except ValueError:
                    return False
            else:
                return False
        else:
            return True
