from tkinter import Entry


class NumberEntry(Entry):
    def __init__(self, master):
        validate_cmd = (master.register(self.validate),
                        '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        Entry.__init__(
            self, master,
            validate='key',
            validatecommand=validate_cmd)

    def validate(self, action, index, value_if_allowed,
                 prior_value, text, validation_type, trigger_type, widget_name):
        if(action == '1'):
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
