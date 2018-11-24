class Label():
    def __init__(self, title, color=None):
        self.title = title
        self.hidden = False
        if not color:
            color = 'b'
        self.color = color

    def __str__(self):
        if self.hidden:
            return "{} (Hidden)".format(self.title)
        return self.title

    def change_hidden(self):
        self.hidden = self.hidden == False

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color
