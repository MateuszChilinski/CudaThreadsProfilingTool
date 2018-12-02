class Label():
    def __init__(self, title, color=None):
        self.title = title
        self.hidden = False
        self.color = color

    def __str__(self):
        if self.hidden:
            return "{} (Hidden)".format(self.title)
        return self.title

    def change_hidden(self):
        self.hidden = self.hidden == False

    def set_hidden(self, value):
        self.hidden = value

    def get_hidden(self):
        return self.hidden

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def get_255_color(self):
        red = int(self.color[0]*255)
        green = int(self.color[1]*255)
        blue = int(self.color[2]*255)
        return (red, green, blue)

    def set_title(self, title):
        self.title = title

    def get_title(self):
        return self.title
