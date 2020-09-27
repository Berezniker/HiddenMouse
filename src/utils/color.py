class Color:
    def __init__(self):
        self.color = {
            'none': '\033[0m',
            'boldFont': '\033[1m',
            'italics': '\033[3m',
            'underline': '\033[4m',
            # character color:
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'purple': '\033[35m',
            'cyan': '\033[36m',
            'gray': '\033[37m',
            # background color:
            'bg-black': '\033[40m',
            'bg-red': '\033[41m',
            'bg-green': '\033[42m',
            'bg-yellow': '\033[43m',
            'bg-blue': '\033[44m',
            'bg-purple': '\033[45m',
            'bg-cyan': '\033[46m',
            'bg-gray': '\033[47m'
        }

    def __getitem__(self, item):
        # return self.color[item]  # color is only displayed in the console
        return ""  # for the .txt log


COLOR = Color()
