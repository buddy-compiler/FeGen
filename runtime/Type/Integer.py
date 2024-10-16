from .Type import Type


class Integer(Type):
    def __init__(self, size: int = 32) -> None:
        self.size = size
        super().__init__()
