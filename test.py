from types import MethodType

class test:
    def __init__(self):
        self.x = 100

# f = lambda self: print(self.x)

def f(self):
    print(self.x)

x = test()
x.f = MethodType(f, x)
setattr(x, "f", MethodType(f, x))
x.f()


