import sys

d = {}

def update_d():
    func_frame = sys._getframe(1)
    func_locals = func_frame.f_locals
    d.update(func_locals)

# @capture_locals
def example_function():
    a = 10
    b = 20
    c = a + b
    update_d()
    return c


print(d)
example_function()
print(d)
    