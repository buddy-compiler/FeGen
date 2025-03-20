from types import FunctionType

code = """
def test():
    global b
    print(b)
"""
g = {**globals()}
exec(code, g)
test : FunctionType = g["test"]
test.__globals__.update({"b": 100})
test()