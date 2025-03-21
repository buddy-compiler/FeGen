import ast

code = """
def test():
    a = 100 + 200
    print(a)
"""

tree = ast.parse(code)
unparsed = ast.unparse(tree)
new_code = compile(unparsed, filename="<code>", mode="exec")
exec(new_code)
test()