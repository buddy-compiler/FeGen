from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    @lexer
    def Number(self):
        g = newTerminalRule("[1-9][0-9]*|[0-9]")
        print(g.text())
        return g

    @lexer
    def Identifier(self):
        g = newTerminalRule("[a-zA-Z_][a-zA-Z0-9_]*")
        print(g.text())
        return g            
    
    @lexer
    def Add(self):
        return newTerminalRule("\+")
    

    @parser
    def add_expr(self):
        g = newParserRule()
        g_lhs = self.Number()
        g_rhs = self.Number()
        g.setProduction(concat(g_lhs, self.Add(), g_rhs))
        lhs = int(g_lhs.getText())
        rhs = int(g_rhs.getText())
        g.set_attr("value", lhs + rhs)
        return g    
    

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer, "add_expr")
code = "11+23"
tree = myparser.parse(code)
tree.eval()
v = tree.get_attr("value")
print(v)