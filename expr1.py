from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    @lexer
    def Number(self):
        g = newTerminalRule(regular_expr("[1-9][0-9]*|[0-9]"))
        print(g.text())
        return g

    @lexer
    def Identifier(self):
        g = newTerminalRule(regular_expr("[a-zA-Z_][a-zA-Z0-9_]*"))
        print(g.text())
        return g            
    
    @lexer
    def Add(self):
        return newTerminalRule("+")
    
    @parser
    def expression(self):
        g_expr = self.add_expr()
        g_expr.visit()
        return newParserRule(g_expr)
        
    @parser
    def add_expr(self):
        g = newParserRule()
        g_lhs = self.prim_expr()
        g_rhs_elem = self.prim_expr()
        g_rhss = zero_or_more(concat(self.Add(), g_rhs_elem))
        for cat in g_rhss:
            print("*"*99)
            print(cat[1].get_attr("v"))
        g.setProduction(concat(g_lhs, g_rhss))
        return g
    
    @parser
    def prim_expr(self):
        g = newParserRule()
        g_num = self.Number()
        g.set_attr("v", int(g_num.getText()))
        g.setProduction(g_num)
        return g
    
    
    

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer, "expression")
code = "1"
tree = myparser.parse(code)
print(tree.getText())
tree.visit()