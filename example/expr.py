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
    
    @lexer
    def LB(self):
        return newTerminalRule("\(")
    
    @lexer
    def RB(self):
        return newTerminalRule("\)")
    
    
    @parser
    def expression(self):
        g = newParserRule()
        g_expr = self.add_expr()
        g.setProduction(g_expr)
        g.set_attr("value", g_expr.get_attr("value"))
        return g
        
    @parser
    def add_expr(self):
        g = newParserRule()
        def alt1():
            g_lhs = self.add_expr()
            g_rhs = self.prim_expr()
            lhs = g_lhs.get_attr("value")
            rhs = g_rhs.get_attr("value")
            g.set_attr("value", lhs + rhs)
            return concat(g_lhs, self.Add(), g_rhs)
        
        def alt2():
            g_prim_expr = self.prim_expr()
            g.set_attr("value", g_prim_expr.get_attr("value"))
            return g_prim_expr
        
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def prim_expr(self):
        g = newParserRule()
        def alt1():
            g_num = self.Number()
            value = int(g_num.getText())
            g.set_attr("value", value)
            return g_num
        
        def alt2():
            g_expr = self.expression()
            value = g_expr.get_attr("value")
            g.set_attr("value", value)
            return concat(self.LB(), g_expr, self.RB())
        
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    
    

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer, "expression")
code = "1+(2+3)"
tree = myparser.parse(code)
tree.__eval()
print(tree.getText())
print(tree.get_attr("value"))