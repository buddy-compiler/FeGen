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
        g_expr = self.add_expr()
        return newParserRule(g_expr)
        
    @parser
    def add_expr(self):
        g = newParserRule()
        def alt1():
            lhs = self.add_expr()
            rhs = self.prim_expr()
            return concat(lhs, self.Add(), rhs)
        
        def alt2():
            return self.prim_expr()
        
        g.setProduction(alternate(alt1, alt2))
        return g
    
    @parser
    def prim_expr(self):
        def alt1():
            return self.Number()
        
        def alt2():
            return concat(self.LB(), self.expression(), self.RB())
        
        g = newParserRule()
        g.setProduction(alternate(alt1, alt2))
        return g
    
    
    

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer, "expression")
code = "1+(2+3)"
tree = myparser.parse(code)
tree.eval()
print(tree.getText())