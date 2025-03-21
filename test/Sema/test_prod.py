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
        
    @parser 
    def number(self):
        g = newParserRule()
        g_n = self.Number()
        g.set_attr("v", int(g_n.getText()))
        g.setProduction(g_n)
        return g
        
    def zero_or_one_Number(self):
        g = newParserRule()
        g_num = self.Number()
        g_opt_num = zero_or_one(g_num)
        
        g.setProduction(g_opt_num)
        return g
        
    
def test_add_expr():
    mygram = MyGrammar()
    mylexer = mygram.lexer()
    myparser = mygram.parser(mylexer, "add_expr")
    code = "11+23"
    tree = myparser.parse(code)
    v = tree.get_attr("value")
    print(v)