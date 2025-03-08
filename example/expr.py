from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

@attr_grammar
class MyGrammar(FeGenGrammar):
    def __init__(self):
        pass
    
    @lexer
    def test(self):
        g = newTerminalRule("test")
        print(g.text())
        g.name = "test"
        return g
    
    
    @lexer
    def test1(self):
        g = newTerminalRule()
        g.setProduction(one_or_more(self.test()))
        print(g.text())
        return g
        

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser()
