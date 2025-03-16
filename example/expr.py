from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        pass
    
    """
        test: 'test';
    """
    @lexer
    def test(self):
        g = newTerminalRule("test")
        print(g.text())
        return g
    
    
    """
        test1: test*;
    """
    @lexer
    def test1(self):
        g = newTerminalRule()
        g.setProduction(one_or_more(self.test()))
        print(g.text())
        return g




mygram = MyGrammar()
mylexer = mygram.lexer()
# myparser = mygram.parser()
