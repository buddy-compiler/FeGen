from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    """
        test: 'test';
    """
    @lexer
    def test(self):
        g = newTerminalRule("test")
        a = 10
        b = 20
        print(g.text())
        return g
    
    @parser
    def parse_test(self):
        g = newParserRule()
        t = self.test()
        t1 = self.test()
        print(t.text())
        g.setProduction(concat(t, t1))
        return g


mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer)
code = "test test"
print(myparser.parse(code))