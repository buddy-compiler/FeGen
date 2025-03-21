from FeGen.AttributeGrammar import *


class MyGram(FeGenGrammar):
    def __init__(self):
        super().__init__()
        self.record = ""
    
    @execute_when("sema")
    def log(self, msg):
        self.record += msg
    
    @lexer
    def TEST(self):
        return newTerminalRule("TEST")
    
    @parser
    def rule1(self):
        self.log("visit rule1")
        t = self.TEST()
        g = newParserRule(t)
        g.set_attr("test", t.getText())
        return g


def test_visit():
    mygram = MyGram()
    lexer = mygram.lexer()
    parser = mygram.parser(lexer, "rule1")
    root = parser.parse("TEST")
    root.get_attr("test")
    root.get_attr("test")
    root.visit()
    root.visit()