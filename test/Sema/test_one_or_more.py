from FeGen.AttributeGrammar import *

class MyGram(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    
    @lexer
    def TEST2(self):
        return newTerminalRule("TEST2")

    @parser
    def test2(self):
        g = newParserRule(self.TEST2())
        self.log("visit test2")
        g.set_attr("b", "attribute of test2")
        return g

    @sema
    def log(self, msg):
        print(msg)
    
    @parser
    def one_or_more_test2(self):
        g = newParserRule()
        g_test2 = self.test2()
        g_more = one_or_more(g_test2)  
        # __getitem__
        try:
            t = g_more[2]
            print(t.get_attr("b"))
        except IndexError as e:
            print("error: ", e)
        # __iter__
        for idx, t in enumerate(g_more):
            print(idx, ": ", t.getText())
        # set_attr
        try:
            g_more.set_attr("error", "error")
        except ProductionSemaError as e:
            print(e)
        # getattr
        attr = g_more.get_attr("b")
        print(attr)
        # visit
        g_more.visit()
        g.setProduction(g_more)
        return g
    

def test_one_or_more():
    g = MyGram()
    lexer = g.lexer()
    parser = g.parser(lexer, "one_or_more_test2")
    print("input: TEST2")
    tree = parser.parse("TEST2 TEST2")
    tree.visit()
