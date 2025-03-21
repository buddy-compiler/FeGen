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
    def zero_or_one_test2(self):
        g = newParserRule()
        g_test2 = self.test2()
        g_opt = zero_or_one(g_test2)
        # exist
        print(g_opt.exist())
        # set_attr
        g_opt.set_attr("a", "attribute set from zero_or_one_test2")
        # get_attr
        print(g_opt.get_attr("b"))
        print(g_opt.get_attr("a"))
        # visit
        g_opt.visit()
        # visit test2
        g_test2.visit()
        self.log(g_test2._ifexist)        
        g.setProduction(g_opt)
        return g
    

def test_opt():
    g = MyGram()
    lexer = g.lexer()
    parser = g.parser(lexer, "zero_or_one_test2")
    print("input: TEST2")
    tree = parser.parse("TEST2")
    tree.visit()
    print()
    print("input: ")
    tree = parser.parse(" ")
    tree.visit()
