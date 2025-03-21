from FeGen.AttributeGrammar import *

class MyGram(FeGenGrammar):
    def __init__(self):
        super().__init__()
        
    @sema
    def log(self, msg):
        print(msg)
    
    @lexer
    def TEST1(self):
        return newTerminalRule("TEST1")
    
    
    @sema
    def log(self, *msg):
        print(*msg)
    
    
    @parser
    def test1(self):
        g = newParserRule(self.TEST1())
        inh_attr = g.get_attr("inh_attr")
        self.log("test1 receive inh_attr from parent node: ", inh_attr)
        g.set_attr("syn_attr", "synthesized attribute from test1")
        return g
    
    @parser
    def tester(self):
        g = newParserRule()
        g_test1 = self.test1()
        g_test1.set_attr("inh_attr", "inherited attribute from tester")
        self.log("tester receive inh_attr from child node test1: ", g_test1.get_attr("syn_attr"))
        g.setProduction(g_test1)
        return g
    

def test_inh_syn():
    g = MyGram()
    lexer = g.lexer()
    parser = g.parser(lexer, "tester")
    tree = parser.parse("TEST1")
    tree.visit()
    