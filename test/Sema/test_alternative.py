from FeGen.AttributeGrammar import *

class MyGram(FeGenGrammar):
    def __init__(self):
        super().__init__()
        
    
    @lexer
    def TEST1(self):
        return newTerminalRule("TEST1")
    
    @lexer
    def TEST2(self):
        return newTerminalRule("TEST2")

    @parser
    def test2(self):
        g = newParserRule(self.TEST2())
        g.set_attr("v", "attribute of test2")
        return g

    @sema
    def log(self, msg):
        print(msg)
    
    @parser
    def alt_test(self):
        g = newParserRule()
        def alt1():
            g_t1 = self.TEST1()
            self.log("visit alt_test.alt1")
            return g_t1
        
        def alt2():
            self.log("visit alt_test.alt2")
            return self.test2()
        
        g_alt = alternate(alt1, alt2)
        # get_actual_alt
        alt = g_alt.get_actual_alt()
        print(alt.getText())
        print(alt.get_attr("v"))
        # get_actual_alt_index
        alt_index = g_alt.get_actual_alt_index()
        print(alt_index)
        # set_attr
        g_alt.set_attr("a", "attribute setted from alt_test")
        # get_attr
        print(g_alt.get_attr("v"))
        print(g_alt.get_attr("a"))
        # visit
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    

def test_alt():
    g = MyGram()
    lexer = g.lexer()
    parser = g.parser(lexer, "alt_test")
    tree = parser.parse("TEST2")
    tree.visit()
    