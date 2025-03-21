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
    
    @lexer
    def TEST2(self):
        return newTerminalRule("TEST2")
    
    @lexer
    def TEST3(self):
        return newTerminalRule("TEST3")
    
    @parser
    def test4(self):
        self.log("visit test4")
        return newParserRule(self.TEST3())
    
    @parser
    def concat_test(self):
        g = newParserRule()
        g_t1 = self.TEST1()
        g_t2 = self.TEST2()
        g_t3 = self.TEST3()
        g_t4 = self.test4()
        g_con = concat(g_t1, g_t2, g_t3, g_t4)
        # test __iter__
        for t in g_con:
            print(t.getText())
        # test __getitem__
        print(g_con[1].getText())
        # test __get_attr__
        assert g_con.get_attr("abc") is None
        # test __set_attr__
        try:
            g_con.set_attr("abc", 10)
            assert False
        except ProductionSemaError as e:
            print("g_con.set_attr got error:", e)
        # test visit
        g_con.visit()
        g.setProduction(g_con)
        return g
    

def test_concat():
    g = MyGram()
    lexer = g.lexer()
    parser = g.parser(lexer, "concat_test")
    tree = parser.parse("TEST1 TEST2 TEST3 TEST3")
    tree.visit()
    
test_concat()