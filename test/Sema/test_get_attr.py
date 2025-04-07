from FeGen import *

class MyGram(FeGenGrammar):
    def __init__(self, output_dir_name=".fegen"):
        super().__init__(output_dir_name)
        
    @parser
    def nums(self):
        g = newParserRule()
        g_concat = concat(zero_or_more(concat(self.number(), self.Comma())), self.number())
        g.setProduction(g_concat)
        g.set_attr("v", g_concat.get_attr("v", True))
        return g
    
    @parser
    def number(self):
        g = newParserRule()
        g_num = self.NUM()
        g.set_attr("v", g_num.getText())
        g.setProduction(g_num)
        return g
    
    @lexer
    def NUM(self):
        return newTerminalRule(regular_expr("[0-9]+"))

    @lexer
    def Comma(self):
        return newTerminalRule(",")

def test():
    g = MyGram()
    myl = g.lexer()
    myp = g.parser(myl, "nums")
    tree = myp.parse("100, 200, 300")
    print(tree.get_attr("v", True))
