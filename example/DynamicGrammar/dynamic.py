from FeGen import *

class DynamicGrammar(FeGenGrammar):
    def __init__(self, isA):
        super().__init__()
        self.isA = isA
        
    @parser
    def dynamic(self):
        g = newParserRule()
        g_child = None
        if self.isA:
            g_child = self.A()
        else:
            g_child = self.B()
        g.setProduction(g_child)
        return g
    
    
    @lexer
    def A(self):
        return newTerminalRule("A")
    
    @lexer
    def B(self):
        return newTerminalRule("B")
    
def main():
    gram = DynamicGrammar(True)
    dyn_lexer = gram.lexer()
    dyn_parserA = gram.parser(dyn_lexer, "dynamic")
    treeA = dyn_parserA.parse("A")
    print(treeA.getText())
    
    gram.isA = False
    dyn_parserB = gram.parser(dyn_lexer, "dynamic")
    treeB = dyn_parserB.parse("B")
    print(treeB.getText())
    
    
main()
    