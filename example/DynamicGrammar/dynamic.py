from FeGen import *

class DynamicGrammar(FeGenGrammar):
    def __init__(self, isA, times):
        super().__init__()
        self.isA = isA
        self.times = times
    @parser
    def dynamic(self):
        g = newParserRule()
        g_child = []
        if self.isA:
            for _ in range(self.times):
                g_child.append(self.A())
        else:
            for _ in range(self.times):
                g_child.append(self.B())
        g.setProduction(*g_child)
        return g
    
    
    @lexer
    def A(self):
        return newTerminalRule("A")
    
    @lexer
    def B(self):
        return newTerminalRule("B")
    
def main():
    gram = DynamicGrammar(True, 2)
    dyn_lexer = gram.lexer()
    dyn_parserA = gram.parser(dyn_lexer, "dynamic")
    treeA = dyn_parserA.parse("A A")
    print(treeA.getText())
    
    gram.isA = False
    gram.times = 3
    dyn_parserB = gram.parser(dyn_lexer, "dynamic")
    treeB = dyn_parserB.parse("B B B")
    print(treeB.getText())

main()
    