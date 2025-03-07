from FeGen.AttributeGrammar import attr_grammar, FeGenGrammar, newTerminalRule, lexer

@attr_grammar
class MyGrammar(FeGenGrammar):
    def __init__(self):
        pass
    
    @lexer
    def test(self):
        g = newTerminalRule("test")
        print(g.text())
        return g

mygram = MyGrammar()
# mylexer = mygram.lexer()
# myparser = mygram.parser()
