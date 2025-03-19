from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    @lexer
    def A(self):
        g = newTerminalRule("A")
        print(g.text())
        return g

    @lexer
    def B(self):
        g = newTerminalRule("B")
        print(g.text())
        return g            
    
    @parser
    def a_and_b(self):
        g = newParserRule()
        a = self.A()
        b = self.B()
        a.text()
        g.setProduction(concat(a, b))
        return g

    @parser
    def a_or_b(self):
        g = newParserRule()
        def alt1():
            a = self.A()
            print(a.text())
            return a
        def alt2():
            return self.B()
        g.setProduction(alternate(alt1, alt2))
        return g

    @parser
    def a_star(self):
        g = newParserRule()
        g.setProduction(zero_or_more(self.A()))
        return g

    @parser
    def a_plus(self):
        g = newParserRule()
        g.setProduction(one_or_more(self.A()))
        return g
    
    @parser
    def a_or_b_plus(self):
        g = newParserRule()
        g.setProduction(one_or_more(self.a_or_b()))
        return g

    @parser
    def opt_a_and_b(self):
        return newParserRule(zero_or_one(self.a_and_b()))
        

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer, "a_or_b_plus")
code = "A B A B A B A A A"
tree = myparser.parse(code)
print(tree.getText())

# code = "A B A"
# tree = myparser.parse(code)
# print(tree.getText())