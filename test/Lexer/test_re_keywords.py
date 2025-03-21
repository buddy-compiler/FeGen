from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()

    @lexer
    def LB(self):
        return newTerminalRule("(")
    
    @lexer
    def RB(self):
        return newTerminalRule(")")
    
    @lexer
    def LP(self):
        return newTerminalRule("[")
    
    @lexer
    def RP(self):
        return newTerminalRule("]")
    
    @lexer
    def LP1(self):
        return newTerminalRule("{")

    @lexer
    def RP1(self):
        return newTerminalRule("}")
    
    @lexer
    def dot(self):
        return newTerminalRule(".")



    @lexer
    def star(self):
        return newTerminalRule("*")
    
    @lexer
    def plus(self):
        return newTerminalRule("+")
    
    @lexer
    def question_mark(self):
        return newTerminalRule("?")
    
    @lexer
    def dollar(self):
        return newTerminalRule("$")
  
    @lexer
    def power(self):
        return newTerminalRule("^")

    @lexer
    def dot_plus(self):
        return newTerminalRule(".+")
    
    @lexer
    def star_plus(self):
        return newTerminalRule(concat(self.star(), self.plus()))
    
def test_CHARSET():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("()[]{}.*+?$^ .+ *+")
    print(tokens)
    