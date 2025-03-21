from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()

    @lexer
    def any_char(self):
        return newTerminalRule(regular_expr(".+"))


    
def test_CHARSET():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("()[]{}.*+?$^ *+")
    print(tokens)
    