from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
        
    @lexer
    def CHARSET(self):
        return newTerminalRule(char_set("[0-9]"))

    @lexer
    def ALTER(self):
        def alt1():
            return "A"
        def alt2():
            return "B"
        return newTerminalRule(alternate(alt1, alt2))
    
    @lexer
    def CONCAT(self):
        return newTerminalRule(concat("C", "D"))

    @lexer
    def PLUS(self):
        return newTerminalRule(one_or_more("E"))

    @lexer
    def STAR(self):
        return newTerminalRule(concat("F" , zero_or_more("G")))

    @lexer
    def OPT(self):
        return newTerminalRule(concat("H" , zero_or_one("I")) )



def test_CHARSET():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("1")
    print(tokens)
    
def test_ALTER():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("A")
    print(tokens)
    tokens = lexer.input("B")
    print(tokens)
    
def test_CONCAT():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("CD")
    print(tokens)

def test_PLUS():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("EEE")
    print(tokens)

def test_STAR():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("F")
    print(tokens)
    tokens = lexer.input("FGGG")
    print(tokens)

def test_OPT():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("H")
    print(tokens)
    tokens = lexer.input("HI")
    print(tokens)