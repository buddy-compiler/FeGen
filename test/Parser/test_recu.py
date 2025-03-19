from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    @lexer
    def NUMBER(self):
        def alt1():
            return char_set("[0-9]")
        def alt2():
            return concat(char_set("[1-9]"), zero_or_more(char_set("[0-9]")))
        return newTerminalRule(alternate(alt2, alt1))
    
    @lexer
    def Identifier(self):
        return newTerminalRule("[a-zA-Z_][a-zA-Z0-9_]*")
    
    @lexer
    def Dot(self):
        return newTerminalRule(".")
    
    @lexer
    def LP(self):
        return newTerminalRule("\[")
    
    @lexer
    def RP(self):
        return newTerminalRule("\]")
    
    @parser
    def variable_access(self):
        g = newParserRule()
        def name_access():
            return self.Identifier()
        def member_access():
            return concat(self.variable_access(), self.Dot(), self.Identifier())
        g.setProduction(alternate(name_access, member_access))
        return g
    
    
    @parser
    def variable_access_1(self):
        g = newParserRule()
        def name_access():
            return self.Identifier()
        def member_access():
            return concat(self.Identifier(), self.Dot(), self.variable_access_1(),)
        g.setProduction(alternate(name_access, member_access))
        return g
    
def test_NUMBER():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("100 200")
    print(tokens)

def test_Identifier():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input("a1 a2")
    print(tokens)

def test_Dot():
    myg = MyGrammar()
    lexer = myg.lexer()
    tokens = lexer.input(". .")
    print(tokens)

def test_variable_access():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "variable_access")
    tree = parser.parse("a.b.c.d")
    print(tree.getText())
    
def test_variable_access_1():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "variable_access_1")
    tree = parser.parse("a.b.c.d")
    print(tree.getText())