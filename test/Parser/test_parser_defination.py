from FeGen.AttributeGrammar import *
import logging

logging.basicConfig(level=logging.DEBUG)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    @lexer
    def NUMBER(self):
        def alt1():
            return regular_expr("[0-9]")
        def alt2():
            return concat(regular_expr("[1-9]"), zero_or_more(regular_expr("[0-9]")))
        return newTerminalRule(alternate(alt2, alt1))
    
    @lexer
    def Identifier(self):
        return newTerminalRule(regular_expr("[a-zA-Z_][a-zA-Z0-9_]*"))
    
    @parser
    def two_number(self):
        return newParserRule(concat(self.NUMBER(), self.NUMBER()))
    
    @parser
    def num_or_id(self):
        alt1 = lambda: self.NUMBER()
        alt2 = lambda: self.Identifier()
        return newParserRule(alternate(alt1, alt2))
    
    @parser
    def multi_num(self):
        return newParserRule(zero_or_more(self.NUMBER()))
    
    @parser
    def multi_id(self):
        return newParserRule(one_or_more(self.Identifier()))
        
    @parser
    def opt_num(self):
        return newParserRule(zero_or_one(self.NUMBER()))
        
    
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



def test_two_number():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "two_number")
    tree = parser.parse("100 200")
    print(tree.getText())

def test_num_or_id():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "num_or_id")
    tree = parser.parse("10")
    print(tree.getText())
    tree = parser.parse("a1")
    print(tree.getText())
    
def test_multi_num():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "multi_num")
    tree = parser.parse("10 10 10 10")
    print(tree.getText())

def test_multi_num_1():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "multi_num")
    tree = parser.parse(" ")
    print(tree.getText())

def test_multi_id():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "multi_id")
    tree = parser.parse("a b c abc _ abc123 ")
    print(tree.getText())
    

def test_opt_num():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "opt_num")
    tree = parser.parse("100")
    print(tree.getText())

def test_opt_num_1():
    myg = MyGrammar()
    lexer = myg.lexer()
    parser = myg.parser(lexer, "opt_num")
    tree = parser.parse(" ")
    print(tree.getText())
    
    