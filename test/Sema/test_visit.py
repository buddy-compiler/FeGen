from FeGen.AttributeGrammar import *

class MyGram(FeGenGrammar):
    def __init__(self):
        super().__init__()
    
    
    @execute_when("sema")
    def log(self, msg):
        print(msg, end="")
    
    @lexer
    def TEST(self):
        return newTerminalRule("TEST")
    
    @parser
    def rule1(self):
        self.log("visit rule1")
        t = self.TEST()
        g = newParserRule(t)
        g.set_attr("test", t.getText())
        return g


def test_visit(capsys):
    mygram = MyGram()
    lexer = mygram.lexer()
    parser = mygram.parser(lexer, "rule1")
    root = parser.parse("TEST")
    root.eval()
    root.get_attr("test")
    captured = capsys.readouterr()
    assert captured.out == "visit rule1"

    root.get_attr("test")
    captured = capsys.readouterr()
    assert captured.out == ""
    
    root.visit()
    captured = capsys.readouterr()
    assert captured.out == "visit rule1"
    
    root.visit()
    captured = capsys.readouterr()
    assert captured.out == "visit rule1"
    