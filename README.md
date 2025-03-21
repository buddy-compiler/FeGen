# Prepare

```bash
pip install -r requirements.txt
```

# How to define a AttributeGrammar?

1. Implement a class, for example `MyGrammar`, that inherits from the `FeGen.FeGenGrammar` class. The constructor of `MyGrammar` calls the constructor of base class.

```python
from FeGen import *

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
```

2. Implement lexer and parser method in `MyGrammar`, they should be decorated by `@lexer` and `@parser` respectively. The return value of former should be instance of `TerminalRule` created by function `newTerminalRule`, the latter should be instance of `ParserRule` created by `newParserRule`.


```python
class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()

    @lexer
    def Number(self):
        g = newTerminalRule(regular_expr("[1-9][0-9]*|[0-9]"))
        print(g.text())
        return g
      
    
    @lexer
    def Add(self):
        return newTerminalRule("+")
    
    ...
    
    @parser
    def expression(self):
        g = newParserRule()
        g_expr = self.add_expr()
        g.setProduction(g_expr)
        g.set_attr("value", g_expr.get_attr("value"))
        return g
        
    @parser
    def add_expr(self):
        g = newParserRule()
        def alt1():
            g_lhs = self.add_expr()
            g_rhs = self.prim_expr()
            lhs = g_lhs.get_attr("value")
            rhs = g_rhs.get_attr("value")
            g.set_attr("value", lhs + rhs)
            return concat(g_lhs, self.Add(), g_rhs)
        
        def alt2():
            g_prim_expr = self.prim_expr()
            g.set_attr("value", g_prim_expr.get_attr("value"))
            return g_prim_expr
        
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
...

```
    
    
3. Create `MyGrammar` instance, and get lexer, parser, input your code, generate parser tree, and get information from tree.


```python 
mygram = MyGrammar()
mylexer = mygram.lexer()
# expression is the start rule of grammar
myparser = mygram.parser(mylexer, "expression")
code = "1+(2+3)"
tree = myparser.parse(code)
print(tree.getText())
print(tree.get_attr("value"))
```

4. Total example:

For more example, see `test/`.

```python
from FeGen.AttributeGrammar import *
import logging
logging.basicConfig(level=logging.INFO)

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()

    @lexer
    def Number(self):
        g = newTerminalRule(regular_expr("[1-9][0-9]*|[0-9]"))
        print(g.text())
        return g

    @lexer
    def Identifier(self):
        g = newTerminalRule(regular_expr("[a-zA-Z_][a-zA-Z0-9_]*"))
        print(g.text())
        return g            
    
    @lexer
    def Add(self):
        return newTerminalRule("+")
    
    @lexer
    def LB(self):
        return newTerminalRule("(")
    
    @lexer
    def RB(self):
        return newTerminalRule(")")
    
    
    @parser
    def expression(self):
        g = newParserRule()
        g_expr = self.add_expr()
        g.setProduction(g_expr)
        g.set_attr("value", g_expr.get_attr("value"))
        return g
        
    @parser
    def add_expr(self):
        g = newParserRule()
        def alt1():
            g_lhs = self.add_expr()
            g_rhs = self.prim_expr()
            lhs = g_lhs.get_attr("value")
            rhs = g_rhs.get_attr("value")
            g.set_attr("value", lhs + rhs)
            return concat(g_lhs, self.Add(), g_rhs)
        
        def alt2():
            g_prim_expr = self.prim_expr()
            g.set_attr("value", g_prim_expr.get_attr("value"))
            return g_prim_expr
        
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    @parser
    def prim_expr(self):
        g = newParserRule()
        def alt1():
            g_num = self.Number()
            value = int(g_num.getText())
            g.set_attr("value", value)
            return g_num
        
        def alt2():
            g_expr = self.expression()
            value = g_expr.get_attr("value")
            g.set_attr("value", value)
            return concat(self.LB(), g_expr, self.RB())
        
        g_alt = alternate(alt1, alt2)
        g_alt.visit()
        g.setProduction(g_alt)
        return g
    
    
    

mygram = MyGrammar()
mylexer = mygram.lexer()
myparser = mygram.parser(mylexer, "expression")
code = "1+(2+3)"
tree = myparser.parse(code)
print(tree.getText())
print(tree.get_attr("value"))
```

output:

```
1 + ( 2 + 3 )
6
```