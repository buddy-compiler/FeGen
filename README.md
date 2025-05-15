# FeGen: a Python-embedded DSL and a framework for rapid prototyping DSL Compiler

FeGen is designed to bridge syntax definition and MLIR code generation. It enables developers to define both grammatical rules and semantic bindings within FeGen DSL, streamlining the process of prototype compiler construction. 

This repository shows the implementation of FeGen.

# How to use?

1. Define Your Grammar Class

```py
from FeGen import *

class MyGrammar(FeGenGrammar):
    def __init__(self):
        super().__init__()
```

2. Add Lexer and Parser Methods to Your Class

```py
    @lexer
    def Number(self):
        """
        Number ::= [1-9][0-9]*|[0-9]
        """
        return newTerminalRule(regular_expr("[1-9][0-9]*|[0-9]"))
      
    ...
    
    @parser
    def module(self):
        """
        module: structDefine* funcDefine+
        """
        r_one_or_more = one_or_more(self.function_definition(), self.NEWLINE()) 
        r = newParserRule(r_one_or_more)
        funcs = r_one_or_more.get_attr("funcop", flatten=True)
        self.themodule = builtin.ModuleOp(funcs) # xDSL
        return r
    ...
```

3. Create Instance and Get Start

```py
...
g = MyGrammar()
mylexer = g.lexer()
myparser = g.parser(mylexer, start="module")
cstroot = myparser.parse(code)
cstroot.visit()
themodule = g.themodule
...
```